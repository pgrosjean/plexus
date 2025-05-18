import pandas as pd
import numpy as np
import zarr
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import anndata as ad
from plexus.embedding_utils.preprocessing import aggregate_by_column
from plexus.embedding_utils.opls import OPLS
from sklearn.linear_model import LogisticRegression


def perform_opls_analysis(adata, group_key, n_folds=3, random_state=1, shuffle_y=False, shuffle_x=False, save_string=None):
    """
    Perform OPLS analysis with cross-validation on AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        Input AnnData object
    group_key : str
        Column name in adata.obs to use for classification
    n_folds : int, optional (default=3)
        Number of cross-validation folds
    random_state : int, optional (default=14)
        Random state for reproducibility
    
    Returns:
    --------
    dict containing:
        'cv_coefficients': List of coefficients from cross-validation models
        'full_model': LDA model trained on full dataset
        'top_cells': AnnData object with top/bottom 25 cells by LDA score
        'cv_accuracies': List of accuracies from cross-validation
    """
    # Create stratification ID combining location and well
    adata.obs['stratify_id'] = (adata.obs['zarr_name'] + '-' + adata.obs['well_id'].astype(str))

    # Get unique conditions
    conditions = adata.obs[group_key].unique()
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {len(conditions)}")
    
    # Subsample wells to balance conditions
    unique_wells = {cond: adata[adata.obs[group_key] == cond].obs['stratify_id'].unique() 
                   for cond in conditions}
    min_wells = min(len(wells) for wells in unique_wells.values())
    
    sampled_wells = {cond: np.random.choice(unique_wells[cond], min_wells, replace=False) for cond in conditions}
    
    num_cells_ntc_sampled = len(adata[adata.obs['stratify_id'].isin(sampled_wells['non-targeting'])])
    num_cells_cond_sampled = len(adata[adata.obs['stratify_id'].isin(sampled_wells[conditions[0]])])

    if np.abs(num_cells_ntc_sampled - num_cells_cond_sampled) > 20:
        min_wells = min_wells + 1
        while np.abs(num_cells_ntc_sampled - num_cells_cond_sampled) > 20 and min_wells < len(unique_wells['non-targeting']):
            sampled_wells['non-targeting'] = np.random.choice(unique_wells['non-targeting'], min_wells, replace=False)
            num_cells_ntc_sampled = num_cells_ntc_sampled = len(adata[adata.obs['stratify_id'].isin(sampled_wells['non-targeting'])])

    folds = {}
    np.random.seed(random_state)
    for cond in conditions:
        folds[cond] = {}
        cond_wells = list(sampled_wells[cond])
        while len(cond_wells) > 0:
            for i in range(n_folds):
                if i not in folds[cond]:
                    folds[cond][i] = []
                if len(cond_wells) > 0:
                    well = np.random.choice(cond_wells)
                    folds[cond][i].append(well)
                    cond_wells.remove(well)
    
    # Initialize cross-validation
    cv_coefficients = []
    cv_coefficients_ortho = []
    cv_aurocs = []
    for fold in range(n_folds):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for cond in conditions:
            train_idx = adata.obs['stratify_id'].isin(folds[cond][fold])
            test_idx = ~train_idx
            X_train.append(adata.X[train_idx])
            y_train.append(adata.obs[group_key].values[train_idx])
            X_test.append(adata.X[test_idx])
            y_test.append(adata.obs[group_key].values[test_idx])
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)
        if shuffle_y:
            print(np.unique(y_train, return_counts=True))
            np.random.shuffle(y_test)
            np.random.shuffle(y_train)
        if shuffle_x:
            np.random.shuffle(X_train)
            np.random.shuffle(X_test)
        # Defining OLPS model
        opls_model = OPLS(n_components=1)
        ss_train = StandardScaler()
        ss_train.fit(X_train)
        X_train = ss_train.transform(X_train)
        X_test = ss_train.transform(X_test)
        y_train = np.array([-1.0 if y == conditions[1] else 1.0 for y in y_train])
        y_test = np.array([-1.0 if y == conditions[1] else 1.0 for y in y_test])
        opls_model.fit(X_train, y_train)
        y_pred = opls_model.predict(X_test)
        beta_orth = np.zeros((X_test.shape[1],))
        # for i in range(opls_model.W_o_.shape[1]):
        #     w_o_i = opls_model.W_o_[:, i]
        #     p_o_i = opls_model.P_o_[:, i]
        #     beta_orth += w_o_i * (p_o_i.T @ w_o_i)
        cv_coefficients_ortho.append(opls_model.W_o_[:, 0])
        cv_coefficients.append(opls_model.W_p_[:, 0])
        cv_aurocs.append(roc_auc_score(y_test, y_pred))
        # Plot distributions
        T_p_test, T_o_test = opls_model.transform(X_test)
        
    all_stat_ids = []
    for cond in conditions:
        all_stat_ids.extend(folds[cond][fold])
    adata_sub = adata[adata.obs['stratify_id'].isin(all_stat_ids)]

    # Fit model on full dataset
    full_opls_model = OPLS(n_components=1)
    X_full = adata_sub.X
    y_full = np.array([-1.0 if y == conditions[1] else 1.0 for y in adata_sub.obs[group_key]])
    full_ss = StandardScaler()
    full_ss.fit(X_full)
    X_full = full_ss.transform(X_full)
    full_opls_model.fit(X_full, y_full)
    
    # Plotting the full projection of the data
    not_ntc_cond = [cond for cond in conditions if cond != 'non-targeting']
    if not shuffle_y and not shuffle_x:
        plt.figure(figsize=(4, 4))
        palette = ["#9c9a97", "#9e4928"]
        palette = sns.color_palette(palette)
        plt.rcParams.update({'font.size': 8, 'font.family': 'Sans-Serif'})
        hue_order = ['non-targeting', not_ntc_cond[0]]
        X_p_full, X_o_full = full_opls_model.transform(X_full)
        sns.kdeplot(x=X_p_full.ravel(), hue=adata_sub.obs[group_key],
                fill=True, bw_adjust=0.8, alpha=0.3, palette=palette, hue_order=hue_order)
        
        plt.xlabel('OPLS Predictive Score')
        path_to_save = './'
        if save_string is None:
            path_to_save = path_to_save + f'{not_ntc_cond[0]}_vs_non_targeting_opls_projection.pdf'
            plt.title('OPLS Projection')
        else:
            plt.title(f'OPLS Predictive Score ({save_string})')
            path_to_save = path_to_save + f'{not_ntc_cond[0]}_vs_non_targeting_opls_projection_{save_string}.pdf'
        plt.tight_layout()
        plt.savefig(path_to_save , dpi=800)
        plt.show()

    # Finding the top/bottom 25 cells by opls score to return the adata
    opls_scores, _ = np.array(full_opls_model.transform(adata_sub.X))

    # Printing how many cells are in each group
    for cond in conditions:
        print(f'{cond}: {np.sum(adata_sub.obs[group_key] == cond)}')
    
    # Finding cell indexes with the top scores
    opls_scores_ntc = opls_scores[adata_sub.obs[group_key] == 'non-targeting']
    indices_ntc = np.arange(len(opls_scores))[adata_sub.obs[group_key] == 'non-targeting']
    opls_scores_cond = opls_scores[adata_sub.obs[group_key] == not_ntc_cond[0]]
    indices_cond = np.arange(len(opls_scores))[adata_sub.obs[group_key] == not_ntc_cond[0]]
    
    # Want top 25 cells by OPLS score NTC
    # Wamt bottom 25 cells by OPLS score non-NTC
    top_cells_idx_ntc = np.argsort(opls_scores_ntc.ravel())[-15:]
    bottom_cells_idx_cond = np.argsort(opls_scores_cond.ravel())[:15]

    # indexing the top cells from total dataset
    top_cells_idx_ntc = indices_ntc[top_cells_idx_ntc]
    bottom_cells_idx_cond = indices_cond[bottom_cells_idx_cond]

    top_cell_idx = top_cells_idx_ntc
    bottom_cell_idx = bottom_cells_idx_cond
    all_top_cells = adata_sub[top_cell_idx]
    bottom_cells = adata_sub[bottom_cell_idx]
    all_top_cells = all_top_cells.concatenate(bottom_cells)

    return {
        'cv_coefficients': cv_coefficients,
        'full_coefficients': full_opls_model.coef_,
        'cv_coefficients_ortho': cv_coefficients_ortho,
        'full_model': full_opls_model,
        'top_cells': all_top_cells,
        'cv_aurocs': cv_aurocs
    }


def perform_logreg_analysis(adata, group_key, n_folds=3, random_state=1, shuffle_y=False, shuffle_x=False, save_string=None):
    """
    Perform logistic regression with elastic net regularization using manual CV on AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    group_key : str
        Column name in adata.obs to use for binary classification.
    n_folds : int, optional
        Number of cross-validation folds. Default is 3.
    random_state : int, optional
        Random seed for reproducibility. Default is 1.
    shuffle_y : bool, optional
        Whether to shuffle labels during training. Default is False.
    shuffle_x : bool, optional
        Whether to shuffle features during training. Default is False.
    save_string : str or None
        If provided, used to name the saved plot file.

    Returns
    -------
    dict
        Contains model coefficients, cross-validation scores, trained model, and selected top cells.
    """
    adata.obs['stratify_id'] = adata.obs['zarr_name'] + '-' + adata.obs['well_id'].astype(str)

    # Ensure binary classification
    conditions = sorted(adata.obs[group_key].unique())
    print(conditions)
    if len(conditions) != 2:
        raise ValueError(f"Expected exactly 2 conditions, got {len(conditions)}")

    unique_wells = {cond: adata[adata.obs[group_key] == cond].obs['stratify_id'].unique() for cond in conditions}
    min_wells = min(len(wells) for wells in unique_wells.values())
    sampled_wells = {cond: np.random.choice(unique_wells[cond], min_wells, replace=False) for cond in conditions}

    folds = {}
    np.random.seed(random_state)
    for cond in conditions:
        folds[cond] = {}
        cond_wells = list(sampled_wells[cond])
        while cond_wells:
            for i in range(n_folds):
                if i not in folds[cond]:
                    folds[cond][i] = []
                if cond_wells:
                    well = np.random.choice(cond_wells)
                    folds[cond][i].append(well)
                    cond_wells.remove(well)

    # Cross-validation loop
    cv_coefficients = []
    cv_aurocs = []

    for fold in range(n_folds):
        X_train, y_train, X_test, y_test = [], [], [], []
        for cond in conditions:
            is_train = adata.obs['stratify_id'].isin(folds[cond][fold])
            is_test = ~is_train
            X_train.append(adata.X[is_train])
            y_train.append(adata.obs[group_key].values[is_train])
            X_test.append(adata.X[is_test])
            y_test.append(adata.obs[group_key].values[is_test])

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        if shuffle_y:
            np.random.shuffle(y_train)
            np.random.shuffle(y_test)
        if shuffle_x:
            np.random.shuffle(X_train)
            np.random.shuffle(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train_bin = (y_train == conditions[1]).astype(int)
        y_test_bin = (y_test == conditions[1]).astype(int)

        logreg = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=1.0,
            max_iter=1000,
            random_state=random_state
        )
        logreg.fit(X_train, y_train_bin)
        y_pred_proba = logreg.predict_proba(X_test)[:, 1]

        cv_coefficients.append(logreg.coef_.ravel())
        cv_aurocs.append(roc_auc_score(y_test_bin, y_pred_proba))

    # Use all stratify_id wells from last fold for full model
    all_stat_ids = []
    for cond in conditions:
        all_stat_ids.extend(folds[cond][fold])
    adata_sub = adata[adata.obs['stratify_id'].isin(all_stat_ids)]

    # Full model training
    X_full = adata_sub.X
    y_full_bin = (adata_sub.obs[group_key] == conditions[1]).astype(int)
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X_full)

    full_model = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        C=1.0,
        max_iter=1000,
        random_state=random_state
    )
    full_model.fit(X_full, y_full_bin)

    scores = full_model.decision_function(X_full)

    # Plotting
    if not shuffle_y and not shuffle_x:
        plt.figure(figsize=(4, 4))
        palette = sns.color_palette(["#9c9a97", "#9e4928"])
        not_ntc_cond = [c for c in conditions if c != 'non-targeting'][0]
        hue_order = ['non-targeting', not_ntc_cond]
        sns.kdeplot(x=scores, hue=adata_sub.obs[group_key], fill=True, bw_adjust=0.8,
                    alpha=0.3, palette=palette, hue_order=hue_order)
        plt.xlabel('Logistic Regression Score')
        title = f'Logistic Regression Score ({save_string})' if save_string else 'Logistic Regression Projection'
        plt.title(title)
        save_path = f'./{not_ntc_cond}_vs_non_targeting_logreg_projection'
        if save_string:
            save_path += f'_{save_string}'
        save_path += '.pdf'
        plt.tight_layout()
        plt.savefig(save_path, dpi=800)
        plt.show()

    # Top/bottom 25 cell selection
    not_ntc = [cond for cond in conditions if cond != 'non-targeting'][0]
    score_vals = scores.ravel()
    ntc_scores = score_vals[adata_sub.obs[group_key] == 'non-targeting']
    ntc_indices = np.where(adata_sub.obs[group_key] == 'non-targeting')[0]
    cond_scores = score_vals[adata_sub.obs[group_key] == not_ntc]
    cond_indices = np.where(adata_sub.obs[group_key] == not_ntc)[0]

    top_ntc_idx = ntc_indices[np.argsort(ntc_scores)[-15:]]
    bottom_cond_idx = cond_indices[np.argsort(cond_scores)[:15]]

    all_top_cells = adata_sub[top_ntc_idx].concatenate(adata_sub[bottom_cond_idx])

    return {
        'cv_coefficients': cv_coefficients,
        'full_coefficients': full_model.coef_.ravel(),
        'full_model': full_model,
        'top_cells': all_top_cells,
        'cv_aurocs': cv_aurocs
    }



def plot_calcium_traces(gcamp_signal: np.ndarray,
                        plot_color: str,
                        cond: str,
                        save_string: str,
                        fs: float = 25.0):
    time = np.arange(gcamp_signal.shape[1]) / fs
    plt.figure(figsize=(4, 4))
    sig_max = 0.8
    print(np.amax(gcamp_signal))
    for c, cell in enumerate(gcamp_signal):
        cell_scaled = (cell / sig_max) + c
        plt.plot(time, cell_scaled, color='k', linewidth=0.5)
        plt.fill_between(time, c, cell_scaled, color=plot_color, alpha=.4)
    
    # Change the x-axis to be in seconds
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Number')
    plt.tight_layout()
    plt.savefig(f'./{cond}_top_cells_{save_string}_OPLS.pdf', dpi=300)
    plt.show()


def plot_top_cells(adata, cond_key, zarr_root_dict, save_string="WTC11"):
    """
    Plot top/bottom cells by LDA score.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with 'lda_score' in obs
    n_cells : int, optional (default=25)
        Number of top/bottom cells to plot
    """
    for cond in adata.obs[cond_key].unique():
        sc_signals = []
        adata_cond = adata[adata.obs[cond_key] == cond]
        for cell_idx in range(adata_cond.shape[0]):
            zarr_name = adata_cond.obs['zarr_name'].values[cell_idx]
            well_id = adata_cond.obs['well_id'].values[cell_idx]
            fov_id = adata_cond.obs['fov_id'].values[cell_idx]
            sc_idx = adata_cond.obs['fov_cell_idx'].values[cell_idx]
            root = zarr_root_dict[zarr_name]
            gcamp_signal = np.array(root[f'well_Well{well_id}'][fov_id]['raw_signal'])
            sc_gcamp_signal = gcamp_signal[sc_idx]
            sc_gcamp_signal = (sc_gcamp_signal - np.min(sc_gcamp_signal))/np.amin(sc_gcamp_signal)
            sc_signals.append(sc_gcamp_signal)

        sc_signals = np.vstack(sc_signals)
        if cond == 'non-targeting':
            plot_color = "#9c9a97"
        else:
            plot_color = "#9e4928"
        plot_calcium_traces(sc_signals, plot_color=plot_color, cond=cond, save_string=save_string)


def get_zarr_info(zarr_location):
    """
    Walk through the directory tree and save any directories ending in .zarr.
    
    Parameters
    ----------
    zarr_location : str
        The root directory to search for .zarr files.
    
    Returns
    -------
    dict
        A dictionary where the keys are the base names of .zarr directories 
        and the values are the opened zarr objects.
    """
    root_dict = {}
    # Walk through the directory tree
    for root, dirs, files in os.walk(zarr_location):
        for directory in dirs:
            if directory.endswith('.zarr'):
                # Extract the base name without the .zarr extension
                base_name = directory.split('.')[0]
                # Get the full path to the .zarr directory
                zarr_path = os.path.join(root, directory)
                print(f'Opening {zarr_path}')
                # Open the .zarr directory and store it in the dictionary
                root_dict[base_name] = zarr.open(zarr_path, 'r')
    return root_dict


def main():
    base_loc = '/../../plexus_data_archive'

    # Setting up the plotting style
    wtc11_wt_color = "#5D8195"
    wtc11_mut_color = "#FFC20A"
    patient_wt_color = "#35B5FD"
    patient_mut_color = "#FF7C0A"
    # making sns color palette from list of hex colors
    palette = [wtc11_wt_color, wtc11_mut_color, patient_wt_color, patient_mut_color]
    palette = sns.color_palette(palette)
    hue_order = ['WT-WTC11', 'Mutant-WTC11', 'WT-Patient Line', 'Mutant-Patient Line']

    # Reading in the single-cell h5ad files WTC11 WT and Patient WT
    adata_wt_wtc11_sc = ad.read_h5ad(f'{base_loc}/plexus_embeddings/crispri_screen/tvn_corrected_embeddings/single_cell_tvn_corrected_embeddings_WT_WTC11_CRISPRi.h5ad')
    zarr_name_base_wtc11 = adata_wt_wtc11_sc.obs['zarr_name'].apply(lambda x: '_'.join(x.split('_')[:-3])).astype('str').values
    fov_ids = adata_wt_wtc11_sc.obs['fov_id'].astype('str').values
    well_ids = adata_wt_wtc11_sc.obs['well_id'].astype('str').values
    cell_ids = adata_wt_wtc11_sc.obs['fov_cell_idx'].astype('str').values
    adata_wt_wtc11_sc.obs['aggregate_by'] = zarr_name_base_wtc11 + '-' + well_ids + '-' + fov_ids + '-' + cell_ids
    adata_wt_wtc11_sc = aggregate_by_column(adata_wt_wtc11_sc, 'aggregate_by')
    adata_wt_wtc11_sc.obs['zarr_name'] = adata_wt_wtc11_sc.obs['aggregate_by'].apply(lambda x: x.split('-')[0])
    scaler_wtc11 = StandardScaler()
    scaler_wtc11.fit(adata_wt_wtc11_sc.X)
    adata_wt_wtc11_sc.X = scaler_wtc11.transform(adata_wt_wtc11_sc.X)

    adata_wt_patient_sc = ad.read_h5ad(f'{base_loc}/plexus_embeddings/crispri_screen/tvn_corrected_embeddings/single_cell_tvn_corrected_embeddings_WT_Patient_Line_CRISPRi.h5ad')
    zarr_name_base = adata_wt_patient_sc.obs['zarr_name'].apply(lambda x: '_'.join(x.split('_')[:-3])).astype('str').values
    fov_ids = adata_wt_patient_sc.obs['fov_id'].astype('str').values
    well_ids = adata_wt_patient_sc.obs['well_id'].astype('str').values
    cell_ids = adata_wt_patient_sc.obs['fov_cell_idx'].astype('str').values
    adata_wt_patient_sc.obs['aggregate_by'] = zarr_name_base + '-' + well_ids + '-' + fov_ids + '-' + cell_ids
    adata_wt_patient_sc = aggregate_by_column(adata_wt_patient_sc, 'aggregate_by')
    adata_wt_patient_sc.obs['zarr_name'] = adata_wt_patient_sc.obs['aggregate_by'].apply(lambda x: x.split('-')[0])
    scaler_patient = StandardScaler()
    scaler_patient.fit(adata_wt_patient_sc.X)
    adata_wt_patient_sc.X = scaler_patient.transform(adata_wt_patient_sc.X)

    # setting the plt style
    adata_wt_wtc11_sc_opls_kcnq2 = adata_wt_wtc11_sc[adata_wt_wtc11_sc.obs['gene_id'].isin(['KCNQ2', 'non-targeting'])]
    adata_wt_patient_sc_opls_kcnq2 = adata_wt_patient_sc[adata_wt_patient_sc.obs['gene_id'].isin(['KCNQ2', 'non-targeting'])]

    results_wtc11_kcnq2 = perform_opls_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, save_string='WTC11')
    results_patient_kcnq2 = perform_opls_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, save_string='Patient')

    results_wtc11_kcnq2_shuffle_y = perform_opls_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_y=True)
    results_patient_kcnq2_shuffle_y = perform_opls_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_y=True)

    results_wtc11_kcnq2_shuffle_x = perform_opls_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_x=True)
    results_patient_kcnq2_shuffle_x = perform_opls_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_x=True)

    results_lr_wtc11_kcnq2 = perform_logreg_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, save_string='WTC11')
    results_lr_patient_kcnq2 = perform_logreg_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, save_string='Patient')
    results_lr_wtc11_kcnq2_shuffle_y = perform_logreg_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_y=True)
    results_lr_patient_kcnq2_shuffle_y = perform_logreg_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_y=True)
    results_lr_wtc11_kcnq2_shuffle_x = perform_logreg_analysis(adata_wt_wtc11_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_x=True)
    results_lr_patient_kcnq2_shuffle_x = perform_logreg_analysis(adata_wt_patient_sc_opls_kcnq2, group_key='gene_id', n_folds=3, shuffle_x=True)
    
    # Plotting the accuracies on a barplot with CV shown per cell line with and without y shuffling
    all_aurocs_no_shuffle = results_wtc11_kcnq2['cv_aurocs'] + results_patient_kcnq2['cv_aurocs']
    all_aurocs_shuffle = results_wtc11_kcnq2_shuffle_y['cv_aurocs'] + results_patient_kcnq2_shuffle_y['cv_aurocs']
    all_aurocs_shuffle_x = results_wtc11_kcnq2_shuffle_x['cv_aurocs'] + results_patient_kcnq2_shuffle_x['cv_aurocs']
    all_labels_no_shuffle = ['WTC11'] * 3 + ['Patient'] * 3
    all_labels_shuffle = ['WTC11'] * 3 + ['Patient'] * 3
    all_labels_shuffle_x = ['WTC11'] * 3 + ['Patient'] * 3

    aurocs_df = pd.DataFrame({'cell_line': all_labels_no_shuffle + all_labels_shuffle + all_labels_shuffle_x,
                                'shuffled': ['OPLS model'] * 6 + ['y shuffled'] * 6 + ['x shuffled'] * 6,
                                    'AUROC': all_aurocs_no_shuffle + all_aurocs_shuffle + all_aurocs_shuffle_x})

    palette = [patient_wt_color, wtc11_wt_color]
    palette = sns.color_palette(palette)
    hue_order = ['Patient', 'WTC11']
    plt.figure(figsize=(4, 4))
    # setting the font size and type to be arial size 8
    sns.barplot(data=aurocs_df, hue='cell_line', y='AUROC', x='shuffled', palette=palette, hue_order=hue_order)
    # plotting the dots
    sns.stripplot(data=aurocs_df, hue='cell_line', y='AUROC', x='shuffled', hue_order=hue_order, dodge=True, color='black', legend=False)
    # plotting horizontal line at 0.5
    plt.axhline(0.5, color='black', linestyle='--')
    plt.title('OPLS Cross-Validation AUROCs')
    plt.legend()
    plt.ylabel('AUROC')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('./opls_cross_validation_aurocs.pdf', dpi=300)
    plt.show()

    # Plotting the AUROCs for the logistic regression model
    all_aurocs_lr_no_shuffle = results_lr_wtc11_kcnq2['cv_aurocs'] + results_lr_patient_kcnq2['cv_aurocs']
    all_aurocs_lr_shuffle = results_lr_wtc11_kcnq2_shuffle_y['cv_aurocs'] + results_lr_patient_kcnq2_shuffle_y['cv_aurocs']
    all_aurocs_lr_shuffle_x = results_lr_wtc11_kcnq2_shuffle_x['cv_aurocs'] + results_lr_patient_kcnq2_shuffle_x['cv_aurocs']
    all_labels_lr_no_shuffle = ['WTC11'] * 3 + ['Patient'] * 3
    all_labels_lr_shuffle = ['WTC11'] * 3 + ['Patient'] * 3
    all_labels_lr_shuffle_x = ['WTC11'] * 3 + ['Patient'] * 3
    aurocs_lr_df = pd.DataFrame({'cell_line': all_labels_lr_no_shuffle + all_labels_lr_shuffle + all_labels_lr_shuffle_x,
                                'shuffled': ['Logistic model'] * 6 + ['y shuffled'] * 6 + ['x shuffled'] * 6,
                                    'AUROC': all_aurocs_lr_no_shuffle + all_aurocs_lr_shuffle + all_aurocs_lr_shuffle_x})
    plt.figure(figsize=(4, 4))
    sns.barplot(data=aurocs_lr_df, hue='cell_line', y='AUROC', x='shuffled', palette=palette, hue_order=hue_order)
    sns.stripplot(data=aurocs_lr_df, hue='cell_line', y='AUROC', x='shuffled', hue_order=hue_order, dodge=True, color='black', legend=False)
    plt.axhline(0.5, color='black', linestyle='--')
    plt.title('Cross-Validation AUROCs')
    plt.legend()
    plt.ylabel('AUROC')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('./logreg_cross_validation_aurocs.pdf', dpi=300)

    zarr_dict = get_zarr_info(f'{base_loc}/processed_zarr_files/crispri_screen/full_zarr_files/')
    plot_top_cells(results_wtc11_kcnq2['top_cells'], 'gene_id', zarr_dict, save_string='WTC11')
    plot_top_cells(results_patient_kcnq2['top_cells'], 'gene_id', zarr_dict, save_string='Patient')
    plot_top_cells(results_lr_wtc11_kcnq2['top_cells'], 'gene_id', zarr_dict, save_string='WTC11_LR')
    plot_top_cells(results_lr_patient_kcnq2['top_cells'], 'gene_id', zarr_dict, save_string='Patient_LR')

    ### FOR WTC11 WT
    # Creating the adata frame for single cell embeddings
    zarr_name_base = adata_wt_wtc11_sc.obs['zarr_name'].apply(lambda x: '_'.join(x.split('_')[:-3])).astype('str').values
    fov_ids = adata_wt_wtc11_sc.obs['fov_id'].astype('str').values
    well_ids = adata_wt_wtc11_sc.obs['well_id'].astype('str').values
    cell_ids = adata_wt_wtc11_sc.obs['fov_cell_idx'].astype('str').values
    adata_wt_wtc11_sc.obs['aggregate_by'] = adata_wt_wtc11_sc.obs.index
    # Grouping the adata by the aggregate_by key
    grouped_adata_wt_wtc11_sc = adata_wt_wtc11_sc

    ### FOR PATIENT WT
    # Creating the adata frame for single cell embeddings
    zarr_name_base = adata_wt_patient_sc.obs['zarr_name'].apply(lambda x: '_'.join(x.split('_')[:-3])).astype('str').values
    fov_ids = adata_wt_patient_sc.obs['fov_id'].astype('str').values
    well_ids = adata_wt_patient_sc.obs['well_id'].astype('str').values
    cell_ids = adata_wt_patient_sc.obs['fov_cell_idx'].astype('str').values
    adata_wt_patient_sc.obs['aggregate_by'] = adata_wt_patient_sc.obs.index
    # Grouping the adata by the aggregate_by key
    grouped_adata_wt_patient_sc = adata_wt_patient_sc

    # Reading in the manual features
    param_adata = ad.read_h5ad(f'{base_loc}/plexus_embeddings/crispri_screen/crispri_screen_manual_features.h5ad')
    scaler = StandardScaler()
    param_adata.X = scaler.fit_transform(param_adata.X)
    param_adata.obs['mapping_id'] = param_adata.obs['zarr_file'].astype(str) + '-' + param_adata.obs['well_id'].apply(lambda x: x[-3:]).astype(str) + '-' + param_adata.obs['fov_id'].astype(str) + '-' + param_adata.obs['fov_cell_idx'].astype(str)

    # Finding the mappings between the manual features and the parameters
    unique_mappings_param = param_adata.obs['mapping_id'].unique()
    unique_mappings_embed_wtc11 = grouped_adata_wt_wtc11_sc.obs['aggregate_by'].unique()
    unique_mappings_embed_patient = grouped_adata_wt_patient_sc.obs['aggregate_by'].unique()
    unique_mappings_both_wtc11 = np.intersect1d(unique_mappings_param, unique_mappings_embed_wtc11)
    unique_mappings_both_patient = np.intersect1d(unique_mappings_param, unique_mappings_embed_patient)
    grouped_adata_wt_wtc11_sc = grouped_adata_wt_wtc11_sc[grouped_adata_wt_wtc11_sc.obs['aggregate_by'].isin(unique_mappings_both_wtc11)]
    grouped_adata_wt_patient_sc = grouped_adata_wt_patient_sc[grouped_adata_wt_patient_sc.obs['aggregate_by'].isin(unique_mappings_both_patient)]
    param_adata_wtc11 = param_adata[param_adata.obs['mapping_id'].isin(unique_mappings_both_wtc11)]
    param_adata_patient = param_adata[param_adata.obs['mapping_id'].isin(unique_mappings_both_patient)]

    # Reordering param_adata sorting by mapping_id
    param_adata_wtc11 = param_adata_wtc11[param_adata_wtc11.obs.sort_values(by='mapping_id').index, :]
    grouped_adata_wt_wtc11_sc = grouped_adata_wt_wtc11_sc[grouped_adata_wt_wtc11_sc.obs.sort_index().index, :]
    param_adata_patient = param_adata_patient[param_adata_patient.obs.sort_values(by='mapping_id').index, :]
    grouped_adata_wt_patient_sc = grouped_adata_wt_patient_sc[grouped_adata_wt_patient_sc.obs.sort_index().index, :]

    correlations_wtc11 = np.cov(grouped_adata_wt_wtc11_sc.X.T, param_adata_wtc11.X.T)[:768, 768:]
    correlations_patient = np.cov(grouped_adata_wt_patient_sc.X.T, param_adata_patient.X.T)[:768, 768:]

    mean_opls_coefficients_wtc11 = np.mean(np.vstack(results_wtc11_kcnq2['cv_coefficients']), axis=0)
    mean_opls_coefficients_patient = np.mean(np.vstack(results_patient_kcnq2['cv_coefficients']), axis=0)

    mean_opls_orth_coefficients_wtc11 = np.mean(np.vstack(results_wtc11_kcnq2['cv_coefficients_ortho']), axis=0)
    mean_opls_orth_coefficients_patient = np.mean(np.vstack(results_patient_kcnq2['cv_coefficients_ortho']), axis=0)

    mean_lr_coefficients_wtc11 = np.mean(np.vstack(results_lr_wtc11_kcnq2['cv_coefficients']), axis=0)
    mean_lr_coefficients_patient = np.mean(np.vstack(results_lr_patient_kcnq2['cv_coefficients']), axis=0)

    opls_vector_wtc11 = mean_opls_coefficients_wtc11.ravel()
    projected_eigenvector_wtc11 = np.dot(correlations_wtc11.T, opls_vector_wtc11)
    projected_eigenvector_wtc11 = projected_eigenvector_wtc11 / np.linalg.norm(projected_eigenvector_wtc11)

    opls_vector_patient = mean_opls_coefficients_patient.ravel()
    projected_eigenvector_patient = np.dot(correlations_patient.T, opls_vector_patient)
    projected_eigenvector_patient = projected_eigenvector_patient / np.linalg.norm(projected_eigenvector_patient)

    feature_names = param_adata_wtc11.var.index.values
    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=projected_eigenvector_patient, alpha=0.4, color=patient_wt_color, label='Patient')
    sns.barplot(x=feature_names, y=projected_eigenvector_wtc11, alpha=0.4, color=wtc11_wt_color, label='WTC11')
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Projected OPLS Coefficients')
    plt.tight_layout()
    plt.savefig('./opls_coefficients_projected_importance.pdf', dpi=800)
    plt.show()

    # Plotting the projections for the LR
    lr_vector_wtc11 = mean_lr_coefficients_wtc11.ravel()
    projected_eigenvector_lr_wtc11 = np.dot(correlations_wtc11.T, lr_vector_wtc11)
    projected_eigenvector_lr_wtc11 = projected_eigenvector_lr_wtc11 / np.linalg.norm(projected_eigenvector_lr_wtc11)
    lr_vector_patient = mean_lr_coefficients_patient.ravel()
    projected_eigenvector_lr_patient = np.dot(correlations_patient.T, lr_vector_patient)
    projected_eigenvector_lr_patient = projected_eigenvector_lr_patient / np.linalg.norm(projected_eigenvector_lr_patient)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=projected_eigenvector_lr_patient, alpha=0.4, color=patient_wt_color, label='Patient')
    sns.barplot(x=feature_names, y=projected_eigenvector_lr_wtc11, alpha=0.4, color=wtc11_wt_color, label='WTC11')
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('./logistic_coefficients_projected_importance.pdf', dpi=800)
    plt.show()


if __name__ == '__main__':
    main()
