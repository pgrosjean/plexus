import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests
import anndata as ad
from plexus.embedding_utils.opls import OPLS


def main():
    # Setting up the plotting style
    wtc11_wt_color = "#5D8195"
    wtc11_mut_color = "#FFC20A"
    patient_wt_color = "#35B5FD"
    patient_mut_color = "#FF7C0A"
    # making sns color palette from list of hex colors
    palette = [wtc11_wt_color, wtc11_mut_color, patient_wt_color, patient_mut_color]
    palette = sns.color_palette(palette)
    hue_order = ['WT-WTC11', 'Mutant-WTC11', 'WT-Patient Line', 'Mutant-Patient Line']

    # Plotting the PCA of the embedding data per cell line
    adata_crispri = ad.read_h5ad('/../../plexus_data_archive/plexus_embeddings/crispri_screen/crispri_screen_plexus_embeddings.h5ad')
    zarr_name_base = adata_crispri.obs['location_id'].apply(lambda x: x.split('-')[0]).astype('str').values
    well_ids = adata_crispri.obs['well_id'].astype('str').values
    fov_ids = adata_crispri.obs['fov_id'].astype('str').values
    cell_ids = adata_crispri.obs['fov_cell_idx'].astype('str').values
    adata_crispri.obs['aggregate_by'] = zarr_name_base + '-' + well_ids + '-' + fov_ids + '-' + cell_ids
    adata_crispri = adata_crispri[adata_crispri.obs['aggregate_by'].sort_values().index, :]
    adata_crispri.obs.index = adata_crispri.obs['aggregate_by'].values.copy()
    
    adata_wtc11 = adata_crispri[adata_crispri.obs['donor_id'] == 'WTC11', :]
    adata_patient = adata_crispri[adata_crispri.obs['donor_id'] == 'Patient Line', :]
    scaler_wtc11 = StandardScaler()
    scaler_patient = StandardScaler()
    adata_wtc11_ntc = adata_wtc11[adata_wtc11.obs['guide_id'].str.contains('non-targeting'), :]
    adata_patient_ntc = adata_patient[adata_patient.obs['guide_id'].str.contains('non-targeting'), :]
    X_wtc11_ntc = scaler_wtc11.fit_transform(adata_wtc11_ntc.X)
    X_patient_ntc = scaler_patient.fit_transform(adata_patient_ntc.X)
    pca_wtc11 = PCA(n_components=2)
    pca_patient = PCA(n_components=2)
    pca_wtc11.fit(X_wtc11_ntc)
    pca_patient.fit(X_patient_ntc)
    X_wtc11_pca_ntc = pca_wtc11.transform(X_wtc11_ntc)
    X_patient_pca_ntc = pca_patient.transform(X_patient_ntc)
    # plotting the PCA for WTC11
    plt.figure(figsize=(4, 4))
    wtc11_palette = [wtc11_wt_color, wtc11_mut_color]
    wtc11_hue_order = ['WT-WTC11', 'Mutant-WTC11']
    sns.scatterplot(x=X_wtc11_pca_ntc[:, 0],
                    y=X_wtc11_pca_ntc[:, 1],
                    hue=adata_wtc11_ntc.obs['cell_line'],
                    palette=wtc11_palette,
                    hue_order=wtc11_hue_order,
                    s=10,
                    alpha=1)
    plt.xlabel(f'PC1 ({pca_wtc11.explained_variance_ratio_[0] * 100:.2f}%)')
    plt.ylabel(f'PC2 ({pca_wtc11.explained_variance_ratio_[1] * 100:.2f}%)')
    # Increasing the y axis limits to show the legend inside the plot
    current_ylims = plt.gca().get_ylim()
    # expand the y axis limit in the upper direction by 10%
    plt.gca().set_ylim(current_ylims[0], current_ylims[1] * 1.3)
    plt.legend(loc='upper left')
    plt.title('Non-targeting Controls\nWTC11')
    plt.tight_layout()
    plt.savefig('./PCA_ntc_WTC11_figure5.pdf', dpi=800)
    plt.show()
    # plotting the PCA for Patient Line
    plt.figure(figsize=(4, 4))
    patient_palette = [patient_wt_color, patient_mut_color]
    patient_hue_order = ['WT-Patient Line', 'Mutant-Patient Line']
    sns.scatterplot(x=X_patient_pca_ntc[:, 0],
                    y=X_patient_pca_ntc[:, 1],
                    hue=adata_patient_ntc.obs['cell_line'],
                    palette=patient_palette,
                    hue_order=patient_hue_order,
                    s=10,
                    alpha=1)
    plt.xlabel(f'PC1 ({pca_patient.explained_variance_ratio_[0] * 100:.2f}%)')
    plt.ylabel(f'PC2 ({pca_patient.explained_variance_ratio_[1] * 100:.2f}%)')
    # Increasing the y axis limits to show the legend inside the plot
    current_ylims = plt.gca().get_ylim()
    # expand the y axis limit in the upper direction by 10%
    plt.gca().set_ylim(current_ylims[0], current_ylims[1] * 1.3)
    plt.legend(loc='upper left')
    plt.title('Non-targeting Controls\nPatient Line')
    plt.tight_layout()
    plt.savefig('./PCA_ntc_Patient_figure5.pdf', dpi=800)
    plt.show()

    # OPLS analysis
    # Creating the OPLS model for the WTC11 data
    train_indices = []
    test_indices = []
    adata_wtc11_ntc.obs['strat_id'] = adata_wtc11_ntc.obs['aggregate_by'].apply(lambda x: '-'.join(x.split('-')[:2]))
    for cond in adata_wtc11_ntc.obs['cell_line'].unique():
        cond_adata = adata_wtc11_ntc[adata_wtc11_ntc.obs['cell_line'] == cond, :]
        unq_cond_strat = cond_adata.obs['strat_id'].unique()
        # Splitting the data into train and test
        train_strat, test_strat = train_test_split(unq_cond_strat, test_size=0.5, random_state=14)
        train_idx = np.arange(len(adata_wtc11_ntc))[adata_wtc11_ntc.obs['strat_id'].isin(train_strat).values]
        val_idx = np.arange(len(adata_wtc11_ntc))[adata_wtc11_ntc.obs['strat_id'].isin(test_strat).values]
        train_indices.append(train_idx)
        test_indices.append(val_idx)

    adata_wtc11_ntc_train = adata_wtc11_ntc[np.concatenate(train_indices), :]
    adata_wtc11_ntc_test = adata_wtc11_ntc[np.concatenate(test_indices), :]

    opls_wtc11 = OPLS(n_components=1)
    X_train = adata_wtc11_ntc_train.X

    scaler_wtc11 = StandardScaler()
    X_train = scaler_wtc11.fit_transform(X_train)

    y_train = adata_wtc11_ntc_train.obs['cell_line'].values
    y_train = np.array([-10 if x == 'WT-WTC11' else 10 for x in y_train]).astype(float)
    X_test = adata_wtc11_ntc_test.X
    X_test = scaler_wtc11.transform(X_test)
    y_test = adata_wtc11_ntc_test.obs['cell_line'].values
    y_test = np.array([-10 if x == 'WT-WTC11' else 10 for x in y_test]).astype(float)

    # Balancing the classes in the training data
    num_to_sample = np.min(np.unique(y_train, return_counts=True)[1])
    idx_train_wt = np.arange(len(y_train))[y_train == -10]
    idx_train_mut = np.arange(len(y_train))[y_train == 10]
    np.random.seed(42)
    idx_train_wt = np.random.choice(idx_train_wt, size=num_to_sample, replace=False)
    idx_train_mut = np.random.choice(idx_train_mut, size=num_to_sample, replace=False)
    idx_train_balanced = np.concatenate([idx_train_wt, idx_train_mut])
    X_train_balanced = X_train[idx_train_balanced, :]
    y_train_balanced = y_train[idx_train_balanced]

    # Balancing the classes in the validation data
    num_to_sample = np.min(np.unique(y_test, return_counts=True)[1])
    idx_test_wt = np.arange(len(y_test))[y_test == -10]
    idx_test_mut = np.arange(len(y_test))[y_test == 10]
    np.random.seed(42)
    idx_test_wt = np.random.choice(idx_test_wt, size=num_to_sample, replace=False)
    idx_test_mut = np.random.choice(idx_test_mut, size=num_to_sample, replace=False)
    idx_test_balanced = np.concatenate([idx_test_wt, idx_test_mut])
    X_test_balanced = X_test[idx_test_balanced, :]
    y_test_balanced = y_test[idx_test_balanced]

    opls_wtc11.fit(X_train_balanced, y_train_balanced)
    T_p_test_wtc11, T_o_test_wtc11 = opls_wtc11.transform(X_test_balanced)

    # Setting up the plotting style
    wtc11_wt_color = "#5D8195"
    wtc11_mut_color = "#FFC20A"
    patient_wt_color = "#35B5FD"
    patient_mut_color = "#FF7C0A"
    # making sns color palette from list of hex colors
    palette = [wtc11_wt_color, wtc11_mut_color, patient_wt_color, patient_mut_color]
    palette = sns.color_palette(palette)
    hue_order = ['WT-WTC11', 'Mutant-WTC11', 'WT-Patient Line', 'Mutant-Patient Line']

    # Plotting the Train
    T_p_train_wtc11, T_o_train_wtc11 = opls_wtc11.transform(X_train_balanced)
    plt.figure(figsize=(4, 4))
    pred_scores_train = T_p_train_wtc11.flatten()
    pred_offtarget_scores_train = T_o_train_wtc11.flatten()
    cell_line_train = ['WT-WTC11' if x == -10 else 'Mutant-WTC11' for x in y_train_balanced]
    # Plotting the Train on-target axis
    sns.kdeplot(x=pred_scores_train,
                hue=cell_line_train,
                fill=True,
                palette=wtc11_palette,
                bw_adjust=1,
                hue_order=wtc11_hue_order)
    plt.legend(hue_order, loc='upper right')
    plt.xlabel('O-PLS Disease Axis Projection')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

    # Plotting the Train off-target axis
    plt.figure(figsize=(4, 4))
    sns.kdeplot(x=pred_offtarget_scores_train, hue=cell_line_train, fill=True, palette=wtc11_palette, bw_adjust=0.8, hue_order=wtc11_hue_order)
    plt.legend(hue_order, loc='upper right')
    plt.xlabel('O-PLS Off-target Axis Projection')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

    # Plotting the Test
    plt.figure(figsize=(4, 4))
    pred_scores_test = T_p_test_wtc11.flatten()
    pred_offtarget_scores_test = T_o_test_wtc11.flatten()
    p_scores_test_wt = pred_scores_test[y_test_balanced == -10]
    o_scores_test_wt = pred_offtarget_scores_test[y_test_balanced == -10]
    p_scores_test_mut = pred_scores_test[y_test_balanced == 10]
    o_scores_test_mut = pred_offtarget_scores_test[y_test_balanced == 10]
    cell_line_test = ['WT-WTC11' if x == -10 else 'Mutant-WTC11' for x in y_test_balanced]
    sns.kdeplot(x=pred_scores_test,
                hue=cell_line_test,
                fill=True,
                palette=wtc11_palette,
                bw_adjust=1.0,
                hue_order=wtc11_hue_order)
    plt.xlabel('O-PLS Disease Axis Projection')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('OPLS_WTC11_kde_figure5.pdf', dpi=800)
    plt.show()
    # Plotting the Test off-target axis
    plt.figure(figsize=(4, 4))
    sns.kdeplot(x=pred_offtarget_scores_test, hue=cell_line_test, fill=True, palette=wtc11_palette, bw_adjust=0.8, hue_order=wtc11_hue_order)
    plt.legend(hue_order, loc='upper right')
    plt.xlabel('O-PLS Off-target Axis Projection')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

    # Printing the Accuracy of the OPLS model with a threshold of 0
    pred_scores_test = T_p_test_wtc11.flatten()
    pred_scores_train = T_p_train_wtc11.flatten()
    y_test = adata_wtc11_ntc_test.obs['cell_line'].values
    y_train = adata_wtc11_ntc_train.obs['cell_line'].values
    y_test = np.array([-10 if x == 'WT-WTC11' else 10 for x in y_test]).astype(float)
    y_train = np.array([-10 if x == 'WT-WTC11' else 10 for x in y_train]).astype(float)
    y_pred_test = np.array([10 if x > 0 else -10 for x in pred_scores_test])
    y_pred_train = np.array([10 if x > 0 else -10 for x in pred_scores_train])
    acc_test = np.mean(y_test_balanced == y_pred_test)
    acc_train = np.mean(y_train_balanced == y_pred_train)
    print(f'Test Accuracy: {acc_test}')
    print(f'Train Accuracy: {acc_train}')


    # using the OPPLS model to create shift scores for the WTC11 data
    adata_wtc11_mut = adata_wtc11[adata_wtc11.obs['cell_line'] == 'Mutant-WTC11', :]
    adata_wtc11_wt = adata_wtc11[adata_wtc11.obs['cell_line'] == 'WT-WTC11', :]

    # WTC11 WT Scoring
    wtc11_wt_dist_p_score = {}
    wtc11_wt_dist_o_score = {}
    for guide_id in adata_wtc11_wt.obs['guide_id'].unique():
        if 'non-targeting' in guide_id:
            continue
        guide_adata = adata_wtc11_wt[adata_wtc11_wt.obs['guide_id'] == guide_id, :]
        X_guide = guide_adata.X
        X_guide = scaler_wtc11.transform(X_guide)
        T_p_guide, T_o_guide = opls_wtc11.transform(X_guide)
        wtc11_wt_dist_o_score[guide_id] = T_o_guide.flatten()
        wtc11_wt_dist_p_score[guide_id] = T_p_guide.flatten()

    # WTC11 Mutant Scoring
    wtc11_mut_dist_p_score = {}
    wtc11_mut_dist_o_score = {}
    for guide_id in adata_wtc11_mut.obs['guide_id'].unique():
        if 'non-targeting' in guide_id:
            continue
        guide_adata = adata_wtc11_mut[adata_wtc11_mut.obs['guide_id'] == guide_id, :]
        X_guide = guide_adata.X
        X_guide = scaler_wtc11.transform(X_guide)
        T_p_guide, T_o_guide = opls_wtc11.transform(X_guide)
        wtc11_mut_dist_o_score[guide_id] = T_o_guide.flatten()
        wtc11_mut_dist_p_score[guide_id] = T_p_guide.flatten()

    wtc11_wt_p_values = {}
    wtc11_mut_p_values = {}

    null_dist_wt = p_scores_test_wt
    null_dist_mut = p_scores_test_mut
    null_dist_o_wt = o_scores_test_wt
    null_dist_o_mut = o_scores_test_mut

    for guide_id in adata_wtc11_wt.obs['guide_id'].unique():
        if 'non-targeting' in guide_id:
            continue
        guide_p_scores = wtc11_wt_dist_p_score[guide_id]
        guide_o_scores = wtc11_wt_dist_o_score[guide_id]
        p_value_p_score = ranksums(null_dist_wt, guide_p_scores)[1]
        p_value_o_score = ranksums(null_dist_o_wt, guide_o_scores)[1]
        shift_score =  np.mean(guide_p_scores) - np.mean(null_dist_wt)
        shift_o_score = np.mean(guide_o_scores) - np.mean(null_dist_o_wt)
        wtc11_wt_p_values[guide_id] = {'p_value_p_score': p_value_p_score,
                                    'p_value_o_score': p_value_o_score,
                                    'shift_o_score': shift_o_score,
                                    'shift_score': shift_score}

    for guide_id in adata_wtc11_mut.obs['guide_id'].unique():
        if 'non-targeting' in guide_id:
            continue
        guide_p_scores = wtc11_mut_dist_p_score[guide_id]
        guide_o_scores = wtc11_mut_dist_o_score[guide_id]
        p_value_p_score = ranksums(null_dist_mut, guide_p_scores)[1]
        p_value_o_score = ranksums(null_dist_o_mut, guide_o_scores)[1]
        shift_score =  np.mean(guide_p_scores) - np.mean(null_dist_mut)
        shift_o_score = np.mean(guide_o_scores) - np.mean(null_dist_o_mut)
        wtc11_mut_p_values[guide_id] = {'p_value_p_score': p_value_p_score,
                                        'p_value_o_score': p_value_o_score,
                                        'shift_o_score': shift_o_score,
                                        'shift_score': shift_score}


    # plotting the shift_o_scores vs shift scores for WTC11 WT
    shift_scores = [x['shift_score'] for x in wtc11_wt_p_values.values()]
    shift_o_scores = [x['shift_o_score'] for x in wtc11_wt_p_values.values()]

    pvals_p_score = [x['p_value_p_score'] for x in wtc11_wt_p_values.values()]
    pvals_o_score = [x['p_value_o_score'] for x in wtc11_wt_p_values.values()]

    adj_pvals_p_score = multipletests(pvals_p_score, method='fdr_bh')[1]
    adj_pvals_o_score = multipletests(pvals_o_score, method='fdr_bh')[1]


    # Plotting the -log10(adj p-values) for the off-target and on-target scores
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=-np.log10(adj_pvals_p_score)*shift_scores, y=-np.log10(adj_pvals_o_score)*shift_o_scores, c='k')
    plt.ylabel('Off-Axis Score')
    plt.xlabel('Disease-Axis Score')
    plt.title('WTC11 WT')
    plt.tight_layout()
    plt.savefig('on_off_target_scores_WTC11_WT_figure5.pdf', dpi=800)
    plt.show()

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=-np.log10(adj_pvals_p_score)*shift_scores, y=-np.log10(adj_pvals_o_score)*shift_o_scores, c='k')
    plt.ylabel('Off-Axis Score')
    plt.xlabel('Disease-Axis Score')
    plt.title('WTC11 WT')
    # labeling the guides
    for guide_id, adj_pval_p_score, adj_pval_o_score, ss, sos in zip(wtc11_wt_p_values.keys(),
                                                            adj_pvals_p_score,
                                                            adj_pvals_o_score,
                                                            shift_scores,
                                                            shift_o_scores):
        plt.text(-np.log10(adj_pval_p_score)*ss, -np.log10(adj_pval_o_score)*sos, guide_id, fontsize=6)
    plt.tight_layout()
    plt.show()

    # plotting the shift_o_scores vs shift scores for WTC11 Mutant
    shift_scores = [x['shift_score'] for x in wtc11_mut_p_values.values()]
    shift_o_scores = [x['shift_o_score'] for x in wtc11_mut_p_values.values()]

    pvals_p_score = [x['p_value_p_score'] for x in wtc11_mut_p_values.values()]
    pvals_o_score = [x['p_value_o_score'] for x in wtc11_mut_p_values.values()]

    adj_pvals_p_score = multipletests(pvals_p_score, method='fdr_bh')[1]
    adj_pvals_o_score = multipletests(pvals_o_score, method='fdr_bh')[1]

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=-np.log10(adj_pvals_p_score)*shift_scores, y=-np.log10(adj_pvals_o_score)*shift_o_scores, c='k')
    plt.xlabel('Disease-Axis Score')
    plt.ylabel('Off-Axis Score')
    plt.title('WTC11 Mutant')
    # labeling the guides
    for guide_id, adj_pval_p_score, adj_pval_o_score, ss, sos in zip(wtc11_mut_p_values.keys(),
                                                            adj_pvals_p_score,
                                                            adj_pvals_o_score,
                                                            shift_scores,
                                                            shift_o_scores):
        plt.text(-np.log10(adj_pval_p_score)*ss, -np.log10(adj_pval_o_score)*sos, guide_id, fontsize=6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=-np.log10(adj_pvals_p_score)*shift_scores, y=-np.log10(adj_pvals_o_score)*shift_o_scores, c='k')
    plt.xlabel('Disease-Axis Score')
    plt.ylabel('Off-Axis Score')
    plt.title('WTC11 Mutant')
    plt.tight_layout()
    plt.savefig('on_off_target_scores_WTC11_mutant_figure5.pdf', dpi=800)
    plt.show()


if __name__ == '__main__':
    main()
