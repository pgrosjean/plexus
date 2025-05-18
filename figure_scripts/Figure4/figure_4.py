import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from plexus.embedding_utils.preprocessing import (aggregate_by_column,
                                                  apply_typical_variation_normalization)


def compute_auroc_for_guides(
    adata,
    gene_id_col='gene_id',
    guide_id_col='guide_id',
    well_id_col='well_id',
    non_targeting_label='non-targeting',
    train_fraction=0.5,
    plot_curves=True) -> pd.DataFrame:
    """
    This function computes the AUROC for each guide in the dataset using a binary classifier.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the embeddings
    gene_id_col : str
        The column in adata.obs that contains the gene id
    guide_id_col : str
        The column in adata.obs that contains the guide id
    well_id_col : str
        The column in adata.obs that contains the well id
    non_targeting_label : str
        The label for the non-targeting control
    train_fraction : float
        The fraction of wells to use for training
    plot_curves : bool
        Whether to plot the ROC curves for each guide
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the AUROC values for each guide
    """
    np.random.seed(0)
    guide_id_list = []
    auroc_list = []
    auroc_list_2 = [] # for the second fold of the split

    # Standardize the data
    ss = StandardScaler()
    adata.X = ss.fit_transform(adata.X)
    
    #########
    ### Handeling NTCs
    #########
    
    # Extract non-targeting controls
    adata_ntc_temp = adata[adata.obs[gene_id_col] == non_targeting_label, :]

    # Check if there are enough NTC wells to split
    ntc_well_ids = np.unique(adata_ntc_temp.obs[well_id_col])
    if len(ntc_well_ids) < 2:
        print('Not enough NTC wells to perform split. Exiting function.')
        return pd.DataFrame({guide_id_col: guide_id_list, 'auroc': auroc_list})
    # Split wells for NTC into train and test
    
    ntc_train_wells = np.random.choice(
        ntc_well_ids,
        max(1, int(train_fraction * len(ntc_well_ids))), 
        replace=False
    )
    ntc_test_wells = np.setdiff1d(ntc_well_ids, ntc_train_wells)
    if len(ntc_test_wells) == 0:
        print('Not enough NTC wells for testing after split. Exiting function.')
        return pd.DataFrame({guide_id_col: guide_id_list, 'auroc': auroc_list})
    ntc_train = adata_ntc_temp[adata_ntc_temp.obs[well_id_col].isin(ntc_train_wells), :]
    ntc_test = adata_ntc_temp[adata_ntc_temp.obs[well_id_col].isin(ntc_test_wells), :]
    if ntc_train.n_obs == 0 or ntc_test.n_obs == 0:
        print('Not enough NTC samples after splitting for training/testing. Exiting function.')
        return pd.DataFrame({guide_id_col: guide_id_list, 'auroc': auroc_list})
    # defining the NTC Train and Test data
    X_ntc_train_raw = ntc_train.X
    y_ntc_train = np.zeros(X_ntc_train_raw.shape[0])
    X_ntc_test_raw = ntc_test.X
    y_ntc_test = np.zeros(X_ntc_test_raw.shape[0])

    #########
    ### Handeling each guide
    #########
    unique_guides = np.unique(adata.obs[guide_id_col])
    for unique_guide_id in unique_guides:
        # Reseting the train and test data for the NTC to account for previous changes in the class balancing
        X_ntc_train = X_ntc_train_raw
        y_ntc_train = np.zeros(X_ntc_train_raw.shape[0])
        X_ntc_test = X_ntc_test_raw
        y_ntc_test = np.zeros(X_ntc_test_raw.shape[0])
        if 'non-targeting' in unique_guide_id:
            ntc_train_temp = ntc_train[ntc_train.obs[guide_id_col] != unique_guide_id, :]
            ntc_test_temp = ntc_test[ntc_test.obs[guide_id_col] != unique_guide_id, :]
            X_ntc_train = ntc_train_temp.X
            y_ntc_train = np.zeros(X_ntc_train.shape[0])
            X_ntc_test = ntc_test_temp.X
            y_ntc_test = np.zeros(X_ntc_test.shape[0])

        adata_guide_temp = adata[adata.obs[guide_id_col] == unique_guide_id, :]
        all_wells = np.unique(adata_guide_temp.obs[well_id_col])
        if len(all_wells) < 2:
            print(f'Not enough wells to perform split for guide {unique_guide_id}. Skipping.')
            continue
        train_wells = np.random.choice(
            all_wells,
            max(1, int(train_fraction * len(all_wells))), 
            replace=False
        )
        test_wells = np.setdiff1d(all_wells, train_wells)
        if len(test_wells) == 0:
            print(f'Not enough wells for testing after split for guide {unique_guide_id}. Skipping.')
            continue
        adata_guide_train = adata_guide_temp[adata_guide_temp.obs[well_id_col].isin(train_wells), :]
        adata_guide_test = adata_guide_temp[adata_guide_temp.obs[well_id_col].isin(test_wells), :]
        if adata_guide_train.n_obs == 0 or adata_guide_test.n_obs == 0:
            print(f'Not enough samples after splitting for guide {unique_guide_id}. Skipping.')
            continue
        X_guide_train = adata_guide_train.X
        y_guide_train = np.ones(X_guide_train.shape[0])
        X_guide_test = adata_guide_test.X
        y_guide_test = np.ones(X_guide_test.shape[0])
        # Combine ntc and guide data for training

        #########
        ## Handeling class imbalance
        #########
        sample_size = min(X_ntc_train.shape[0], X_guide_train.shape[0])
        
        # The sample size must be greater than 8
        if sample_size <= 8:
            continue
        
        if X_ntc_train.shape[0] > sample_size:
            rand_inds = np.random.choice(X_ntc_train.shape[0], sample_size, replace=False)
            X_ntc_train = X_ntc_train[rand_inds]
            y_ntc_train = y_ntc_train[rand_inds]
        if X_guide_train.shape[0] > sample_size:
            rand_inds = np.random.choice(X_guide_train.shape[0], sample_size, replace=False)
            X_guide_train = X_guide_train[rand_inds]
            y_guide_train = y_guide_train[rand_inds]

        X_train = np.vstack([X_ntc_train, X_guide_train])
        y_train = np.hstack([y_ntc_train, y_guide_train])
        # Combine ntc and guide data for testing
        X_test = np.vstack([X_ntc_test, X_guide_test])
        y_test = np.hstack([y_ntc_test, y_guide_test])
        # Check if y_train contains both classes
        if len(np.unique(y_train)) < 2:
            print(f'Training data does not contain both classes for guide {unique_guide_id}. Skipping.')
            continue
        log_reg_model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            penalty='elasticnet', 
            solver='saga', 
            l1_ratio=0.5
        )
        log_reg_model_2 = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            penalty='elasticnet', 
            solver='saga',
            l1_ratio=0.5
        )
        try:
            log_reg_model.fit(X_train, y_train)
            log_reg_model_2.fit(X_test, y_test)
        except Exception as e:
            print(f'Error fitting model for guide {unique_guide_id}: {e}. Skipping.')
            continue
        # Check if y_test contains both classes
        if len(np.unique(y_test)) < 2:
            print(f'Testing data does not contain both classes for guide {unique_guide_id}. Skipping.')
            continue
        try:
            y_pred = log_reg_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            y_pred_2 = log_reg_model_2.predict_proba(X_train)[:, 1]
            fpr_2, tpr_2, _ = roc_curve(y_train, y_pred_2)
            roc_auc_2 = auc(fpr_2, tpr_2)
        except Exception as e:
            print(f'Error evaluating model for guide {unique_guide_id}: {e}. Skipping.')
            continue
        if plot_curves:
            plt.figure(figsize=(10, 10))
            lw = 2
            plt.plot(
                fpr, tpr, color='darkorange', lw=lw,
                label='ROC curve (area = %0.2f)' % roc_auc
            )
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic Curve {unique_guide_id}')
            plt.legend(loc="lower right")
            # Adding the AUROC score to the plot
            plt.text(0.6, 0.2, f'AUROC = {roc_auc:.2f}', fontsize=12)
            plt.show()
        guide_id_list.append(unique_guide_id)
        auroc_list.append(roc_auc)
        auroc_list_2.append(roc_auc_2)
        return_df = pd.DataFrame({
            guide_id_col: np.hstack([np.array(guide_id_list), np.array(guide_id_list)]),
            'auroc': np.hstack([np.array(auroc_list), np.array(auroc_list_2)])
        })
    return return_df


def main():
    base_path = "/scratch/pgrosjean/plexus_data_archive"
    wtc11_wt_color = "#5D8195"
    wtc11_mut_color = "#FFC20A"
    patient_wt_color = "#35B5FD"
    patient_mut_color = "#FF7C0A"
    # making sns color palette from list of hex colors
    palette = [wtc11_wt_color, wtc11_mut_color, patient_wt_color, patient_mut_color]
    palette = sns.color_palette(palette)
    
    # Reading in the adata for embeddings
    adata_wt_wtc11_sc_tvn_embed = ad.read_h5ad(f'{base_path}/plexus_embeddings/crispri_screen/tvn_corrected_embeddings/single_cell_tvn_corrected_embeddings_WT_WTC11_CRISPRi.h5ad')
    adata_wt_patient_sc_tvn_embed = ad.read_h5ad(f'{base_path}/plexus_embeddings/crispri_screen/tvn_corrected_embeddings/single_cell_tvn_corrected_embeddings_WT_Patient_Line_CRISPRi.h5ad')

    # Reading in the adata for the raw data
    adata_wt_wtc11_sc_tvn_embed = aggregate_by_column(adata_wt_wtc11_sc_tvn_embed, 'aggregate_by')
    adata_wt_patient_sc_tvn_embed = aggregate_by_column(adata_wt_patient_sc_tvn_embed, 'aggregate_by')

    scaler = StandardScaler()
    man_feat_adata = ad.read_h5ad(f"{base_path}/plexus_embeddings/crispri_screen/crispri_screen_manual_features.h5ad")
    man_feat_adata.X = scaler.fit_transform(man_feat_adata.X)
    adata_wt = man_feat_adata[man_feat_adata.obs['tau_status'] == 'WT']
    adata_wt_wtc11_sc_feat = adata_wt[adata_wt.obs['donor_id'] == 'WTC11']
    adata_wt_patient_sc_feat = adata_wt[adata_wt.obs['donor_id'] == 'Patient Line']
    adata_wt_wtc11_sc_tvn_feat = apply_typical_variation_normalization(adata_wt_wtc11_sc_feat,
                                                                    batch_obs_col='zarr_file',
                                                                    control_obs_col='gene_id',
                                                                    control_key='non-targeting')
    adata_wt_patient_sc_tvn_feat = apply_typical_variation_normalization(adata_wt_patient_sc_feat,
                                                                     batch_obs_col='zarr_file',
                                                                     control_obs_col='gene_id',
                                                                     control_key='non-targeting')
    adata_wt_wtc11_sc_tvn_feat.obs['loc_id'] = adata_wt_wtc11_sc_tvn_feat.obs['well_id'].astype('str') + '_' + adata_wt_wtc11_sc_tvn_feat.obs['zarr_file'].astype('str')
    adata_wt_patient_sc_tvn_feat.obs['loc_id'] = adata_wt_patient_sc_tvn_feat.obs['well_id'].astype('str') + '_' + adata_wt_patient_sc_tvn_feat.obs['zarr_file'].astype('str')

    wt_wtc11_tvn_feat_auroc = compute_auroc_for_guides(
                        adata_wt_wtc11_sc_tvn_feat,
                        gene_id_col='gene_id',
                        guide_id_col='guide_id',
                        well_id_col='loc_id',
                        non_targeting_label='non-targeting',
                        train_fraction=0.5,
                        plot_curves=False)
    wt_patient_tvn_feat_auroc = compute_auroc_for_guides(
                            adata_wt_patient_sc_tvn_feat,
                            gene_id_col='gene_id',
                            guide_id_col='guide_id',
                            well_id_col='loc_id',
                            non_targeting_label='non-targeting',
                            train_fraction=0.5,
                            plot_curves=False)
    
    adata_wt_wtc11_sc_tvn_embed.obs['loc_id'] = adata_wt_wtc11_sc_tvn_embed.obs['aggregate_by'].apply(lambda x: '_'.join(x.split('_')[:2])) + '_' + adata_wt_wtc11_sc_tvn_embed.obs['well_id'].astype('str')
    adata_wt_patient_sc_tvn_embed.obs['loc_id'] = adata_wt_patient_sc_tvn_embed.obs['aggregate_by'].apply(lambda x: '_'.join(x.split('_')[:2])) + '_' + adata_wt_patient_sc_tvn_embed.obs['well_id'].astype('str')

    wt_wtc11_tvn_embed_auroc = compute_auroc_for_guides(adata_wt_wtc11_sc_tvn_embed,
                        gene_id_col='gene_id',
                        guide_id_col='guide_id',
                        well_id_col='loc_id',
                        non_targeting_label='non-targeting',
                        train_fraction=0.5,
                        plot_curves=False)
    wt_patient_tvn_embed_auroc = compute_auroc_for_guides(adata_wt_patient_sc_tvn_embed,
                        gene_id_col='gene_id',
                        guide_id_col='guide_id',
                        well_id_col='loc_id',
                        non_targeting_label='non-targeting',
                        train_fraction=0.5,
                        plot_curves=False)
    
    # converting from UIDs to unique integer values for plotting
    guide_id_dict_mapping = {}
    unq_ids = sorted(wt_wtc11_tvn_feat_auroc['guide_id'].astype(str).unique())
    for unq_id in unq_ids:
        gene_name = unq_id.split('_')[0]
        c = 1
        new_guide_id = f"{gene_name}_{c}"
        while new_guide_id in guide_id_dict_mapping.values():
            c += 1
            new_guide_id = f"{gene_name}_{c}"
        guide_id_dict_mapping[unq_id] = new_guide_id

    wt_patient_tvn_feat_auroc['guide_id_new'] = wt_patient_tvn_feat_auroc['guide_id'].astype(str).map(guide_id_dict_mapping)
    wt_wtc11_tvn_feat_auroc['guide_id_new'] = wt_wtc11_tvn_feat_auroc['guide_id'].astype(str).map(guide_id_dict_mapping)
    wt_patient_tvn_embed_auroc['guide_id_new'] = wt_patient_tvn_embed_auroc['guide_id'].astype(str).map(guide_id_dict_mapping)
    wt_wtc11_tvn_embed_auroc['guide_id_new'] = wt_wtc11_tvn_embed_auroc['guide_id'].astype(str).map(guide_id_dict_mapping)

    wtc11_wt_color = "#5D8195"
    wtc11_mut_color = "#FFC20A"
    patient_wt_color = "#35B5FD"
    patient_mut_color = "#FF7C0A"
    # making sns color palette from list of hex colors
    palette = [wtc11_wt_color, wtc11_mut_color, patient_wt_color, patient_mut_color]
    palette = sns.color_palette(palette)

    plt.figure(figsize=(8, 4))
    wt_wtc11_tvn_feat_auroc = wt_wtc11_tvn_feat_auroc.sort_values(by='auroc', ascending=False)
    sns.barplot(x='guide_id_new', y='auroc', data=wt_wtc11_tvn_feat_auroc, color=wtc11_wt_color, alpha=0.5)
    sns.swarmplot(x='guide_id_new', y='auroc', data=wt_wtc11_tvn_feat_auroc, color='k', alpha=1)
    plt.xlabel('Guide ID')
    plt.ylabel('AUROC Binary Classifier')
    plt.xticks(rotation=90)
    plt.title('WTC11 Cell Line (Manual Features)')
    # Plotting horizontal line
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axhline(y=0.5, color='k')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('guide_auroc_wt_wtc11_line_engineered_features.pdf', dpi=800)
    plt.show()

    plt.figure(figsize=(8, 4))
    wt_patient_tvn_feat_auroc = wt_patient_tvn_feat_auroc.sort_values(by='auroc', ascending=False)
    sns.barplot(x='guide_id_new', y='auroc', data=wt_patient_tvn_feat_auroc, color=patient_wt_color, alpha=0.5)
    sns.swarmplot(x='guide_id_new', y='auroc', data=wt_patient_tvn_feat_auroc, color='k', alpha=1)
    plt.xlabel('Guide ID')
    plt.ylabel('AUROC Binary Classifier')
    plt.xticks(rotation=90)
    plt.title('Patient Line Cell Line (Manual Features)')
    # Plotting horizontal line
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axhline(y=0.5, color='k')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('guide_auroc_wt_patient_line_engineered_features.pdf', dpi=800)
    plt.show()

    plt.figure(figsize=(8, 4))
    wt_wtc11_tvn_embed_auroc = wt_wtc11_tvn_embed_auroc.sort_values(by='auroc', ascending=False)
    sns.barplot(x='guide_id_new', y='auroc', data=wt_wtc11_tvn_embed_auroc, color=wtc11_wt_color, alpha=1)
    sns.swarmplot(x='guide_id_new', y='auroc', data=wt_wtc11_tvn_embed_auroc, color='k', alpha=1)
    plt.xlabel('Guide ID')
    plt.ylabel('AUROC Binary Classifier')
    plt.xticks(rotation=90)
    plt.title('WTC11 Cell Line (Plexus Embeddings)')
    # Plotting horizontal line
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axhline(y=0.5, color='k')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('guide_auroc_wt_wtc11_line_embeddings.pdf', dpi=800)
    plt.show()

    plt.figure(figsize=(8, 4))
    wt_patient_tvn_embed_auroc = wt_patient_tvn_embed_auroc.sort_values(by='auroc', ascending=False)
    sns.barplot(x='guide_id_new', y='auroc', data=wt_patient_tvn_embed_auroc, color=patient_wt_color, alpha=1)
    sns.swarmplot(x='guide_id_new', y='auroc', data=wt_patient_tvn_embed_auroc, color='k', alpha=1)
    plt.xlabel('Guide ID')
    plt.ylabel('AUROC Binary Classifier')
    plt.xticks(rotation=90)
    plt.title('Patient Line Cell Line (Plexus Embeddings)')
    # Plotting horizontal line
    plt.axhline(y=0.8, color='r', linestyle='--')
    plt.axhline(y=0.5, color='k')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('guide_auroc_wt_patient_line_embeddings.pdf', dpi=800)
    plt.show()

    x = wt_patient_tvn_embed_auroc.sort_values(by='guide_id', ascending=False)['auroc']
    y = wt_wtc11_tvn_embed_auroc.sort_values(by='guide_id', ascending=False)['auroc']
    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=x, y=y, color='k', s=3)
    plt.xlabel('Patient Line AUROC')
    plt.ylabel('WTC11 AUROC')
    # Plotting diagonal line
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    corr_coeff = np.corrcoef(x, y)[0, 1]
    print(f'Correlation coefficient: {corr_coeff}')
    plt.show()


if __name__ == '__main__':
    main()
