from typing import Tuple, Dict, List, Union
import gc
from tqdm import tqdm

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
from scipy import linalg
import matplotlib.pyplot as plt


def apply_robust_center_scale(adata: ad.AnnData,
                              batch_obs_col: str,
                              control_obs_col: str,
                              control_key: str,
                              whiten: bool = True):
    """
    Apply robust center scaling to the data based on a control key.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to be scaled
    batch_obs_col : str
        The column in adata.obs that contains the batch information
    control_obs_col : str
        The column in adata.obs that contains the control information
    control_key : str
        The key in control_obs_col that should be used as the control
    whiten : bool
        Whether to whiten the data after scaling
    
    Returns
    -------
    AnnData
        The processed AnnData object
    """
    assert control_key in adata.obs[control_obs_col].unique(), f"Key {control_key} not found in adata.obs"
    robust_center_scaled_data = []
    for batch_id in adata.obs[batch_obs_col].unique():
        batch_adata = adata[adata.obs[batch_obs_col] == batch_id]
        batch_control_adata = batch_adata[batch_adata.obs[control_obs_col] == control_key]
        batch_control_median = np.median(batch_control_adata.X, axis=0)
        batch_control_median_abs_dev = stats.median_abs_deviation(batch_control_adata.X, axis=0, scale='normal')
        if np.any(batch_control_median_abs_dev == 0) or batch_control_adata.shape[0] < 2:
            print(f"Batch {batch_id} dropped due to insufficient control data")
            continue
        scaled_embeddings = (batch_adata.X - batch_control_median) / batch_control_median_abs_dev
        scaled_adata = ad.AnnData(X=scaled_embeddings, obs=batch_adata.obs)
        robust_center_scaled_data.append(scaled_adata)
    robust_center_scaled_adata = ad.concat(robust_center_scaled_data)
    del robust_center_scaled_data
    gc.collect()
    if whiten:
        pca = PCA(n_components=robust_center_scaled_adata.shape[1], whiten=True)
        whitened_X = pca.fit_transform(np.array(robust_center_scaled_adata.X))
        robust_center_scaled_adata.X = whitened_X
    return robust_center_scaled_adata


def centerscale_on_controls(
    adata: ad.AnnData,
    batch_obs_col: str,
    control_obs_col: str,
    control_key: str
) -> ad.AnnData:
    """
    Center and scale the embeddings on the control perturbation units in the metadata.
    If batch information is provided, the embeddings are centered and scaled by batch.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the embeddings
    batch_obs_col : str
        The column in adata.obs that contains the batch information
    control_obs_col : str
        The column in adata.obs that contains the control information
    control_key : str
        The key in control_obs_col that should be used as the control
    
    Returns
    -------
    AnnData
        The processed AnnData object
    """
    embeddings = np.array(adata.X)
    metadata = adata.obs
    batch_col = batch_obs_col
    pert_col = control_obs_col

    embeddings = embeddings.copy()
    if batch_col is not None:
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = batch_ind & (metadata[pert_col] == control_key)
            embeddings[batch_ind] = StandardScaler().fit(embeddings[batch_control_ind]).transform(embeddings[batch_ind])
    center_scaled_adata = ad.AnnData(X=embeddings, obs=adata.obs)
    return center_scaled_adata


def apply_typical_variation_normalization(adata: ad.AnnData,
                                        batch_obs_col: str,
                                        control_obs_col: str,
                                        control_key: str) -> ad.AnnData:
    """
    Apply typical variation normalization to the data based on a control key.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to be normalized
    batch_obs_col : str
        The column in adata.obs that contains the batch information
    control_obs_col : str
        The column in adata.obs that contains the control information
    control_key : str
        The key in control_obs_col that should be used as the control
    
    Returns
    -------
    AnnData
        The processed AnnData object
    """
    adata = centerscale_on_controls(adata, batch_obs_col, control_obs_col, control_key)
    ctrl_ind = adata.obs[control_obs_col] == control_key
    embeddings = PCA().fit(adata[ctrl_ind].X).transform(adata.X)
    adata = ad.AnnData(X=embeddings, obs=adata.obs)
    del embeddings
    gc.collect()
    adata = centerscale_on_controls(adata, batch_obs_col, control_obs_col, control_key)
    gc.collect()
    embeddings = adata.X.copy()
    if batch_obs_col is not None:
        batches = adata.obs[batch_obs_col].unique()
        for batch in batches:
            batch_ind = adata.obs[batch_obs_col] == batch
            batch_control_ind = batch_ind & (adata.obs[control_obs_col] == control_key)
            source_cov = np.cov(adata[batch_control_ind].X, rowvar=False, ddof=1) + 0.5 * np.eye(adata.X.shape[1])
            source_cov_half_inv = linalg.fractional_matrix_power(source_cov, -0.5)
            embeddings[batch_ind] = np.matmul(adata[batch_ind].X, source_cov_half_inv)
    adata_out = ad.AnnData(X=embeddings, obs=adata.obs)
    return adata_out


def aggregate_by_column(adata: ad.AnnData, col: str) -> ad.AnnData:
    """
    Groups an AnnData object by a specified key in `obs` and calculates the mean 
    of `X` for each group. Columns in `obs` that are identical within each group 
    are preserved in the output AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        The AnnData object containing data to be grouped and averaged.
    col : str
        The key in `adata.obs` to group by.

    Returns
    -------
    ad.AnnData
        A new AnnData object with the grouped mean `X` values and relevant `obs` metadata.
    """
    obs_key = col
    # Group by the specified key and calculate mean over X
    for_grouping = pd.DataFrame(adata.X, columns=[f'col_{i}' for i in range(adata.X.shape[1])])
    for_groupby = adata.obs[obs_key].values
    for_grouping['group_id'] = for_groupby
    group_means = for_grouping.groupby('group_id').mean()
    # Determine constant columns in obs for each group
    constant_obs_columns = [
        col for col in adata.obs.columns 
        if adata.obs.groupby(obs_key)[col].nunique().eq(1).all()
    ]
    obs_means = adata.obs.groupby(obs_key)[constant_obs_columns].first()
    # Create a new AnnData object with the grouped means and updated obs
    grouped_adata = ad.AnnData(X=group_means, obs=obs_means)
    return grouped_adata


def save_h5ad_file(adata: ad.AnnData, filename: str):
    """
    Save an AnnData object to an h5ad file.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object to save.
    filename : str
        Name of the file to save to.
    """
    adata.write(filename)


def multivariate_energy_distance(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> float:
    """
    Calculate the energy distance between two high-dimensional distributions.
    
    Parameters
    ----------
    X : np.ndarray
        An array of shape (n, d) representing n samples from the first distribution.
    Y : np.ndarray
        An array of shape (m, d) representing m samples from the second distribution.
    metric : str, optional
        The distance metric to use. Options are 'euclidean' or 'cosine'. Default is 'euclidean'.
    
    Returns
    -------
    float
        The computed energy distance between the distributions.
    """
    if metric not in ['euclidean', 'cosine']:
        raise ValueError("The 'metric' parameter must be 'euclidean' or 'cosine'.")

    n, d = X.shape
    m, _ = Y.shape
    
    # Compute pairwise distances between all points in X and Y
    if metric == 'euclidean':
        # Pairwise distances between all X and Y (cross-term)
        dists_XY = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))
        dists_X_X = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2))
        dists_Y_Y = np.sqrt(np.sum((Y[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))
    elif metric == 'cosine':
        # Normalize X and Y for cosine distance calculation
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        
        # Cosine distance = 1 - cosine similarity
        dists_XY = 1 - np.dot(X_normalized, Y_normalized.T)
        dists_X_X = 1 - np.dot(X_normalized, X_normalized.T)
        dists_Y_Y = 1 - np.dot(Y_normalized, Y_normalized.T)
    
    # Compute the means of the pairwise distances
    mean_dists_XY = np.mean(dists_XY)
    mean_dists_X_X = np.sum(dists_X_X) / (n * (n - 1))
    mean_dists_Y_Y = np.sum(dists_Y_Y) / (m * (m - 1))
    
    # Energy distance formula
    energy_dist = 2 * mean_dists_XY - mean_dists_X_X - mean_dists_Y_Y
    
    return energy_dist


def calculate_null_distribution_energy_distance(ctrl_adata: ad.AnnData,
                                                ctrl_rand_idx: np.ndarray,
                                                num_samples: int = 500,
                                                max_sample_per_dist: int = 500,
                                                metric: str = 'euclidean',
                                                ) -> np.ndarray:
    """
    This function calculates the null distribution of energy distances between the embeddings of the control units.
    """
    energy_distances = []
    sample_1 = ctrl_adata[ctrl_rand_idx].X
    print("Generating null distribution for Energy Distances...")
    for i in tqdm(range(num_samples)):
        ctrl_adata_temp = ctrl_adata[~ctrl_adata.obs.index.isin(ctrl_rand_idx)]
        sample_2_idx = np.random.choice(np.arange(ctrl_adata_temp.shape[0]), max_sample_per_dist, replace=False)
        sample_2 = ctrl_adata_temp[sample_2_idx].X
        energy_distance = multivariate_energy_distance(sample_1, sample_2, metric=metric)
        energy_distances.append(energy_distance)
    return np.array(energy_distances)


def calculate_energy_distance(adata: ad.AnnData,
                              ctrl_adata: ad.AnnData,
                              pert_col: str,
                              max_sample_per_dist: int = 500,
                              metric: str = 'euclidean',
                              ) -> Dict[str, float]:
    """
    This function calculates the energy distance between the embeddings of the control and perturbation units.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the embeddings
    ctrl_adata : AnnData
        The AnnData object containing the embeddings of the control units
    pert_col : str
        The column in adata.obs that contains the perturbation information
    metric : str
        The distance metric to use. Options are 'euclidean' or 'cosine'. Default is 'euclidean'.
    """
    energy_distances = {}

    if (ctrl_adata.shape[0] > max_sample_per_dist) and (max_sample_per_dist <= int(ctrl_adata.shape[0] * 0.6)):
        ctrl_rand_idx = np.random.choice(np.arange(ctrl_adata.shape[0]), max_sample_per_dist, replace=False)
        null_dist = calculate_null_distribution_energy_distance(ctrl_adata, ctrl_rand_idx, metric=metric)
        ctrl_adata = ctrl_adata[ctrl_rand_idx, :]
    else:
        print(f"Ctrl adata shape: {ctrl_adata.shape} is less than max_sample_per_dist: {max_sample_per_dist}")
        print("Updating num_sample_per_dist to enable calculation...")
        max_sample_per_dist = int(ctrl_adata.shape[0] * 0.6)
        ctrl_rand_idx = np.random.choice(np.arange(ctrl_adata.shape[0]), max_sample_per_dist, replace=False)
        null_dist = calculate_null_distribution_energy_distance(ctrl_adata, ctrl_rand_idx, metric=metric)
    
    print(null_dist)
        
    print("Calculating energy distances...")
    for pert in tqdm(adata.obs[pert_col].unique()):
        pert_ind = adata.obs[pert_col] == pert
        if np.sum(pert_ind) > max_sample_per_dist:
            pert_rand_idx = np.random.choice(np.where(pert_ind)[0], max_sample_per_dist, replace=False)
        else:
            pert_rand_idx = np.where(pert_ind)[0]
        energy_distance = multivariate_energy_distance(adata[pert_rand_idx].X, ctrl_adata.X, metric=metric)

        # Calculating where in the sorted null distribution the energy distance falls
        sorted_array = np.sort(null_dist)
        index = np.searchsorted(sorted_array, energy_distance, side='right')
        p_value = (len(sorted_array) - index) / len(sorted_array)
        energy_distances[pert] = (energy_distance, p_value)
    return energy_distances


def generate_cosine_similarity_matrix(adata: ad.AnnData,
                                      col: str = 'gene_id') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a cosine similarity matrix from the embeddings in the AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing the embeddings.
    col : str
        Name of the column containing the key for names per embedding.

    Returns
    -------
    np.ndarray
        Cosine similarity matrix.
    """
    cosine_sim_matrix = cosine_similarity(adata.X)
    gene_names = adata.obs[col].values
    return cosine_sim_matrix, gene_names


def process_cell_line_adata(adata: ad.AnnData,
                            wandb_uid: str,
                            cell_line_name: str,
                            single_cell_id_col: str = "single_cell_id",
                            gene_col: str = "gene_id",
                            guide_col: str = "guide_id",
                            control_key: str = "non-targeting",
                            guides_to_drop: Union[List, None] = None
                            ) -> None:
    """
    This funciton processes the adata object to prepare it for downstream analysis
    and saves the processed objects to h5ad files.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing the embeddings
    wandb_uid : str
        The unique identifier for the wandb run
    cell_line_name : str
        The name of the cell line
    single_cell_id_col : str
        The column in adata.obs that contains the single cell id
    gene_col : str
        The column in adata.obs that contains the gene id
    guide_col : str
        The column in adata.obs that contains the guide id
    control_key : str
        The key in control_obs_col that should be used as the
    guides_to_drop : Union[List, None]
        A list of guides to drop from the analysis (optional)
    """
    if guides_to_drop is not None:
        adata = adata[~adata.obs[guide_col].isin(guides_to_drop)]

    # Step 1: Apply TVN
    adata_tvn = apply_typical_variation_normalization(adata,
                                                      batch_obs_col="plate_id",
                                                      control_obs_col="gene_id",
                                                      control_key="non-targeting")
    # Step 2: Aggregate by single_cell_id
    adata_agg = aggregate_by_column(adata_tvn, col=single_cell_id_col)

    # Step 3: Save the processed objects
    save_h5ad_file(adata_tvn, f"Single_Cell_TVN_embeddings_{wandb_uid}_{cell_line_name}.h5ad")
    sig_guides_adata = adata_agg

    # Step 5: Aggregate by guide_id
    sig_guides_adata_agg = aggregate_by_column(sig_guides_adata, col=guide_col)

    # Step 6: Save the processed objects
    save_h5ad_file(sig_guides_adata_agg, f"TVN_Guide_Aggregated_Embeddings_{wandb_uid}_{cell_line_name}.h5ad")

    # Step 7: Aggregate by gene_id
    adata_agg_gene = aggregate_by_column(sig_guides_adata_agg, col=gene_col)

    # Step 8: Save the processed objects
    save_h5ad_file(adata_agg_gene, f"TVN_Gene_Aggregated_Embeddings_{wandb_uid}_{cell_line_name}.h5ad")

    # Step 9: Generate the cosine similarity matrix
    cosine_sim_matrix, gene_names = generate_cosine_similarity_matrix(adata_agg_gene,
                                                                      col=gene_col)
    np.save(f"Gene_Cosine_Similarity_Matrix_{wandb_uid}_{cell_line_name}.npy", cosine_sim_matrix)
    np.save(f"Gene_Names_{wandb_uid}_{cell_line_name}.npy", gene_names)



def compute_auroc_for_guides(
    adata,
    gene_id_col='gene_id',
    guide_id_col='guide_id',
    well_id_col='well_id',
    non_targeting_label='non-targeting',
    train_fraction=0.6,
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
    guide_id_list = []
    auroc_list = []
    
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
    X_ntc_train = ntc_train.X
    y_ntc_train = np.zeros(X_ntc_train.shape[0])
    X_ntc_test = ntc_test.X
    y_ntc_test = np.zeros(X_ntc_test.shape[0])

    #########
    ### Handeling each guide
    #########
    unique_guides = np.unique(adata.obs[guide_id_col])
    for unique_guide_id in unique_guides:
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
        try:
            log_reg_model.fit(X_train, y_train)
        except Exception as e:
            print(f'Error fitting model for guide {unique_guide_id}: {e}. Skipping.')
            continue
        # Combine ntc and guide data for testing
        X_test = np.vstack([X_ntc_test, X_guide_test])
        y_test = np.hstack([y_ntc_test, y_guide_test])
        # Check if y_test contains both classes
        if len(np.unique(y_test)) < 2:
            print(f'Testing data does not contain both classes for guide {unique_guide_id}. Skipping.')
            continue
        try:
            y_pred = log_reg_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
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
    return pd.DataFrame({guide_id_col: guide_id_list, 'auroc': auroc_list})
