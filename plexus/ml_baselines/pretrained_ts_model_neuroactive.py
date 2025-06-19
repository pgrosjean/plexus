import torch
import pandas as pd
import zarr
import os
import torch
import numpy as np
from tqdm import tqdm
import anndata as ad
import torch
from chronos import ChronosPipeline
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def get_zarr_info(zarr_location):
    root_dict = {}
    for zarr_file in os.listdir(zarr_location):
        print(f'{zarr_location}{zarr_file}')
        if zarr_file.endswith('.zarr'):
            root_dict[zarr_file.split('.')[0]] = zarr.open(f'{zarr_location}{zarr_file}', 'r')
    return root_dict


def plot_roc_curves_with_ci(X_dict, y_dict, groups_dict, color_dict, n_splits=2):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for treatment, X in X_dict.items():
        y = y_dict[treatment]
        groups = groups_dict[treatment]
        color = color_dict[treatment]
        
        cv = GroupKFold(n_splits=n_splits)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv.split(X, y, groups)):
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X[train], y[train])

            y_pred = model.predict_proba(X[test])[:, 1]
            fpr, tpr, _ = roc_curve(y[test], y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = roc_auc_score(y[test], y_pred)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color=color,
                label=f'{treatment} vs. Neg Ctrl (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                lw=1, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)

    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    return fig

def main():
    base_path = "../../plexus_data_archive"
    # Reading in the Zarr Information
    zarr_info = get_zarr_info(f"{base_path}/processed_zarr_files/neuroactive_stimulation/split_zarr_files/")
    # Loading in the pre-trained amazong chronos pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map=device,
    torch_dtype=torch.bfloat16,
    )
    all_embeddings = []
    well_ids = []
    fov_ids = []
    with torch.no_grad():
        for zarr_root in zarr_info.values():
            for well in tqdm(list(zarr_root.keys())):
                for fov in list(zarr_root[well].keys()):
                    signal = torch.Tensor(np.array(zarr_root[well][fov]['signal']))
                    embeddings, tokenizer_state = pipeline.embed(signal)
                    all_embeddings.append(embeddings.mean(dim=1).float().detach().cpu().numpy())
                    fov_ids.append([fov]*len(embeddings))
                    well_ids.append([well]*len(embeddings))
    all_embeddings = np.vstack(all_embeddings)
    well_ids = np.hstack(well_ids)
    fov_ids = np.hstack(fov_ids)
    metadata_df = pd.read_csv(f"{base_path}/metadata_files/neuroactive_stimulation_metadata.csv")
    metadata_map = {k: v for k, v in zip(metadata_df['well_id'].values, metadata_df['treatment'].values)}
    treatment_labels = np.array([metadata_map[x[-3:]] for x in well_ids])
    mapping_dict_treat = {'2mM_Ca': '2 mM Ca2+', '2mM_Mg': '2 mM Mg2+', 'TeNT': 'TeNT', 'negative_control': 'Negative Control', 'TTX': 'TTX'}
    treatment_labels = np.array([mapping_dict_treat[x] for x in treatment_labels])
    cell_line = ['WTC11-WT-Tau' if int(x[-2:]) % 2 == 0 else 'WTC11-V337M-Tau' for x in well_ids]
    obs = pd.DataFrame({'well_id': well_ids, 'fov_id': fov_ids, 'treatment': treatment_labels, 'cell_line': cell_line})
    adata = ad.AnnData(X=all_embeddings, obs=obs)
    adata.write_h5ad(f"{base_path}/plexus_embeddings/neuroactive_stimulation/chronos_embed_neuroactive_stim.h5ad")

    # Plotting the Performance for the different treatments
    palette = sns.color_palette(['#8ea1ab', 
                         '#d6278a', 
                         '#14db5d', 
                         '#d67c0d',
                         '#7c4fd6'])
    hue_order = ['Negative Control', '2 mM Ca2+', '2 mM Mg2+','TeNT', 'TTX']
    treat_color_dict = {k: v for k, v in zip(hue_order, palette)}

    adata_stim_embed = adata
    adata_stim_embed.obs['condition'] = adata_stim_embed.obs['cell_line'].copy().astype('str') + '-' + adata_stim_embed.obs['treatment'].astype('str').copy()
    adata_stim_embed_wt = adata_stim_embed[adata_stim_embed.obs["cell_line"] == "WTC11-WT-Tau"]
    adata_stim_embed_wt.obs['location_id'] = adata_stim_embed_wt.obs['condition'].astype('str') + '-' + adata_stim_embed_wt.obs['well_id'].astype('str')

    X_dict = {}
    y_dict = {}
    groups_dict = {}
    treatment_nt = 'Negative Control'
    treatments_to_test = [t for t in adata_stim_embed_wt.obs['treatment'].unique() if t != treatment_nt]

    for treatment in treatments_to_test:
        adata_d14_wt_treat_temp = adata_stim_embed_wt[adata_stim_embed_wt.obs['treatment'].isin([treatment, treatment_nt])]
        X_dict[treatment] = adata_d14_wt_treat_temp.X
        y_dict[treatment] = adata_d14_wt_treat_temp.obs['treatment'].apply(lambda x: 1 if x == treatment else 0).values
        groups_dict[treatment] = list(adata_d14_wt_treat_temp.obs['location_id'].values)

    _ = plot_roc_curves_with_ci(X_dict, y_dict, groups_dict, treat_color_dict)
    plt.savefig('roc_curves_neuroactive_stimulation_chronos_embeddings.pdf', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()