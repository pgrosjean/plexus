import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from plexus.ml_baselines.pretrained_ts_model_neuroactive import plot_roc_curves_with_ci


def main():
    base_path = "../../plexus_data_archive"
    new_embeddings = pd.read_parquet(f'{base_path}/plexus_embeddings/neuroactive_stimulation/Embeddings_from_checkpoint_8_cells.parquet')
    register_tokens = np.load(f'{base_path}/plexus_embeddings/neuroactive_stimulation/register_tokens_neuroactive.npy')
    ablation_embeddings = pd.read_parquet(f'{base_path}/plexus_embeddings/neuroactive_stimulation/Embeddings_rh2gvwis_8_cells.parquet')
    cell_indices = np.array([list(x)[0] for x in new_embeddings['cell_indices'].values])
    fov_info = new_embeddings['paths'].values
    agg_info = np.array([x.astype(str) + '_' + y for x, y in zip(cell_indices, fov_info)])
    embeddings = np.vstack([np.array(list(x)) for x in new_embeddings['embeddings']])
    abl_embeddings = np.vstack([np.array(list(x)) for x in ablation_embeddings['embeddings']])
    cls_tokens = np.vstack([np.array(list(x)) for x in new_embeddings['cls_tokens']])
    # Aggregating the embeddings
    mean_embed = []
    mean_embed_abl = []
    mean_cls = []
    mean_register = {}
    for i in range(register_tokens.shape[0]):
        mean_register[i] = []
    well_ids = []
    fov_ids = []
    for agg in tqdm(np.unique(agg_info)):
        mean_embed.append(np.sum(embeddings[agg_info==agg], axis=0))
        mean_cls.append(np.sum(cls_tokens[agg_info==agg], axis=0))
        mean_embed_abl.append(np.sum(abl_embeddings[agg_info==agg], axis=0))
        well_ids.append(fov_info[agg_info==agg][0].split('/')[0])
        fov_ids.append(fov_info[agg_info==agg][0].split('/')[1])
        for i in range(register_tokens.shape[0]):
            mean_register[i].append(np.sum(register_tokens[i][agg_info==agg], axis=0))
    mean_embed = np.vstack(mean_embed)
    mean_cls = np.vstack(mean_cls)
    fov_ids = np.hstack(fov_ids)
    well_ids = np.hstack(well_ids)
    mean_embed_abl = np.vstack(mean_embed_abl)
    for i in range(register_tokens.shape[0]):
        mean_register[i] = np.vstack(mean_register[i])
    
    # Reading in the metadata
    metadata_df = pd.read_csv(f"{base_path}/plexus_data_archive/metadata_files/neuroactive_stimulation_metadata.csv")
    metadata_map = {k: v for k, v in zip(metadata_df['well_id'].values, metadata_df['treatment'].values)}
    treatment_labels = np.array([metadata_map[x[-3:]] for x in well_ids])
    # making nicer treatement labelsd 
    mapping_dict_treat = {'2mM_Ca': '2 mM Ca2+', '2mM_Mg': '2 mM Mg2+', 'TeNT': 'TeNT', 'negative_control': 'Negative Control', 'TTX': 'TTX'}
    treatment_labels = np.array([mapping_dict_treat[x] for x in treatment_labels])
    cell_line = ['WTC11-WT-Tau' if int(x[-2:]) % 2 == 0 else 'WTC11-V337M-Tau' for x in well_ids]
    obs = pd.DataFrame({'well_id': well_ids, 'fov_id': fov_ids, 'treatment': treatment_labels, 'cell_line': cell_line})
    adata = ad.AnnData(X=mean_embed, obs=obs)
    adata_cls = ad.AnnData(X=mean_cls, obs=obs)
    adata_ablation = ad.AnnData(X=mean_embed_abl, obs=obs)
    adata_reg0 = ad.AnnData(X=mean_register[0], obs=obs)
    adata_reg1 = ad.AnnData(X=mean_register[1], obs=obs)
    adata_reg2 = ad.AnnData(X=mean_register[2], obs=obs)
    adata_reg3 = ad.AnnData(X=mean_register[3], obs=obs)
    adata_reg4 = ad.AnnData(X=mean_register[4], obs=obs)

    palette = sns.color_palette(['#8ea1ab', 
                         '#d6278a', 
                         '#14db5d', 
                         '#d67c0d',
                         '#7c4fd6'])
    hue_order = ['Negative Control', '2 mM Ca2+', '2 mM Mg2+','TeNT', 'TTX']
    treat_color_dict = {k: v for k, v in zip(hue_order, palette)}

    save_str = ["", "cls_", "abl_", "reg0_", "reg1_", "reg2_", "reg3_", "reg4_"]
    adata_list = [adata, adata_cls, adata_ablation, adata_reg0, adata_reg1, adata_reg2, adata_reg3, adata_reg4]
    for i, adata_stim_embed in enumerate(adata_list):
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
        plt.savefig(f'{save_str[i]}roc_curves_neuroactive_stimulation_CRISPRi_OOD_embeddings.pdf', dpi=800)
        plt.show()
    