import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from plexus.ml_baselines.pretrained_ts_model_neuroactive import get_zarr_info, plot_roc_curves_with_ci
from plexus.ssl_training.datasets import generate_neuroactive_datasets
from plexus.ml_baselines.cnn_autoencoder_simulation import CNN1DAutoencoder


def main():
    base_path = "../../plexus_data_archive"
    model = CNN1DAutoencoder()
    # Load the datasets
    train_ds, val_ds = generate_neuroactive_datasets(1)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=24, num_workers=2)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=24, num_workers=2, shuffle=False)
    wandb_logger = WandbLogger(project="Plexus_baselines")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=100, accelerator=device, logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)
    # Running inference
    zarr_info = get_zarr_info("../../plexus_data_archive/processed_zarr_files/simulation/")
    all_embeddings = []
    well_ids = []
    fov_ids = []
    model.eval()
    with torch.no_grad():
        for zarr_root in zarr_info.values():
            for well in tqdm(list(zarr_root.keys())):
                for fov in list(zarr_root[well].keys()):
                    signal = torch.Tensor(np.array(zarr_root[well][fov]['signal']))
                    if len(signal.shape) == 1:
                        signal = signal.unsqueeze(0)
                    embeddings, x_hat = model(signal)
                    all_embeddings.append(embeddings.float().detach().cpu().numpy())
                    fov_ids.append([fov]*len(embeddings))
                    well_ids.append([well]*len(embeddings))
    all_embeddings = np.vstack(all_embeddings)
    well_ids = np.hstack(well_ids)
    fov_ids = np.hstack(fov_ids)
    metadata_df = pd.read_csv(f"{base_path}/metadata_files/neuroactive_stimulation_metadata.csv")
    metadata_map = {k: v for k, v in zip(metadata_df['well_id'].values, metadata_df['treatment'].values)}
    treatment_labels = np.array([metadata_map[x[-3:]] for x in well_ids])
    # making nicer treatement labels
    mapping_dict_treat = {'2mM_Ca': '2 mM Ca2+', '2mM_Mg': '2 mM Mg2+', 'TeNT': 'TeNT', 'negative_control': 'Negative Control', 'TTX': 'TTX'}
    treatment_labels = np.array([mapping_dict_treat[x] for x in treatment_labels])
    cell_line = ['WTC11-WT-Tau' if int(x[-2:]) % 2 == 0 else 'WTC11-V337M-Tau' for x in well_ids]
    obs = pd.DataFrame({'well_id': well_ids, 'fov_id': fov_ids, 'treatment': treatment_labels, 'cell_line': cell_line})
    adata = ad.AnnData(X=all_embeddings, obs=obs)
    adata.write_h5ad(f"{base_path}/plexus_embeddings/1dcnn_embed_neuroactive_stim.h5ad")
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
    plt.savefig('roc_curves_neuroactive_stimulation_1DCNNAE_embeddings.pdf', dpi=800)
    plt.show()


if __name__ == "__main__":
    main()
    
