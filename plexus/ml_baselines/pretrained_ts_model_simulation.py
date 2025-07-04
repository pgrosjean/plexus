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
import matplotlib.pyplot as plt
import seaborn as sns
from plexus.ml_baselines.pretrained_ts_model_neuroactive import get_zarr_info


def main():
    zarr_info = get_zarr_info("../../plexus_data_archive/processed_zarr_files/simulation/")
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
    metadata_df = pd.read_csv("../../plexus_data_archive/metadata_files/simulation_metadata.csv")
    metadata_df['paths'] = metadata_df['well_id'].astype('str') + '/' + metadata_df['fov_id'].astype('str')
    metadata_df = metadata_df.rename(columns={'phenotype_number': 'simulation_phenotype'})
    metadata_df['well_group'] = metadata_df['zarr_file'].astype('str') + '_' + metadata_df['well_id'].astype('str')
    metadata_map = {k: v for k, v in zip(metadata_df['well_id'].values, metadata_df['simulation_phenotype'].values)}
    pheno_labels = np.array([metadata_map[x] for x in well_ids])
    obs = pd.DataFrame({'well_id': well_ids, 'fov_id': fov_ids, 'simulation_phenotype': pheno_labels})
    obs['zarr_file'] = ['simulation_2']*len(obs)
    obs['paths'] = obs['well_id'].astype('str') + '/' + obs['fov_id'].astype('str')
    obs['well_group'] = obs['zarr_file'].astype('str') + '_' + obs['well_id'].astype('str')
    adata = ad.AnnData(X=all_embeddings, obs=obs)
    adata.write_h5ad("../../plexus_data_archive/plexus_embeddings/simulations/chronos_embed_simulation_2.h5ad")


if __name__ == "__main__":
    main()