import os
import pandas as pd
import numpy as np
import torch
import hydra
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import json
from plexus.ssl_inference.utils.inference_data_utils import generate_inference_dataset
from plexus.ssl_inference.utils.model_load import load_model_from_wandb_for_inference


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--wandb_entity", type=str, help="WandB entity where the model is stored.")
    argparser.add_argument("--wandb_uid", type=str, help="WandB unique identifier for the model.")
    argparser.add_argument("--zarr_path", type=str, help="Path to the Zarr file.")
    argparser.add_argument("--config", type=str, help="Path to the configuration file used during training.")
    argparser.add_argument("--save_path", type=str, help="Path to save the embeddings.")
    # Adding an optional gpu_num argument to specify the GPU to use for inference if desired
    argparser.add_argument("--gpu_num", type=int, default=0, help="GPU number to use for inference.")
    argparser.add_argument("--network_average", action="store_true", help="Use network average instead of cell average.")
    argparser.add_argument("--dataset_stats_json", type=str, help="Path to the JSON file containing the dataset mean and std per plate.")
    argparser.add_argument("--dataset_mean", type=float, help="Dataset mean (float). Only used if dataset_stats_json is not provided.")
    argparser.add_argument("--dataset_std", type=float, help="Dataset std (float). Only used if dataset_stats_json is not provided.")
    argparser.add_argument("--seed", type=int, default=14, help="Random seed to use for generating network samples. (default: 14)")
    argparser.add_argument("--only_nuclei_positive", action="store_true", help="Only use nuclei positive cells.")
    args = argparser.parse_args()
    network_average = args.network_average

    if args.dataset_stats_json:
        # reading json into a dictionary
        with open(args.dataset_stats_json, "r") as f:
            dataset_stats = json.load(f)
    else:
        dataset_stats = (args.dataset_mean, args.dataset_std)

    config_path = Path(__file__).parent / ".." / "ssl_training" / "config"
    
    full_path = Path.cwd() / config_path
    relative_path = os.path.relpath(full_path, Path(__file__).parent)

    with hydra.initialize(version_base=None, config_path=str(relative_path)):
        config = hydra.compose(config_name=str(args.config))
    
    model = load_model_from_wandb_for_inference(config, args.wandb_entity, args.wandb_uid)
    num_cells = config.model_config.num_channels
    inference_dataset = generate_inference_dataset(zarr_path=args.zarr_path,
                                                   num_cells=num_cells,
                                                   seed=args.seed,
                                                   dataset_stats=dataset_stats,
                                                   only_nuclei_positive=args.only_nuclei_positive)
    print("Generating inference dataset...")
    inference_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=24, shuffle=False)

    embeddings = []
    print(f"Performing inference on the model with checkpoint from WandB {args.wandb_uid}...")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_num}")
        model.to(device)
        num_tokens_per_cell = model.num_patches
        for batch in tqdm(inference_dataloader):
            with torch.no_grad():
                output, _, _, _ = model(batch.to(device), inference=True)
                output = output.cpu().detach().numpy()
                if network_average:
                    output = np.mean(output, axis=1)
                else:
                    output = np.mean(output[:, :num_tokens_per_cell, :], axis=1)
                embeddings.append(output)
    else:
        for batch in tqdm(inference_dataloader):
            with torch.no_grad():
                output, _, _, _ = model(batch)
                output = output.numpy()
                if network_average:
                    output = np.mean(output, axis=1)
                else:
                    output = np.mean(output[:, :num_tokens_per_cell, :], axis=1)
                embeddings.append(output)
    
    embeddings = np.vstack(embeddings)
    paths = inference_dataset.paths
    cell_indices = inference_dataset.cell_indices
    zarr_files = inference_dataset.root
    assert len(embeddings) == len(paths) == len(cell_indices) == len(zarr_files), "Lengths of embeddings, paths, cell_indices, and zarr_files do not match"

    # saving the embeddings to a parquet file
    embeddings_df = pd.DataFrame({"zarr_files": zarr_files,
                                  "paths": paths,
                                  "cell_indices": [ci for ci in cell_indices],
                                  "embeddings": [embed for embed in embeddings]})
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path[-1] == "/":
        save_path = save_path[:-1]
    if network_average:
        embeddings_df.to_parquet(f"{save_path}/MAE_embeddings_{args.wandb_uid}_{num_cells}_cells_network_average.parquet")
    else:
        embeddings_df.to_parquet(f"{save_path}/MAE_embeddings_{args.wandb_uid}_{num_cells}_cells.parquet")


if __name__ == "__main__":
    main()
