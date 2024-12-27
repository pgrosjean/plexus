from typing import Tuple, Union, Dict
import zarr
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import random

from plexus.ssl_training.datasets import (find_zarr_files,
                                                find_zarr_paths)


class NetworkContextualizedSingleCellDataset(Dataset):
    def __init__(self,
                 zarr_files: np.ndarray,
                 zarr_paths: np.ndarray,
                 n_cells_per_fov: np.ndarray,
                 n_cells: int,
                 dataset_stats: Union[Tuple[float, float], Dict[str, Tuple[float, float]]],
                 only_nuclei_positive: bool = False
                 ):
        self.n_cells = n_cells
        self.dataset_stats = dataset_stats
        self.only_nuclei_positive = only_nuclei_positive
        self.root, self.paths, self.cell_indices = self._construct_sampling_strategy(zarr_files,
                                                                                     zarr_paths,
                                                                                     n_cells_per_fov,
                                                                                     only_nuclei_positive)
        
    def _construct_sampling_strategy(self,
                                     roots: np.ndarray,
                                     paths: np.ndarray,
                                     n_cells_per_fov: np.ndarray,
                                     only_nuclei_positive: bool):
        root_list = []
        path_list = []
        idx_list = []
        print("Constructing Sampling Strategy for Inference...")
        if only_nuclei_positive:
            for zf, path, nc_pfv in tqdm(zip(roots, paths, n_cells_per_fov), total=len(roots)):
                root = zarr.open(zf, 'r')
                nuc_pos = np.where(np.array(root[path]['contains_nuclei']))[0]
                for c in nuc_pos:
                    bg_subset = list(np.arange(nc_pfv))
                    bg_subset.remove(c)
                    for _ in range(10):
                        bg_choice = np.random.choice(bg_subset, self.n_cells-1, replace=False)
                        subset = [c] + list(bg_choice)
                        subset = np.array(subset)
                        root_list.append(str(zf))
                        path_list.append(str(path))
                        idx_list.append(subset)
        else:
            for root, path, nc_pfv in tqdm(zip(roots, paths, n_cells_per_fov), total=len(roots)):
                if nc_pfv == self.n_cells:
                    for c in range(nc_pfv):
                        subset = list(np.arange(nc_pfv))
                        subset.remove(c)
                        subset = [c] + subset
                        subset = np.array(subset)
                        root_list.append(str(root))
                        path_list.append(str(path))
                        idx_list.append(subset)
                else:
                    # sample combinations of n_cells from n_cells_per_fov for each cell
                    all_subsets = sample_random_subsets(np.arange(nc_pfv), self.n_cells-1, 10)
                    for subset in all_subsets:
                        assert len(subset) == self.n_cells, "Length of subset is not equal to n_cells"
                        root_list.append(str(root))
                        path_list.append(str(path))
                        idx_list.append(subset)
        print("Stacking the lists...")
        print("Length of root_list: ", len(root_list))
        current_max = 0
        for i in tqdm(range(len(idx_list))):
            current_max = max(current_max, len(idx_list[i]))
        root_list = np.hstack(root_list)
        path_list = np.hstack(path_list)
        idx_list = np.vstack(idx_list)
        return root_list, path_list, idx_list
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        root = zarr.open(self.root[idx], 'r')
        full_signal = np.array(root[self.paths[idx]]['signal'])
        sampled_inds = self.cell_indices[idx]
        sampled_signal = full_signal[sampled_inds, :]  # Shape: [n_cells, len_signal]
        ds = self.dataset_stats
        if isinstance(ds, dict):
            key = self.root[idx].split('/')[-1].split('.')[0]
            ds = ds[key]
        signal = torch.Tensor(sampled_signal) - ds[0]
        signal = signal / ds[1]
        return signal
    

def generate_inference_dataset(zarr_path: str,
                               dataset_stats: Union[Tuple[float, float], Dict[str, Tuple[float, float]]],
                               only_nuclei_positive: bool = False,
                               seed: int = 14,
                               num_cells: int = 8):
    """
    This function generates a dataset for inference from Zarr files.

    Parameters
    ----------
    zarr_path : str
        The path to the Zarr files.
    dataset_stats : Tuple[float, float]
        The mean and standard deviation of the dataset.
    seed : int, optional
        The random seed to use, by default 14.
    num_cells : int, optional
        The number of cells to sample for each FOV, by default 8.
    """
    np.random.seed(seed)
    zfs = find_zarr_files(zarr_path)
    all_paths = []
    zarr_files = []
    n_cells_per_fov = []
    print('Collating Zarr Files...')
    for zarr_file in tqdm(zfs):
        paths = find_zarr_paths(zarr.open(zarr_file , 'r'))
        paths = sorted([x for x in paths if len(x.split('/')) == 2])
        root = zarr.open(zarr_file, 'r')
        for path in paths:
            n_cells, _ = root[path]['signal'].shape
            if n_cells >= num_cells:
                all_paths.append(path)
                zarr_files.append(zarr_file)
                n_cells_per_fov.append(n_cells)
    all_paths = np.hstack(all_paths)
    n_cells_per_fov = np.hstack(n_cells_per_fov)
    zarr_files = np.hstack(zarr_files)
    # Constructing the Datasets to return
    dataset = NetworkContextualizedSingleCellDataset(zarr_files,
                                                     all_paths,
                                                     n_cells_per_fov,
                                                     num_cells,
                                                     dataset_stats=dataset_stats,
                                                     only_nuclei_positive=only_nuclei_positive)
    return dataset


def sample_random_subsets(arr, k, z):
    """
    Sample `z` random unique subsets of `k` other cells from the `n_cells` array for each cell.
    For each subset, the cell of interest (the current cell) will be in the 0th index, and the 
    remaining indices will correspond to the sampled cells.
    
    Parameters
    ----------
    arr : np.ndarray
        The input array of shape (n_cells, signal_len), where each row represents a cell, 
        and each column represents the signal for that cell.
    k : int
        The number of other cells to sample for each cell.
    z : int
        The number of random subsets to sample for each cell.

    Returns
    -------
    list of np.ndarray
        A list where each element is an ndarray of shape (k+1,), representing a subset of indices.
        The 0th index in each array corresponds to the cell of interest (the current cell), and 
        the remaining `k` indices correspond to the randomly sampled other cells.
    """
    n_cells = arr.shape[0]
    
    # List to store the subsets for each cell
    all_subsets = []

    for i in range(n_cells):
        # Generate the list of other cell indices excluding the current cell i
        other_cells = list(range(n_cells))
        other_cells.remove(i)
        
        # Set to track seen combinations
        seen_combinations = set()
        
        # List to store the sampled subsets for this specific cell
        sampled_subsets = []
        
        count = 0
        # Continue sampling until we get z unique subsets
        max_combinations = 1000
        while len(sampled_subsets) < z and count < max_combinations:
            # Randomly sample k cells from the remaining other_cells
            subset = tuple(sorted(random.sample(other_cells, k)))
            
            # If the subset has not been seen before, add it to the result
            if subset not in seen_combinations:
                seen_combinations.add(subset)
                # Prepend the current cell index i at the 0th position
                subset_with_cell = np.array([i] + list(subset))
                sampled_subsets.append(subset_with_cell)
            else:
                count += 1
        
        # Append the subsets for this cell to the final list
        all_subsets.extend(sampled_subsets)

    return all_subsets