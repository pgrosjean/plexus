from typing import Tuple, Dict, Union
import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import zarr

# Plexus Imports
from plexus.ssl_training.utils.train_utils import TrainConfig

# Constants
# CHANGE THIS BASE PATH TO THE LOCATION OF THE DATASETS ON YOUR SYSTEM  P
BASE_PATH = '/home/plexus_datasets/'

NEUROACTIVE_ZARR_PATH = f"{BASE_PATH}/neuroactive_stimulation/zarr_files/"
NEUROACTIVE_DATASET_NO_SPLIT_STATS_DICT = {"d14_delta": (0.036551, 0.027913),
                                  "d17_delta": (0.051059, 0.041993),
                                  "d21_delta": (0.083181, 0.092586)}

NEUROACTIVE_ZARR_PATH = f"{BASE_PATH}/neuroactive_stimulation/zarr_files_split/"
NEUROACTIVE_DATASET_STATS_DICT = {"d14_delta_first_half_signal": (0.036551, 0.027913),
                                  "d14_delta_second_half_signal": (0.036551, 0.027913),
                                  "d17_delta_first_half_signal": (0.051059, 0.041993),
                                  "d17_delta_second_half_signal": (0.051059, 0.041993),
                                  "d21_delta_first_half_signal": (0.083181, 0.092586),
                                  "d21_delta_second_half_signal": (0.083181, 0.092586)}

FULL_DFF_GCAMP_ZARR_PATH = f"{BASE_PATH}/full_screen/zarr_files_split/"
SIMULATION_2_ZARR_PATH = f"{BASE_PATH}/simulation/zarr_files/"

D14_DFF_DATASET_STATS_DICT = {"Plate1_A1_delta_first_half_signal": (0.094651, 0.093178),
                              "Plate1_A1_delta_second_half_signal": (0.094651, 0.093178),
                              "Plate2_A2_delta_first_half_signal": (0.100261, 0.097043),
                                "Plate2_A2_delta_second_half_signal": (0.100261, 0.097043),
                                "Plate3_B1_delta_first_half_signal": (0.091021, 0.090452),
                                "Plate3_B1_delta_second_half_signal": (0.091021, 0.090452),
                                "Plate4_B2_delta_first_half_signal": (0.096454, 0.098339),
                                "Plate4_B2_delta_second_half_signal": (0.096454, 0.098339),
                                "Plate5_A1_delta_first_half_signal": (0.023678, 0.026054),
                                "Plate5_A1_delta_second_half_signal": (0.023678, 0.026054),
                                "Plate6_A2_delta_first_half_signal": (0.022302, 0.023459),
                                "Plate6_A2_delta_second_half_signal": (0.022302, 0.023459),
                                "Plate7_B1_delta_first_half_signal": (0.020054, 0.021148),
                                "Plate7_B1_delta_second_half_signal": (0.020054, 0.021148),
                                "Plate8_B2_delta_first_half_signal": (0.022764, 0.024016),
                                "Plate8_B2_delta_second_half_signal": (0.022764, 0.024016)}

D21_DFF_DATASET_STATS_DICT = {"Plate1_A1_delta_1200stimwin_first_half_signal": (0.112126, 0.108346),
                              "Plate1_A1_delta_1200stimwin_second_half_signal": (0.112126, 0.108346),
                                "Plate2_A2_delta_1200stimwin_first_half_signal": (0.107874, 0.113154),
                                "Plate2_A2_delta_1200stimwin_second_half_signal": (0.107874, 0.113154),
                                "Plate3_B1_delta_1200stimwin_first_half_signal": (0.107024, 0.116887),
                                "Plate3_B1_delta_1200stimwin_second_half_signal": (0.107024, 0.116887),
                                "Plate4_B2_delta_1200stimwin_first_half_signal": (0.110841, 0.124655),
                                "Plate4_B2_delta_1200stimwin_second_half_signal": (0.110841, 0.124655),
                                "Plate5_A1_delta_1200stimwin_first_half_signal": (0.049153, 0.054843),
                                "Plate5_A1_delta_1200stimwin_second_half_signal": (0.049153, 0.054843),
                                "Plate6_A2_delta_1200stimwin_first_half_signal": (0.045409, 0.047129),
                                "Plate6_A2_delta_1200stimwin_second_half_signal": (0.045409, 0.047129),
                                "Plate7_B1_delta_1200stimwin_first_half_signal": (0.044288, 0.048469),
                                "Plate7_B1_delta_1200stimwin_second_half_signal": (0.044288, 0.048469),
                                "Plate8_B2_delta_1200stimwin_first_half_signal": (0.043195, 0.046206),
                                "Plate8_B2_delta_1200stimwin_second_half_signal": (0.043195, 0.046206)}

SIMULATION_2_DATASET_STATS_DICT = {"simulation_2": (0.123046, 0.101397)}
SIMULATION_3_DATASET_STATS_DICT = {"simulation_3": (0.146757, 0.131694)}
FULL_DFF_DATASET_STATS_DICT = {**D14_DFF_DATASET_STATS_DICT, **D21_DFF_DATASET_STATS_DICT}


class NetworkDataset(Dataset):
    def __init__(self, zarr_files, zarr_paths, num_cells, dataset_stats):
        self.root = zarr_files
        self.paths = zarr_paths
        self.n_cells = num_cells
        self.dataset_stats = dataset_stats
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        root = zarr.open(self.root[idx], 'r')
        full_signal = np.array(root[self.paths[idx]]['signal'])
        sampled_inds = np.random.choice(np.arange(full_signal.shape[0]), self.n_cells, replace=False)
        sampled_signal = full_signal[sampled_inds, :]  # Shape: [n_cells, len_signal]
        ds = self.dataset_stats
        signal = torch.Tensor(sampled_signal) - ds[0]
        signal = signal / ds[1]
        return signal
    
class NetworkDatasetPlateStats(Dataset):
    def __init__(self, zarr_files, zarr_paths, num_cells, dataset_stats_dict):
        self.root = zarr_files
        self.zarr_name = np.array([x.split('/')[-1].split('.')[0] for x in zarr_files])
        self.paths = zarr_paths
        self.n_cells = num_cells
        self.dataset_stats_dict = dataset_stats_dict
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        root = zarr.open(self.root[idx], 'r')
        full_signal = np.array(root[self.paths[idx]]['signal'])
        sampled_inds = np.random.choice(np.arange(full_signal.shape[0]), self.n_cells, replace=False)
        sampled_signal = full_signal[sampled_inds, :]
        ds = self.dataset_stats_dict[self.zarr_name[idx]]
        signal = torch.Tensor(sampled_signal) - ds[0]
        signal = signal / ds[1]
        return signal


def find_zarr_paths(root) -> np.ndarray:
    """
    List all the groups in a Zarr file.

    Returns
    -------
    np.ndarray
        An array of all the groups present in the Zarr file.
    """
    # Setting the zarr root
    root_group = root

    # Initialize a list to hold the deepest group names
    deepest_group_names = []

    # Recursive function to traverse the Zarr group hierarchy
    def traverse_group(group, path):
        child_groups = []
        for name, item in group.items():
            new_path = f"{path}/{name}" if path else name
            if isinstance(item, zarr.Group):
                child_groups.append((item, new_path))

        if child_groups:
            for child_group, new_path in child_groups:
                traverse_group(child_group, new_path)
        else:
            deepest_group_names.append(path)
            
    # Start the traversal from the root group
    traverse_group(root_group, "")
    
    return np.array(deepest_group_names)


def find_zarr_files(path):
    """
    Recursively search for .zarr files within a given directory.

    Parameters
    ----------
    path : str
        The root directory path to start searching from.

    Returns
    -------
    zarr_files : list
        A list of paths to .zarr files found within the directory tree.
    """
    zarr_files = []
    # Walk through the directory
    for root, dirs, _ in os.walk(path):
        for dir_ in dirs:
            if dir_.endswith('.zarr'):
                zarr_files.append(os.path.join(root, dir_))
    return zarr_files


def generate_network_ssl_datasets(zarr_path,
                                  dataset_stats: Union[Tuple[float, float], Dict[str, Tuple[float, float]]],
                                  seed=14,
                                  num_cells=8,
                                  ):
    np.random.seed(seed)
    zfs = find_zarr_files(zarr_path)
    all_paths = []
    zarr_files = []
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

    all_paths = np.hstack(all_paths)
    well_paths = np.array([x.split('/')[0] for x in all_paths])
    zarr_files = np.hstack(zarr_files)
    all_ups = np.array([x + '-' + y for x, y in zip(zarr_files, well_paths)])
    unique_paths = np.unique(np.array([x + '-' + y for x, y in zip(zarr_files, well_paths)]))
    # Performing 80-20 train-test split on well level
    train_ups = np.random.choice(unique_paths, int(0.8*len(unique_paths)), replace=False)     
    val_ups = np.array(list(set(list(unique_paths)) - set(list(train_ups))))
    train_inds = np.where(np.isin(all_ups, train_ups))[0]
    val_inds = np.where(np.isin(all_ups, val_ups))[0]
    # Using the train and validation indices to pull out the train and validation sets
    train_paths = all_paths[train_inds]
    train_zarr_files = zarr_files[train_inds]
    val_paths = all_paths[val_inds]
    val_zarr_files = zarr_files[val_inds]
    # Constructing the Datasets to return
    if isinstance(dataset_stats, tuple):
        train_ds = NetworkDataset(train_zarr_files,
                                train_paths,
                                num_cells=num_cells,
                                dataset_stats=dataset_stats)
        val_ds = NetworkDataset(val_zarr_files,
                                val_paths,
                                num_cells=num_cells,
                                dataset_stats=dataset_stats)
    else:
        train_ds = NetworkDatasetPlateStats(train_zarr_files,
                                            train_paths,
                                            num_cells=num_cells,
                                            dataset_stats_dict=dataset_stats)
        val_ds = NetworkDatasetPlateStats(val_zarr_files,
                                            val_paths,
                                            num_cells=num_cells,
                                            dataset_stats_dict=dataset_stats)
    return train_ds, val_ds


def generate_dataloader(train_config: TrainConfig,
                        dataset: Dataset,
                        shuffle: bool = True) -> DataLoader:
    """
    This function generates a PyTorch DataLoader object for training a model.

    Parameters
    ----------
    train_config : TrainConfig
        An object containing the training configuration parameters.
    dataset : Dataset
        The dataset object to be used for training.
    
    Returns
    -------
    DataLoader
        A PyTorch DataLoader object for training the model.
    """
    num_workers = train_config.num_workers
    batch_size = train_config.batch_size
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader


################################################################################
################################################################################
## Neuroactive Compound Data Module
################################################################################
################################################################################


def generate_neuroactive_datasets(num_cells):
    """
    This function generates the training and validation datasets for the neuroactive compound dataset.
    """
    train_ds, val_ds = generate_network_ssl_datasets(
                                zarr_path=NEUROACTIVE_ZARR_PATH,
                                dataset_stats=NEUROACTIVE_DATASET_STATS_DICT,
                                seed=14,
                                num_cells=num_cells)
    return train_ds, val_ds


def generate_neuroactive_dataloaders(train_config: TrainConfig,
                                     num_cells: int) -> Tuple[DataLoader, DataLoader]:
    """
    This function generates PyTorch DataLoader objects for training and validation
    for the neuroactive compound dataset.

    Parameters
    ----------
    train_config : TrainConfig
        An object containing the training configuration parameters.
    
    Returns
    -------
    train_dl : DataLoader
        A PyTorch DataLoader object for training the model.
    val_dl : DataLoader
        A PyTorch DataLoader object for validation.
    """
    train_ds, val_ds = generate_neuroactive_datasets(num_cells)
    train_dl = generate_dataloader(train_config, train_ds)
    val_dl = generate_dataloader(train_config, val_ds)
    return train_dl, val_dl


class NeuroactiveCompoundDataModule(pl.LightningDataModule):
    def __init__(self, train_config: TrainConfig, num_cells: int):
        super().__init__()
        self.train_config = train_config
        self.train_ds, self.val_ds = generate_neuroactive_datasets(num_cells)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)
    

################################################################################
################################################################################
## Full GCaMP Screen (D14 + D21) 4 cell line Data Module
################################################################################
################################################################################

def generate_full_screen_datasets(num_cells):
    """
    This function generates the training and validation datasets for the neuroactive compound dataset.
    """
    train_ds, val_ds = generate_network_ssl_datasets(
                                zarr_path=FULL_DFF_GCAMP_ZARR_PATH,
                                dataset_stats=FULL_DFF_DATASET_STATS_DICT,
                                seed=14,
                                num_cells=num_cells)
    return train_ds, val_ds


class ScreenFullDataModule(pl.LightningDataModule):
    def __init__(self, train_config: TrainConfig, num_cells: int):
        super().__init__()
        self.train_config = train_config
        self.train_ds, self.val_ds = generate_full_screen_datasets(num_cells)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)
    


################################################################################
################################################################################
## Simulation 2 Data Module
################################################################################
################################################################################

def generate_simulation_2_datasets(num_cells):
    """
    This function generates the training and validation datasets for the neuroactive compound dataset.
    """
    train_ds, val_ds = generate_network_ssl_datasets(
                                zarr_path=SIMULATION_2_ZARR_PATH,
                                dataset_stats=SIMULATION_2_DATASET_STATS_DICT,
                                seed=14,
                                num_cells=num_cells)
    return train_ds, val_ds

class Simulation2DataModule(pl.LightningDataModule):
    def __init__(self, train_config: TrainConfig, num_cells: int):
        super().__init__()
        self.train_config = train_config
        self.train_ds, self.val_ds = generate_simulation_2_datasets(num_cells)

    def train_dataloader(self):
        return generate_dataloader(self.train_config, self.train_ds)

    def val_dataloader(self):
        return generate_dataloader(self.train_config, self.val_ds, shuffle=False)
    