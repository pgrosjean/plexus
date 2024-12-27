import zarr
import os
from tqdm import tqdm


def get_zarr_info(zarr_location):
    root_dict = {}
    for zarr_file in os.listdir(zarr_location):
        print(f'{zarr_location}{zarr_file}')
        if zarr_file.endswith('.zarr'):
            root_dict[zarr_file.split('.')[0]] = zarr.open(f'{zarr_location}{zarr_file}', 'r')
    return root_dict


def split_zarr_for_training(zarr_file, zarr_root, output_dir, time_window=1200):
    """
    This function splits the signals in the zarr file into two halves, 
    the first half and the second half.

    Parameters
    ----------
    zarr_file : str
        The path to the zarr file.
    zarr_root : zarr.hierarchy.Group
        The root of the zarr file.
    output_dir : str
        The path to the output directory.
    time_window : int
        The time window to split the signals.
    
    Returns
    -------
    None
    """
    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]
    # Load the zarr file
    zarr_file_prefix = zarr_file
    new_root_1 = zarr.open(output_dir + f'/{zarr_file_prefix}_first_half_signal.zarr', mode='w')
    new_root_2 = zarr.open(output_dir + f'/{zarr_file_prefix}_second_half_signal.zarr', mode='w')
    for well in zarr_root.keys():
        grp1 = new_root_1.create_group(well)
        grp2 = new_root_2.create_group(well)
        for fov in zarr_root[well].keys():
            fov_grp1 = grp1.create_group(fov)
            fov_grp2 = grp2.create_group(fov)
            full_signal = zarr_root[well][fov]['signal'] # shape [cell_num, time]
            raw_signal = zarr_root[well][fov]['raw_signal'] # shape [cell_num, time]
            ic_signal = zarr_root[well][fov]['inferred_calcium'] # shape [cell_num, time]
            signal_1 = full_signal[:, :time_window] # shape [cell_num, time_window]
            raw_signal_1 = raw_signal[:, :time_window] # shape [cell_num, time_window]
            ic_signal_1 = ic_signal[:, :time_window] # shape [cell_num, time_window]
            signal_2 = full_signal[:, -time_window:] # shape [cell_num, time_window]
            raw_signal_2 = raw_signal[:, -time_window:] # shape [cell_num, time_window]
            ic_signal_2 = ic_signal[:, -time_window:]
            fov_grp1.create_dataset('signal', data=signal_1)
            fov_grp1.create_dataset('raw_signal', data=raw_signal_1)
            if 'contains_nuclei' in zarr_root[well][fov].keys():
                fov_grp1.create_dataset('contains_nuclei', data=zarr_root[well][fov]['contains_nuclei'])
            fov_grp1.create_dataset('inferred_calcium', data=ic_signal_1)
            fov_grp2.create_dataset('signal', data=signal_2)
            fov_grp2.create_dataset('raw_signal', data=raw_signal_2)
            if 'contains_nuclei' in zarr_root[well][fov].keys():
                fov_grp2.create_dataset('contains_nuclei', data=zarr_root[well][fov]['contains_nuclei'])
            fov_grp2.create_dataset('inferred_calcium', data=ic_signal_2)
    

def split_all_zarr_files(zarr_dict, output_dir, time_window=1200):
    """
    This function splits all the zarr files in the zarr_dict into two halves, 
    the first half and the second half.

    Parameters
    ----------
    zarr_dict : dict
        The dictionary containing the zarr files.
    output_dir : str
        The path to the output directory.
    time_window : int
        The time window to split the signals.
    
    Returns
    -------
    None
    """
    print("Splitting zarr files...")
    for zarr_file in tqdm(zarr_dict.keys()):
        zarr_root = zarr_dict[zarr_file]
        split_zarr_for_training(zarr_file, zarr_root, output_dir, time_window)
