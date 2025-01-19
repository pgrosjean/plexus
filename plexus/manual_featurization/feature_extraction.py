import os
import pandas as pd
import numpy as np
import zarr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import entropy


def get_zarr_info(zarr_location: str):
    """
    Walk through the directory tree and save any directories ending in .zarr.
    
    Parameters
    ----------
    zarr_location : str
        The root directory to search for .zarr files.
    
    Returns
    -------
    dict
        A dictionary where the keys are the base names of .zarr directories 
        and the values are the opened zarr objects.
    """
    root_dict = {}
    # Walk through the directory tree
    for root, dirs, _ in os.walk(zarr_location):
        for directory in dirs:
            if directory.endswith('.zarr'):
                # Extract the base name without the .zarr extension
                base_name = directory.split('.')[0]
                # Get the full path to the .zarr directory
                zarr_path = os.path.join(root, directory)
                print(f'Opening {zarr_path}')
                # Open the .zarr directory and store it in the dictionary
                root_dict[base_name] = zarr.open(zarr_path, 'r')
    return root_dict


def extract_neuronal_features(spiking_array: np.ndarray, 
                                calcium_array: np.ndarray, 
                                sampling_rate: float,
                                spike_threshold: float = None, 
                                calcium_threshold: float = None,
                                verbose: bool = False) -> pd.DataFrame:
    """
    Extract comprehensive neuronal features from spiking and calcium activity arrays.

    Parameters
    ----------
    spiking_array : np.ndarray
        Array of shape (n_cells, time) containing deconvolved spiking activity
    calcium_array : np.ndarray
        Array of shape (n_cells, time) containing calcium fluorescence signals
    sampling_rate : float
        Sampling rate of the recordings in Hz
    spike_threshold : float, optional
        Threshold for detecting spikes in spiking_array. If None, uses mean + 2*std
    calcium_threshold : float, optional
        Threshold for detecting peaks in calcium_array. If None, uses mean + 2*std
    verbose : bool, optional
        If True, plots signals and features

    Returns
    -------
    pd.DataFrame
        DataFrame containing features for each cell with dimensions (n_cells, n_features)
        Features include signal statistics, spike characteristics, burst properties,
        and calcium dynamics

    Notes
    -----
    Signal features computed for:
    - Raw calcium signal
    - First derivative (rate of change)
    - Second derivative (acceleration)
    
    Feature categories:
    1. Basic signal statistics (min, max, mean, std)
    2. Peak/trough characteristics
    3. Spiking patterns
    4. Bursting properties
    5. Calcium dynamics
    6. Population synchrony measures
    
    NaN handling:
    - Burst metrics: NaN for undefined ratios/durations
    - Other features: Meaningful defaults (-1 for absent events, 0 for correlations)
    """
    n_cells, n_timepoints = spiking_array.shape
    time_vector = np.arange(n_timepoints) / sampling_rate
    features_list = []

    if verbose:
        _, ax_calcium = plt.subplots(figsize=(10, n_cells * 1))
        _, ax_spiking = plt.subplots(figsize=(10, n_cells * 1))

    # Calculate correlation matrix for population analysis
    corr_matrix = np.corrcoef(calcium_array)

    def extract_signal_stats(signal, prefix):
        """Extract comprehensive statistics for a given signal."""
        stats = {}
        
        # Basic statistics
        stats[f'{prefix}_min'] = np.min(signal)
        stats[f'{prefix}_max'] = np.max(signal)
        stats[f'{prefix}_mean'] = np.mean(signal)
        stats[f'{prefix}_std'] = np.std(signal)
        stats[f'{prefix}_var'] = np.var(signal)
        
        # Peak detection
        peaks, peak_props = find_peaks(signal, prominence=np.std(signal)/2)
        troughs, trough_props = find_peaks(-signal, prominence=np.std(signal)/2)
        
        # Peak statistics
        if len(peaks) > 0:
            stats[f'{prefix}_peak_mean_height'] = np.mean(signal[peaks])
            stats[f'{prefix}_peak_max_height'] = np.max(signal[peaks])
            stats[f'{prefix}_peak_min_height'] = np.min(signal[peaks])
            stats[f'{prefix}_mean_prominence'] = np.mean(peak_props['prominences'])
            stats[f'{prefix}_max_prominence'] = np.max(peak_props['prominences'])
            stats[f'{prefix}_num_peaks'] = len(peaks)
        else:
            stats[f'{prefix}_peak_mean_height'] = stats[f'{prefix}_mean']
            stats[f'{prefix}_peak_max_height'] = stats[f'{prefix}_max']
            stats[f'{prefix}_peak_min_height'] = stats[f'{prefix}_max']
            stats[f'{prefix}_mean_prominence'] = 0
            stats[f'{prefix}_max_prominence'] = 0
            stats[f'{prefix}_num_peaks'] = 0
        
        # Trough statistics
        if len(troughs) > 0:
            stats[f'{prefix}_trough_mean_height'] = np.mean(signal[troughs])
            stats[f'{prefix}_trough_max_height'] = np.max(signal[troughs])
            stats[f'{prefix}_trough_min_height'] = np.min(signal[troughs])
            stats[f'{prefix}_mean_trough_prominence'] = np.mean(trough_props['prominences'])
            stats[f'{prefix}_max_trough_prominence'] = np.max(trough_props['prominences'])
            stats[f'{prefix}_num_troughs'] = len(troughs)
        else:
            stats[f'{prefix}_trough_mean_height'] = stats[f'{prefix}_mean']
            stats[f'{prefix}_trough_max_height'] = stats[f'{prefix}_min']
            stats[f'{prefix}_trough_min_height'] = stats[f'{prefix}_min']
            stats[f'{prefix}_mean_trough_prominence'] = 0
            stats[f'{prefix}_max_trough_prominence'] = 0
            stats[f'{prefix}_num_troughs'] = 0
            
        # Calculate peak-to-trough amplitudes if both exist
        if len(peaks) > 0 and len(troughs) > 0:
            stats[f'{prefix}_mean_peak_to_trough'] = (stats[f'{prefix}_peak_mean_height'] - 
                                                     stats[f'{prefix}_trough_mean_height'])
            stats[f'{prefix}_max_peak_to_trough'] = (stats[f'{prefix}_peak_max_height'] - 
                                                    stats[f'{prefix}_trough_min_height'])
        else:
            stats[f'{prefix}_mean_peak_to_trough'] = 0
            stats[f'{prefix}_max_peak_to_trough'] = 0
            
        return stats, peaks, troughs

    for cell_idx in range(n_cells):
        cell_features = {}
        spiking_trace = spiking_array[cell_idx, :]
        calcium_trace = calcium_array[cell_idx, :]
        
        # Calculate derivatives
        dt = 1/sampling_rate
        first_derivative = np.gradient(calcium_trace, dt)
        second_derivative = np.gradient(first_derivative, dt)
        
        # Extract statistics for original signal and derivatives
        calcium_stats, calcium_peaks, calcium_troughs = extract_signal_stats(calcium_trace, 'calcium')
        ddt_stats, ddt_peaks, ddt_troughs = extract_signal_stats(first_derivative, 'ddt')
        d2dt2_stats, d2dt2_peaks, d2dt2_troughs = extract_signal_stats(second_derivative, 'd2dt2')
        
        # Update cell features with all signal statistics
        cell_features.update(calcium_stats)
        cell_features.update(ddt_stats)
        cell_features.update(d2dt2_stats)

        # Set default thresholds if not provided
        if spike_threshold is None:
            spike_threshold = np.mean(spiking_trace) + 2 * np.std(spiking_trace)
        if calcium_threshold is None:
            calcium_threshold = np.mean(calcium_trace) + 2 * np.std(calcium_trace)

        # Spike detection and basic features
        spike_indices, _ = find_peaks(spiking_trace, height=spike_threshold)
        spike_times = spike_indices / sampling_rate
        num_spikes = len(spike_times)
        cell_features['num_spikes'] = num_spikes

        # ISI analysis
        if num_spikes > 1:
            isi = np.diff(spike_times)
            cell_features['mean_isi'] = np.mean(isi)
            cell_features['std_isi'] = np.std(isi)
            cell_features['cv_isi'] = cell_features['std_isi'] / cell_features['mean_isi']
            cell_features['min_isi'] = np.min(isi)
            cell_features['max_isi'] = np.max(isi)
        else:
            cell_features['mean_isi'] = -1
            cell_features['std_isi'] = -1
            cell_features['cv_isi'] = -1
            cell_features['min_isi'] = -1
            cell_features['max_isi'] = -1

        # Firing rate and spike amplitude
        cell_features['firing_rate'] = num_spikes / (n_timepoints / sampling_rate)
        cell_features['max_spike'] = np.max(spiking_trace[spike_indices]) if num_spikes > 0 else 0
        cell_features['mean_spike_amplitude'] = np.mean(spiking_trace[spike_indices]) if num_spikes > 0 else 0

        # Spike train entropy
        binary_spike_train = np.zeros(n_timepoints)
        binary_spike_train[spike_indices] = 1
        prob_spike = np.mean(binary_spike_train)
        cell_features['spike_entropy'] = entropy([prob_spike, 1-prob_spike], base=2) if 0 < prob_spike < 1 else 0

        # Burst analysis
        burst_isi_threshold = 0.1  # seconds
        if num_spikes > 1:
            isi = np.diff(spike_times)
            burst_indices = np.where(isi < burst_isi_threshold)[0]
            if len(burst_indices) > 0:
                # Find burst boundaries
                burst_starts = [spike_times[burst_indices[0]]]
                burst_ends = []
                for i in range(1, len(burst_indices)):
                    if burst_indices[i] != burst_indices[i-1] + 1:
                        burst_ends.append(spike_times[burst_indices[i-1] + 1])
                        burst_starts.append(spike_times[burst_indices[i]])
                burst_ends.append(spike_times[burst_indices[-1] + 1])
                
                # Calculate burst properties
                burst_durations = np.array(burst_ends) - np.array(burst_starts)
                interburst_intervals = np.diff(burst_starts)
                
                cell_features['num_bursts'] = len(burst_durations)
                cell_features['mean_burst_duration'] = np.mean(burst_durations)
                cell_features['std_burst_duration'] = np.std(burst_durations)
                cell_features['min_burst_duration'] = np.min(burst_durations)
                cell_features['max_burst_duration'] = np.max(burst_durations)
                
                if len(interburst_intervals) > 0:
                    cell_features['mean_interburst_period'] = np.mean(interburst_intervals)
                    cell_features['std_interburst_period'] = np.std(interburst_intervals)
                    cell_features['min_interburst_period'] = np.min(interburst_intervals)
                    cell_features['max_interburst_period'] = np.max(interburst_intervals)
                    cell_features['burst_to_interburst_ratio'] = (cell_features['mean_burst_duration'] / 
                                                                cell_features['mean_interburst_period'])
                else:
                    cell_features['mean_interburst_period'] = np.nan
                    cell_features['std_interburst_period'] = np.nan
                    cell_features['min_interburst_period'] = np.nan
                    cell_features['max_interburst_period'] = np.nan
                    cell_features['burst_to_interburst_ratio'] = np.nan
            else:
                cell_features.update({
                    'num_bursts': 0,
                    'mean_burst_duration': np.nan,
                    'std_burst_duration': np.nan,
                    'min_burst_duration': np.nan,
                    'max_burst_duration': np.nan,
                    'mean_interburst_period': np.nan,
                    'std_interburst_period': np.nan,
                    'min_interburst_period': np.nan,
                    'max_interburst_period': np.nan,
                    'burst_to_interburst_ratio': np.nan
                })
        else:
            cell_features.update({
                'num_bursts': 0,
                'mean_burst_duration': np.nan,
                'std_burst_duration': np.nan,
                'min_burst_duration': np.nan,
                'max_burst_duration': np.nan,
                'mean_interburst_period': np.nan,
                'std_interburst_period': np.nan,
                'min_interburst_period': np.nan,
                'max_interburst_period': np.nan,
                'burst_to_interburst_ratio': np.nan
            })

        # Calculate rise/fall times and widths for calcium peaks
        if len(calcium_peaks) > 0:
            rise_times = []
            fall_times = []
            widths = []
            for peak_idx in calcium_peaks:
                # Rise time
                if peak_idx > 0:
                    prev_trough_idx = np.argmin(calcium_trace[:peak_idx])
                    rise_time = (peak_idx - prev_trough_idx) / sampling_rate
                    rise_times.append(rise_time)
                
                # Fall time
                if peak_idx < n_timepoints - 1:
                    next_trough_idx = peak_idx + np.argmin(calcium_trace[peak_idx:])
                    fall_time = (next_trough_idx - peak_idx) / sampling_rate
                    fall_times.append(fall_time)
                
                # Width at half maximum
                results_half = peak_widths(calcium_trace, [peak_idx], rel_height=0.5)
                widths.append(results_half[0][0] / sampling_rate)

            cell_features.update({
                'mean_rise_time': np.mean(rise_times),
                'std_rise_time': np.std(rise_times),
                'min_rise_time': np.min(rise_times),
                'max_rise_time': np.max(rise_times),
                'mean_fall_time': np.mean(fall_times),
                'std_fall_time': np.std(fall_times),
                'min_fall_time': np.min(fall_times),
                'max_fall_time': np.max(fall_times),
                'mean_width_half_max': np.mean(widths),
                'std_width_half_max': np.std(widths),
                'min_width_half_max': np.min(widths),
                'max_width_half_max': np.max(widths)
            })
        else:
            cell_features.update({
                'mean_rise_time': -1,
                'std_rise_time': -1,
                'min_rise_time': -1,
                'max_rise_time': -1,
                'mean_fall_time': -1,
                'std_fall_time': -1,
                'min_fall_time': -1,
                'max_fall_time': -1,
                'mean_width_half_max': -1,
                'std_width_half_max': -1,
                'min_width_half_max': -1,
                'max_width_half_max': -1
            })

        # Population synchrony measures
        if n_cells > 1:
            cell_corr = corr_matrix[cell_idx, :]
            cell_corr = cell_corr[cell_corr != 1]  # Remove self-correlation
            cell_features.update({
                'mean_correlation': np.mean(cell_corr),
                'max_correlation': np.max(cell_corr),
                'min_correlation': np.min(cell_corr),
                'std_correlation': np.std(cell_corr)
            })
        else:
            cell_features.update({
                'mean_correlation': 0,
                'max_correlation': 0,
                'min_correlation': 0,
                'std_correlation': 0
            })

        # Calculate area under the curve for calcium signal
        cell_features['calcium_auc'] = np.trapz(calcium_trace, dx=1/sampling_rate)
        
        # Calculate entropy of calcium signal using histogram
        hist, _ = np.histogram(calcium_trace, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins for entropy calculation
        cell_features['calcium_entropy'] = entropy(hist, base=2) if len(hist) > 0 else 0

        features_list.append(cell_features)

        # Visualization if verbose is True
        if verbose:
            offset = cell_idx * np.max(calcium_array) * 0.5
            
            # Plot calcium trace and peaks
            ax_calcium.plot(time_vector, calcium_trace + offset, label=f'Cell {cell_idx}')
            if len(calcium_peaks) > 0:
                ax_calcium.plot(time_vector[calcium_peaks], 
                              calcium_trace[calcium_peaks] + offset, 
                              'r.', markersize=5)
            
            # Plot spiking trace and detected spikes
            ax_spiking.plot(time_vector, spiking_trace + offset, label=f'Cell {cell_idx}')
            if num_spikes > 0:
                ax_spiking.plot(time_vector[spike_indices], 
                              spiking_trace[spike_indices] + offset, 
                              'r.', markersize=5)

    # Create DataFrame from features
    features_df = pd.DataFrame(features_list)
    features_df.index.name = 'cell_index'

    # Handle visualization formatting if verbose
    if verbose:
        for ax, title in [(ax_calcium, 'Calcium Traces with Detected Peaks'),
                         (ax_spiking, 'Spiking Traces with Detected Spikes')]:
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude + Offset')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.get_figure().tight_layout()
            plt.show()

    return features_df


def manual_features_from_zarr_dict(root_dict, sampling_rate=25, nuclei_filtering=False):
    """
    Calculate manual features for all zarr directories in the root_dict.
    
    Parameters
    ----------
    root_dict : dict
        Dictionary where the keys are the base names of .zarr directories 
        and the values are the opened zarr objects.
    sampling_rate : float, optional
        Sampling rate of the recordings in Hz
    nuclei_filtering : bool, optional
        If True, only calculate features for cells that contain nuclei. This will through and error if called
        on a zarr directory that does not contain a 'contains_nuclei' key.
    
    Returns
    -------
    features_df: pd.DataFrame
        A dictionary where the keys are the base names of .zarr directories 
        and the values are the extracted features DataFrames.
    """
    manual_features = []
    for key in tqdm(root_dict.keys(), total=len(root_dict.keys())):
        for well_group in root_dict[key].keys():
            for fov_group in root_dict[key][well_group].keys():
                if nuclei_filtering:
                    assert 'contains_nuclei' in root_dict[key][well_group][fov_group].keys(), "No 'contains_nuclei' key found."
                    cn_arr = np.array(root_dict[key][well_group][fov_group]['contains_nuclei'])
                    cn_arr = cn_arr.astype(bool)
                signal = np.array(root_dict[key][well_group][fov_group]['signal'])
                spike_train = np.array(root_dict[key][well_group][fov_group]['inferred_spiking'])
                if len(spike_train.shape) == 2 and spike_train.shape[0] >= 1:
                    if nuclei_filtering:
                        feat_df = extract_neuronal_features(spike_train[cn_arr],
                                                            signal[cn_arr],
                                                            sampling_rate,
                                                            spike_threshold=0.01,
                                                            calcium_threshold=0.02,
                                                            verbose=False)
                    else:
                        feat_df = extract_neuronal_features(spike_train,
                                                            signal,
                                                            sampling_rate,
                                                            spike_threshold=0.01,
                                                            calcium_threshold=0.02,
                                                            verbose=False)
                    feat_df['well_id'] = [well_group] * len(feat_df)
                    feat_df['fov_id'] = [fov_group] * len(feat_df)
                    feat_df['zarr_file'] = [key] * len(feat_df)
                    if nuclei_filtering:
                        feat_df['fov_cell_idx'] = np.arange(spike_train.shape[0])[cn_arr]
                    else:
                        feat_df['fov_cell_idx'] = np.arange(spike_train.shape[0])
                    manual_features.append(feat_df)
    features_df = pd.concat(manual_features)
    return features_df
