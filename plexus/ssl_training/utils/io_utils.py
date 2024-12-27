import torch


def patchify(signal: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Creates equal-sized windows from a batch of time series signals.

    Parameters
    ----------
    signal : torch.Tensor
        The input time series signal of shape [batch_size, channels, sig_len]
    n_windows : int
        The number of windows to divide the signal into.
    assert signal_length % n_windows == 0
    """
    batch_size, num_channels, sig_len = signal.shape
    assert sig_len % patch_size == 0, "Signal length must be divisible by patch_size"
    n_patches = sig_len // patch_size
    patches = signal.reshape(batch_size, num_channels, n_patches, patch_size)  # shape: [batch_size, num_channels, n_patches, patch_size]
    return patches


def unpacthify(patches: torch.Tensor) -> torch.Tensor:
    """
    Creates equal-sized windows from a batch of time series signals.

    Parameters
    ----------
    signal : torch.Tensor
        The input time series signal of shape [batch_size, channels, sig_len]
    n_windows : int
        The number of windows to divide the signal into.
    assert signal_length % n_windows == 0
    """
    batch_size, num_channels, n_patches, patch_size = patches.shape
    signal = patches.reshape(batch_size, num_channels, n_patches * patch_size)  # shape: [batch_size, num_channels, sig_len]
    return signal


def create_equal_sized_windows_batch(signal, n_windows):
    """
    Creates equal-sized windows from a batch of time series signals.

    Parameters
    ----------
    signal : torch.Tensor
        The input time series signal of shape [batch_size, channels, sig_len]
    n_windows : int
        The number of windows to divide the signal into.

    Returns
    -------
    windows : ndarray
        Array of shape (batch_size, n_windows, channels, window_size) containing the signal windows.
    """
    signal = torch.Tensor(signal)  # shape [batch_size, num_channels, signal_length]
    signal_length = signal.shape[2]
    num_channels = signal.shape[1]
    batch_size = signal.shape[0]
    assert signal_length % n_windows == 0, "Signal length must be divisible by n_windows"
    window_size = signal_length // n_windows

    # Reshape the signal into n_windows equal-sized windows
    windows = signal.reshape(batch_size, num_channels, n_windows, window_size)
    windows = windows.permute(0, 2, 1, 3)  # shape: [batch_size, n_windows, num_channels, window_size]
    return windows


def compute_power_spectral_density_batch(signal, sampling_rate):
    """
    Computes the Power Spectral Density (PSD) from an FFT of a time series signal.

    Parameters
    ----------
    signal : torch.Tensor
        The input time series signal (a window of data).
    sampling_rate : float
        The sampling rate of the signal in Hz.

    Returns
    -------
    freqs : ndarray
        Array of frequency bins.
    psd : ndarray
        Power Spectral Density corresponding to each frequency bin.
    """
    # input shape: [num_tokens, 12, window_len]
    # Compute the FFT of the signal
    fft_output = torch.fft.fft(signal, dim=-1)

    # Calculate the power spectrum (magnitude of the FFT squared)
    power_spectrum = torch.abs(fft_output)

    # Normalize the power spectrum and convert it to PSD
    # Divide by the length of the signal and the sampling rate
    n = signal.shape[-1]
    psd = power_spectrum / (n * sampling_rate)

    # Generate frequency bins
    freqs = torch.fft.fftfreq(n, d=1/sampling_rate)

    # Consider only the positive part of the spectrum, as it's symmetrical
    half_n = n // 2
    return freqs[:half_n], psd[:, :, :, :half_n]