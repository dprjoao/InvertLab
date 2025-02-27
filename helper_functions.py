import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth lowpass filter to the given data.

    Parameters:
    data (array-like): The input signal data to be filtered.
    cutoff (float): The cutoff frequency of the filter.
    fs (float): The sampling frequency of the input data.
    order (int, optional): The order of the filter. Default is 5.

    Returns:
    array-like: The filtered signal data.
    """
    nyquist_freq = 0.5 * fs
    normalized_cutoff = cutoff / nyquist_freq
    b, a = butter(order, normalized_cutoff, btype="low", analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data