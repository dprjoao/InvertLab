import numpy as np
from scipy.signal import butter, filtfilt
from scipy import signal


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


def generate_ricker(peak_freq, samples, dt):
    """
    retorna a wavelet de Ricker e sua FFT
    """
    # Array do tempo
    t = np.arange(samples) * (dt / 1000)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    # Cálculo da wavelet de Ricker
    pi2_f2_t2 = (np.pi**2) * (peak_freq**2) * (t**2)
    ricker = (1.0 - 2.0 * pi2_f2_t2) * np.exp(-pi2_f2_t2)

    # Cálculo da FFT
    freqs = np.fft.rfftfreq(t.shape[0], d=dt / 1000)
    fft = np.abs(np.fft.rfft(ricker))
    fft = fft / np.max(fft)

    return t, ricker, freqs, fft


def generate_butter(freq_hi, freq_low, samples, dt):
    """
    retorna a wavelet de Butterworth e sua FFT
    """
    # Calcular array de tempo
    t = np.arange(samples) * (dt / 1000)
    t = np.concatenate((np.flipud(-t[1:]), t), axis=0)

    # Criar sinal de impulso
    imp = signal.unit_impulse(t.shape[0], "mid")

    # Aplicar filtro Butterworth passa-alta
    fs = 1000 * (1 / dt)
    b, a = signal.butter(4, freq_hi, fs=fs)
    response_zp = signal.filtfilt(b, a, imp)

    # Aplicar filtro Butterworth passa-baixa
    low_b, low_a = signal.butter(2, freq_low, "hp", fs=fs)
    butter_wvlt = signal.filtfilt(low_b, low_a, response_zp)

    # Normalizar a wavelet
    butter_wvlt = butter_wvlt / np.max(butter_wvlt)

    # Calcular a FFT
    freqs = np.fft.rfftfreq(t.shape[0], d=dt / 1000)
    fft = np.abs(np.fft.rfft(butter_wvlt))
    fft = fft / np.max(fft)

    return t, butter_wvlt, freqs, fft

def estimate_wavelet(seismic_data, sampling_rate):
    nt_wav = 16
    nfft = 2**8

    fft_result = np.fft.fft(seismic_data, nfft, axis=-1)

    wav_est_fft = np.mean(np.abs(fft_result), axis=0)

    fwest = np.fft.fftfreq(nfft, d=4 / 1000)
    wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = wav_est / wav_est.max()
    fwest = fwest[: nfft // 2]
    wav_est_fft = wav_est_fft[: nfft // 2]

    dt = sampling_rate / 1000  # Assuming the sampling interval is 4 ms
    time_values = np.arange(-128 + 1, 129) * dt

    # Check the shape of wav_est
    print("Shape of wav_est:", np.shape(wav_est))

    return time_values, wav_est, fwest, wav_est_fft