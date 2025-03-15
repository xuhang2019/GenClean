from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy
from pywt import wavedec
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import joblib
import xgboost as xgb

def extract_data1d_features(sample, sample_rate=120):
    """
    Extracts a variety of time-domain, frequency-domain, morphological, and nonlinear features from an ABP signal.
    Arguments:
    - sample: 1D numpy array, the ABP signal (e.g., 1200-point, 10-second sample at 120 Hz)
    - sample_rate: int, sampling rate of the signal in Hz

    Returns:
    - features: dict, a dictionary containing extracted features
    """
    # Initialize feature dictionary
    features = {}

    # Time-domain features
    features['max'] = np.max(sample)
    features['min'] = np.min(sample)
    features['mean'] = np.mean(sample)
    features['std'] = np.std(sample)
    features['peak_to_peak'] = features['max'] - features['min']

    # Find peaks for heart rate calculation and waveform features
    systolic_peaks, _ = find_peaks(sample, distance=sample_rate / 2)
    heart_rate = len(systolic_peaks) / (len(sample) / sample_rate) * 60 if len(systolic_peaks) > 1 else 0
    features['heart_rate'] = heart_rate # bpm

    # Morphological features
    features['rise_time'] = np.mean(np.diff(systolic_peaks) / sample_rate) if len(systolic_peaks) > 1 else 0

    # Frequency-domain features
    n = len(sample)
    yf = fft(sample)
    xf = fftfreq(n, 1 / sample_rate)[:n // 2]
    power_spectrum = np.abs(yf[:n // 2]) ** 2
    total_power = np.sum(power_spectrum)

    lf_band = (xf >= 0.1) & (xf < 0.4)
    resp_band = (xf >= 0.4) & (xf < 1.0)
    hr_band = (xf >= 1.0) & (xf < 2.0)
    features['lf_power'] = np.sum(power_spectrum[lf_band]) / total_power
    features['resp_power'] = np.sum(power_spectrum[resp_band]) / total_power
    features['hr_power'] = np.sum(power_spectrum[hr_band]) / total_power
    features['spectral_entropy'] = entropy(power_spectrum / total_power)

    # Nonlinear features
    def approx_entropy(U, m, r):
        """Approximate entropy calculation."""
        N = len(U)
        def _phi(m):
            x = np.array([U[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.abs(x[:, None] - x[None, :]) <= r, axis=0) / (N - m + 1.0)
            return np.sum(np.log(C)) / (N - m + 1.0)
        return abs(_phi(m) - _phi(m + 1))
    
    features['approx_entropy'] = approx_entropy(sample, m=2, r=0.2 * np.std(sample))

    # Wavelet decomposition for multi-scale features
    coeffs = wavedec(sample, 'db4', level=3)
    features['wavelet_approx'] = np.mean(coeffs[0])
    features['wavelet_detail'] = np.mean(coeffs[1])

    return features

def extract_data1d_features_list(sample, sample_rate=120):
    return list(extract_data1d_features(sample, sample_rate).values())

def convert_data2d_to_features(data2d, sample_rate=120, num_workers=32):
    """ 
    Converts a data2d (N, L) to a feature matrix (N, F) using the extract_data_1d_features function.

    Returns:
    - features: 2D numpy array, shape (N, F), where N is the number of samples and F is the number of features
    """
    # Extract features from each sample
    if num_workers <= 1:
        features = []
        for i in tqdm(range(data2d.shape[0])):
            sample = data2d[i]
            sample_features = extract_data1d_features(sample, sample_rate)
            features.append(list(sample_features.values()))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Must disable lambda function !!!
            features = list(tqdm(executor.map(extract_data1d_features_list, data2d), total=len(data2d)))
    return np.array(features)
