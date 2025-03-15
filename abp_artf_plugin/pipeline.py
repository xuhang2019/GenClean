import numpy as np
from scipy import signal
from abp_artf_plugin.inference import Inference


class ProcessingPipeline:
    """
        Given an ABP signal and its original frequency, process to the reconstucted signal and item labels.
    """        
    def fill_na(self, sig1:np.ndarray):
        nan_indices = np.isnan(sig1)
        if np.any(nan_indices):
            print("Detect NaN values in the input signal. Replace them with the nearest non-NaN values.")
            sig1[nan_indices] = np.interp(
                np.flatnonzero(nan_indices), 
                np.flatnonzero(~nan_indices), 
                sig1[~nan_indices]
            )
        return sig1
        
    def resample(self, sig1:np.ndarray, ori_freq, target_freq=120):
        resample_factor = target_freq / ori_freq
        resample_1d = lambda sig1: signal.resample(sig1, int(len(sig1) * resample_factor))
    
        if sig1.ndim == 1:
            resampled_sig = resample_1d(sig1)
        elif sig1.ndim == 2:
            resampled_sig = np.array([resample_1d(channel) for channel in sig1])
        else:
            raise ValueError("Unsupported signal dimension. Only 1D and 2D arrays are supported.")
        
        return resampled_sig
        
            
    def chunkise(self, sig1:np.ndarray, chunk_size=1200):
        assert sig1.ndim == 1, "The input signal should be 1D."
        num_chunks = len(sig1) // chunk_size
        if len(sig1) < chunk_size:
            raise ValueError("The input signal is too short.")
        sig1 = sig1[:num_chunks * chunk_size]  # Drop remainder
        sig1 = sig1.reshape(-1, chunk_size)
        return sig1
        
    def process(self, sig1:np.ndarray, ori_freq, model_name='ABP', **kwargs):
        """
            Allow `sig1` to be 1D or 2D array.
            kwargs: will be passed into `Inference` as attributes.
        """
        is_1d = sig1.ndim == 1
        sig1 = self.fill_na(sig1)
        
        if int(ori_freq) == 120:
            if is_1d:
                sig1 = self.chunkise(sig1)
            return self._process_2d_120Hz(sig1, model_name, **kwargs)
        else:
            sig1 = self.resample(sig1, ori_freq=ori_freq, target_freq=120)
            sig1 = self.chunkise(sig1) if is_1d else sig1
            processed_sig, item_labels, mse_list = self._process_2d_120Hz(sig1, model_name, **kwargs)
            processed_sig = self.resample(processed_sig.flatten() if is_1d else processed_sig, ori_freq=120, target_freq=ori_freq)
            item_labels = self.resample(item_labels.flatten() if is_1d else item_labels, ori_freq=120, target_freq=ori_freq)
            return processed_sig, item_labels, mse_list
        

    def _process_2d_120Hz(self, sig1:np.ndarray, model_name='ABP', **kwargs):
        """
            Helper function.
            Process the input signal with 120Hz frequency.
        """
        processed_sig, item_labels, mse_list = Inference(model_name, **kwargs).fit(sig1) 
        processed_sig, item_labels = np.array(processed_sig), np.array(item_labels)
        return processed_sig, item_labels, mse_list