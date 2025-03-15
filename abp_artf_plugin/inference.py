import gc
from statistics import mode
from typing import List
from pathlib import Path
import os

import numpy as np
import torch

from abp_artf_plugin.model.model import VAE, load_ckpt


class Normalization:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def normalize(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) in [1, 2], f"Data shape {data.shape} is not supported"
        if len(data.shape) == 1:
            self.mean = np.mean(data)
            self.std = np.std(data)
            normalized_data = (data - self.mean) / (self.std + 1e-10)
        elif len(data.shape) == 2:
            self.mean = np.mean(data, axis=1, keepdims=True)
            self.std = np.std(data, axis=1, keepdims=True)
            normalized_data = (data - self.mean) / (self.std + 1e-10)
        return normalized_data

    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        return normalized_data * (self.std + 1e-10) + self.mean

class Inference:
    model_dict={ # (ckpt_name, threshold)
        'ABP': ('input1200_dim20.ckpt', 0.2968),
        'PPG': ('input1200_dim20_ppg.ckpt', 0.617428),
    }
    
    def __init__(self, model_name='ABP', **kwargs) -> None:
        """
            kwargs:
                ckpt_path
                need_preprocess
                threshold
                ml_model
        """
        
        torch.manual_seed(224)
        self.curr_path = Path(os.path.dirname(__file__))
        self.ckpt_name = model_name
        self.batch_size = 64
        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

        ckpt_path = kwargs.get('ckpt_path', self.curr_path /'model' / self.model_dict[model_name][0])
        self.model = load_ckpt(VAE().to(self.device), ckpt_path, self.device)
        self.model.eval()
        self.ml_model = kwargs.get('ml_model', None)
        
        self.normalizer = Normalization()
        self.need_preprocess = kwargs.get('need_preprocess', True)
        self.threshold = kwargs.get('threshold', self.model_dict[model_name][1])
        
    def preprocess(self, segment: np.ndarray) -> np.ndarray:
        # interp = interpolate.interp1d(np.arange(0, len(data), 1), data,
        #                             kind="cubic")
        # new_t = np.linspace(0, len(data)-1, self.resampling)
        # data = interp(new_t)
        if self.need_preprocess:
            segment = self.normalizer.normalize(segment)
        return segment
    
    def postprocess(self, segment: np.ndarray) -> np.ndarray:
        if self.need_preprocess:
            segment = self.normalizer.denormalize(segment)
        return segment
    
    def fit(self, segment_inp: np.ndarray) -> List:
        """
        Process the input data, run inference in batches, denormalize the output,
        and record the MSE between original and reconstructed data.
        """
        if self.ml_model is not None:
            return self._fit_machine_learning(segment_inp)
        else: # deep learning model
            return self._fit_deep_learning(segment_inp)
        
    def _fit_machine_learning(self, segment_inp: np.ndarray) -> List:
        from abp_artf_plugin.model.feature import convert_data2d_to_features, xgb, joblib
        self.model_surrogate = None
        self.model_prob_func = None
        
        if self.ml_model == 'xgboost_feature':
            self.model_surrogate = xgb.XGBClassifier()
            self.model_surrogate.load_model(self.curr_path / 'model' / 'xgb_1104.json')
            self.model_prob_func = self.model_surrogate.predict_proba
        elif self.ml_model == 'ocsvm_feature':
            self.model_surrogate = joblib.load(self.curr_path / 'model' / 'ocsvm_1104.pkl')
            self.model_prob_func = self.model_surrogate.decision_function
        else:
            raise ValueError(f"Invalid ml_model: {self.ml_model}")
        
        segment_features = convert_data2d_to_features(segment_inp)
        item_labels = self.model_surrogate.predict(segment_features)
        prob = self.model_prob_func(segment_features)
        return segment_features, item_labels, prob
            
    
    def _fit_deep_learning(self, segment_inp: np.ndarray) -> List:
        recons = []
        mse_list = []
        
        segment_inp = self.preprocess(segment_inp)
        
        num_samples = segment_inp.shape[0]
        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_data = segment_inp[start_idx:end_idx]
            batch_data = self.preprocess(batch_data)
            tensors = torch.tensor(batch_data, dtype=torch.float32).to(self.device).unsqueeze(1)
            with torch.no_grad():
                outputs = self.model(tensors)[0].cpu().numpy() # idx 1,2 are z_mean, z_log_var
            denormalized_outputs = self.postprocess(outputs.squeeze())
                
            batch_mse = np.mean((batch_data - denormalized_outputs) ** 2, axis=1)
            mse_list.extend(list(batch_mse))
            recons.extend(list(denormalized_outputs))

            # Clean up
            del tensors
            torch.cuda.empty_cache()
            gc.collect()

        item_labels = self.convert_to_item_label(mse_list, threshold=self.threshold)
        return recons, item_labels, mse_list

    def convert_to_item_label(self, mse_list: List, threshold=0.2968) -> List:
        """
        Convert the MSE values to item labels.
        """
        output_list = []
        for mse in mse_list:
            if mse > threshold:
                output_list.append([1] * 1200)
            else:
                output_list.append([0] * 1200)
        return output_list