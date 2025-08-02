import h5py
import numpy as np  
import torch


def normalize_numpy(numpy_array, data_norm='instance'):
    """
        normalize the numpy array (N, L) or (L,)
        
        Convert to (N, 1, L) tensor and then normalize and convert back to numpy array
    """
    assert numpy_array.ndim <= 2, 'numpy_array must be 1D or 2D'
    
    ndim = numpy_array.ndim
    
    if ndim == 1:
        numpy_array = numpy_array.reshape(1, -1)
        
    tensor = torch.tensor(numpy_array)
    tensor = tensor.unsqueeze(1)
    tensor = normalize_tensor(tensor, data_norm)
    
    numpy_array = tensor.squeeze(1).numpy()
    if ndim == 1:
        numpy_array = numpy_array.flatten()
        
    return numpy_array
    

def normalize_tensor(tensor, data_norm='instance'):
    """
        normalize the tensor (N, 1, L)
        
        data_norm:
            - instance: instance normalization
            - minmax: min-max normalization
            - none: no normalization
    """
    if data_norm == 'instance':
        tmean = tensor.mean(dim=2, keepdim=True)
        tstd = tensor.std(dim=2, keepdim=True)
        tensor = (tensor - tmean) / tstd
    elif data_norm == 'minmax':
        min_val = torch.quantile(tensor, 0.025, dim=2, keepdim=True)
        max_val = torch.quantile(tensor, 0.975, dim=2, keepdim=True)
        tensor = (tensor - min_val) / (max_val - min_val)
        tensor = tensor * 2 - 1
    elif data_norm == 'none' or data_norm is None:
        pass
    else:
        raise ValueError(f"Invalid data normalization method: {data_norm}")
    
    return tensor


def load_data(data_path, subset='data',data_norm='instance'):
    with h5py.File(data_path, 'r') as hf:
        train_npy = hf[subset][:]
    train_tensor = torch.tensor(train_npy)
    train_tensor = normalize_tensor(train_tensor, data_norm)
    return train_tensor
    
    
def check_any_nan(tensor):
    return torch.isnan(tensor).any()

def check_low_std(tensor, std_dim=1, threshold=0.05):
    """
    check if any std is less than threshold
    return a list of indices
    """
    return torch.where(tensor.std(dim=std_dim) < threshold)[0]

def check_high_max(tensor, max_dim=1, threshold=300):
    """
    check if any max is greater than threshold
    return a list of indices
    """
    return torch.where(tensor.max(dim=max_dim).values > threshold)[0]

def check_low_min(tensor, min_dim=1, threshold=0):
    """
    check if any min is less than threshold
    return a list of indices
    """
    return torch.where(tensor.min(dim=min_dim).values < threshold)[0]

def remove_index(tensor, index_tensor):
    """
        tensor: N, L
        remove the index in the tensor (dim = 0)
    """
    return tensor[~torch.isin(torch.arange(tensor.shape[0]), index_tensor)]