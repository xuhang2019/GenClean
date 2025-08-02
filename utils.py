import os
import logging
from datetime import datetime
import torch
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ===============================================
# config related utils
# ===============================================

FIGS_DIR = 'figs'
CKPT_DIR = 'ckpt'
LOGS_DIR = 'logs'

DATA_DIR_OLD = '/home/aiden/datasets/time-series/genclean_datapool/train'
DATA_DIR = 'data'
DATA_TRAIN = '250731_clean_train.h5'
# DATA_TRAIN = '240124_abp_150pts_10s_train.h5'
# DATA_VAL = '240124_abp_150pts_10s_val.h5'

@dataclass
class Config:
    # basic params
    run_name: Optional[str] = None
    model_name: str = 'VAE'
    
    # model params
    model_str: str = 'raw'
    input_shape: int = 1200
    latent_dim: int = 20
    use_revin: bool = False
    
    # training params
    batch_size: int = 32
    lr: float = 1e-3
    num_epoch: int = 200
    patience: int = 20
    device: str = None
    data_path: str = None
    max_samples: int = None
    use_filter: bool = False
    
    # data params
    data_norm: str = 'instance'
    data_dir: str = DATA_DIR
    data_train: str = DATA_TRAIN
    
    # log params
    wandb_project: str = 'tbme'
    
    # debug params
    debug: bool = False
    dryrun: bool = False
    verbose: bool = False

    def __post_init__(self):
        assert self.run_name is not None, 'run_name is required'
        timestamp = datetime.now().strftime('%m%d%H%M')
        self.save_name = f"{timestamp}_{self.model_str}_{self.run_name}"    
        
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ckpt_dir = os.path.join(CKPT_DIR, self.save_name)
        self.train_data_path = os.path.join(self.data_dir, self.data_train)
        
        

# ===============================================
# logger related utils
# ===============================================

class Logger:
    def __init__(self, config: Config):
        self.debug = config.debug
        self.dryrun = config.dryrun
        self.verbose = config.verbose
        self.use_wandb = not (self.debug or self.dryrun)
        self.wandb = None
        
        if self.use_wandb:
            import wandb
            self.wandb = wandb.init(project=config.wandb_project, name=config.run_name, config=config.__dict__)
        logging.basicConfig(level=logging.DEBUG if self.debug else logging.INFO)
    
    def log(self, msg=None, wandb_msg=None, **kwargs):
        if msg and self.debug:
            print(msg)
        if self.wandb and wandb_msg:
            assert isinstance(wandb_msg, dict), 'wandb_msg must be a dict'
            self.wandb.log(wandb_msg)
    
    def close(self):
        if self.wandb:
            self.wandb.finish()




# ===============================================
# tools related utils
# ===============================================


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================================
# validation utils
# ===============================================


def validate_with_nan(X, X_recon, y, thres=0.08):
    """
    Args:
        X: np.ndarray, shape (N, L)
        X_recon: np.ndarray, shape (N, L)
        y: np.ndarray, shape (N,)
        thres: float

    Returns:
        acc: float, accuracy of (mse(X, X_recon) > thres or nan in recon) == y
    """
    # Identify samples where any value in X_recon is nan
    nan_mask = np.isnan(X_recon).any(axis=1)
    # Compute MSE for non-nan samples
    mse = np.mean((X - X_recon) ** 2, axis=1)
    # For samples with nan in recon, always predict 1 (anomaly)
    y_pred = np.where(nan_mask, 1, (mse > thres).astype(int))
    acc = np.mean(y_pred == y)
    return acc


def validate(X, X_recon, y, thres=0.2968):
    """
    Args:
        X: np.ndarray, shape (N, L)
        X_recon: np.ndarray, shape (N, L)
        y: np.ndarray, shape (N,)
        thres: float

    Returns:
        acc: float, accuracy of (mse(X, X_recon) > thres) == y
    """
    # Compute MSE for each sample
    mse = np.mean((X - X_recon) ** 2, axis=1)
    y_pred = (mse > thres).astype(int)
    acc = np.mean(y_pred == y)
    return acc



# ===============================================
# visualise utils
# ===============================================

# A4 paper
# Double column: 5.5 * 3
# Single column: 2.8 * 2
# Font size: 7 - 9
# bbox_inches = 'tight'

def init_plt():
    plt.rcParams.update({
        'font.size': 8,          # default font size
        'axes.labelsize': 8,     # fontsize of the x and y labels
        'axes.titlesize': 10,    # fontsize of the axes title
        'xtick.labelsize': 7,    # fontsize of the x tick labels
        'ytick.labelsize': 7,    # fontsize of the y tick labels
        'legend.fontsize': 7,    # legend fontsize
        'figure.titlesize': 10   # fontsize of the figure title
    })


def plot_raw_vs_recon(raw, recon, legends=None):
    mse = np.mean((raw - recon) ** 2)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
    ax.plot(raw, label=legends[0] if legends and len(legends) > 0 else "Raw")
    ax.plot(recon, label=legends[1] if legends and len(legends) > 1 else "Recon")
    ax.set_title(f"Raw vs Recon (MSE={mse:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig


def fast_ax_plot(x, label='x'):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
    ax.plot(x, label=label)
    
    return fig, ax