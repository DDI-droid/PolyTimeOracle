import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    cuda: bool = True
    
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    α: float = 3e-4
    num_envs: int = 1
    """the number of parallel game environments"""
    
    buffer_size: int = int(1e6)
    γ: float = 0.99
    τ: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    
    
