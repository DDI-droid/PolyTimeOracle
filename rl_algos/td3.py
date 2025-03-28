import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tyro
import wandb
from torchrl.envs import EnvBase
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage, PrioritizedSampler

from utils import load_yaml_config


CONFIG_FILE = os.path.join(os.getcwd(), "rl_algos/config", "config.yaml")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    cuda: bool = True
    
    track: bool = True
    wandb_project_name: str = "PolyTimeOracle"
    wandb_entity: str = "deepdecisionintelligence-ddi"
    capture_video: bool = False
    save_model: bool = False

    # Algorithm specific arguments
    total_episodes: int = 10_000
    """total timesteps of the experiments"""
    α: float = 3e-4
    # num_envs: int = None
    # """the number of parallel game environments"""
    
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
    alpha_prioritized_replay: float = 0.6
    beta_prioritized_replay: float = 0.4
    epsilon_prioritized_replay: float = 1e-6
    
    max_f_halted_envs_init: float = 0.7
    """maximum initial fraction of halted environments"""
    max_f_halted_envs_final: float = 0.3
    """maximum final fraction of halted environments"""


class TD3:
    def __init__(self) -> None:
        self.args = Args(**load_yaml_config(CONFIG_FILE))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self, π: nn.Module, π_t: nn.Module, qf1: nn.Module, qf2: nn.Module, qf1_t: nn.Module, qf2_t: nn.Module, env: EnvBase):
        run_name = f"{self.args.exp_name}_{self.args.seed}_{int(time.time())}"
        
        if self.args.track:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                sync_tensorboard=False,
                config=vars(self.args),
                name=run_name,
                save_code=False,
            )
        
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
        
        π_t.load_state_dict(π.state_dict())
        qf1_t.load_state_dict(qf1.state_dict())
        qf2.load_state_dict(qf2.state_dict())
        
        q_opt = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=self.args.α)
        π_opt = optim.Adam(list(π.parameters()), lr=self.args.α)
        
        rb = TensorDictReplayBuffer(
            priority_key="priority",
            storage=LazyTensorStorage(self.args.buffer_size, device=self.device),
            sampler=PrioritizedSampler(max_capacity=self.args.buffer_size, alpha=self.args.alpha_prioritized_replay, beta=self.args.beta_prioritized_replay, eps=self.args.epsilon_prioritized_replay, max_priority_within_buffer=True)
        )
        
        start_time = time.time()
        
        t_dict = env.reset()
        env.set_seed(self.args.seed)
        
        for episode in range(self.args.total_episodes):
            max_f_halted_envs = self.args.max_f_halted_envs_init + (self.args.max_f_halted_envs_final - self.args.max_f_halted_envs_init) * (episode / self.args.total_episodes)
            
            t_dict = env.reset()
            time_step = 0    
            
            while t_dict['done'].sum().item()/env.batch_size[0] <= max_f_halted_envs:
                #action logic
                action = None
                if time_step < self.args.learning_starts:
                    #frost start
                    action = env.action_spec.sample()
                else:
                    with torch.no_grad():
                        action = π(t_dict['observation'])
                        action = action + torch.randn_like(action) * self.args.exploration_noise
                
                t_dict = env.step(action)
                
                
                
                time_step += 1                
                
         
        