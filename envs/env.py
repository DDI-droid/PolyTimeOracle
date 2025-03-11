from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

def kendall_tau_distance_from_vectors(p1: torch.Tensor, p2: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kendall Tau distance (number of discordant pairs) between two permutation vectors,
    while excluding elements marked in a one-hot encoded exclude mask.

    Args:
        p1 (torch.Tensor): Permutation vector of shape (B, V, N). Each row is a permutation of element IDs.
        p2 (torch.Tensor): Permutation vector of shape (B, N).
        exclude (torch.Tensor): One-hot encoded exclusion mask of shape (B, N) where 1 indicates the element is excluded.

    Returns:
        torch.Tensor: Tensor of shape (B,) containing the Kendall Tau distance for each batch element.
    """
    B, N = p1.shape

    valid = (exclude == 0)  # shape (B, N)

    rank1 = torch.empty_like(p1)
    rank2 = torch.empty_like(p2)

    pos = torch.arange(N, device=p1.device).unsqueeze(0).expand(B, -1)
    rank1.scatter_(1, p1, pos)
    rank2.scatter_(1, p2, pos)

    idx = torch.arange(N, device=p1.device)
    e, f = torch.meshgrid(idx, idx, indexing='ij')

    pair_mask = (e < f)  # shape (N, N)

    pair_mask = pair_mask.unsqueeze(0).expand(B, -1, -1)

    valid_e = valid.unsqueeze(2).expand(B, N, N)
    valid_f = valid.unsqueeze(1).expand(B, N, N)
    valid_pairs = valid_e & valid_f & pair_mask

    rank1_e = rank1.unsqueeze(2).expand(B, N, N)  # rank of element e in p1
    rank1_f = rank1.unsqueeze(1).expand(B, N, N)  # rank of element f in p1
    rank2_e = rank2.unsqueeze(2).expand(B, N, N)  # rank of element e in p2
    rank2_f = rank2.unsqueeze(1).expand(B, N, N)  # rank of element f in p2

    discordant = ((rank1_e < rank1_f) & (rank2_e > rank2_f)) | ((rank1_e > rank1_f) & (rank2_e < rank2_f))
    discordant = discordant & valid_pairs

    kendall_tau_dist = discordant.sum(dim=(1, 2))
    
    return kendall_tau_dist


class PolyTimeOracle(EnvBase):
    """
    A MDP environment from which a policy learns a polynomial time Oracle/a polynomial time algorithm
    """
    def __init__(self, tape_size: int = 2048, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.state = None
        self.problem = None
        
        self.tape_size = tape_size
        
        self.epsilon = 1e-3
                
        self.action_spec = Unbounded(
            shape=(self.batch_size[0], self.tape_size, self.tape_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.tape_spec = Unbounded(
            shape=(self.batch_size[0], self.tape_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.observation_spec = Composite(
            tape=self.tape_spec,
            shape=(self.batch_size[0], self.tape_size)
        )
        
        self.reward_spec = Bounded(
            shape=(self.batch_size[0],),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
            device=self.device
        )
                
        self.halted_envs = None
        
        self.halted_kd = None
        
        self.running_time = None
        
        
    def _reset(self, problem: torch.Tensor) -> TensorDict:
        
        self.halted_envs = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
        
        self.halted_kd = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
        
        self.running_time = torch.zeros(self.batch_size).to(self.device)
        
        self.state = torch.zeros(self.batch_size[0], self.tape_size).to(self.device)
        self.problem = problem.to(self.device)
        
        return TensorDict(
            tape=self.state
        )

    def _step(self, action: TensorDict) -> TensorDict:
        self.state = torch.where(action["w"] @ self.state + action["b"], self.state)
        
        halt_flag = self.state[..., -1]
        
        self.halted_envs = self.halted_envs | (halt_flag > self.epsilon).to(self.device)

        
        candidate_flags = self.state[..., -self.problem["n_candidates"]-1:-1]
        
        kd = kendall_tau_distance_from_vectors(self.problem["V"], self.problem["X"], c_flags)

        c_flags = torch.where(candidate_flags > self.epsilon, torch.ones_like(candidate_flags), torch.zeros_like(candidate_flags))

        cst = c_flags @ self.problem["costs"]
                
        self.halted_kd = self.halted_kd | (self.halted_envs & (kd > self.problem["k"])).to(self.device)
        
        r = torch.where(self.halted_kd, -torch.ones(self.batch_size), torch.zeros(self.batch_size)).to(self.device)
        
        r = torch.where(self.halted_envs & ~self.halted_kd, (2/(1 + torch.exp(-cst)) - 1).to(self.device), r)
        
        r = torch.where(~self.halted_envs, (2/(1 + torch.exp(self.r_alpha * (torch.where(kd > self.problem["k"]), kd, torch.zeros_like(kd)) + self.r_beta * cst + self.r_gamma * torch.ones(self.batch_size))) - 1).to(self.device), r)
        
        return TensorDict(
            observation=self.state,
            reward=r,
            done=self.halted_envs,
            device=self.device,
            batch_size=self.batch_size
        )
        
        
    def _close(self) -> None:
        pass
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        pass
    
    def _render(self, mode: str = "manim") -> None:
        pass
    
    
if __name__ == "__main__":
    pt = PolyTimeOracle(batch_size=(1, ))
    pt.reset()