from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict, TensorDictBase, tensorclass
from tensordict.nn import TensorDictModule
from torch import nn
import torch.nn.functional as F

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

from utils import GREEN, RESET

import random

MIN_CANDIDATES = 5
MAX_CANDIDATES = 100
MIN_VOTERS = 7
MAX_VOTERS = 1_000
MIN_COSTS = 10
MAX_COSTS = 500

SEQS_PADDING = 0
COSTS_PADDING = 0
K_DIVIDER = 3.7
V_PADDING = 0



@tensorclass
class problem:
    X: torch.Tensor # (B, MAX_CANDIDATES)
    V: torch.Tensor # (B, MAX_VOTERS, MAX_CANDIDATES)
    costs: torch.Tensor # (B, MAX_CANDIDATES)
    k: torch.Tensor # (B, 1)
    n_candidates: torch.Tensor # (B, 1)
    n_voters: torch.Tensor # (B, 1)
    embed: torch.Tensor = None # (B, embed_dim)


def kendall_tau_distance_from_vectors(p1: torch.Tensor, p2: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
    B, V, N = p1.shape
    onehot_p1 = F.one_hot(p1.long(), num_classes=N+1)[..., 1:]
    onehot_p2 = F.one_hot(p2.long(), num_classes=N+1)[..., 1:]
    
    rank1 = onehot_p1.float().argmax(dim=2)
    rank2 = onehot_p2.float().argmax(dim=1)
    
    candidate_valid = (onehot_p2.sum(dim=1) > 0)
    candidate_excluded = (onehot_p2 * exclude.unsqueeze(-1)).sum(dim=1) > 0
    valid_candidate = candidate_valid & ~candidate_excluded
    
    idx = torch.arange(N, device=p1.device)
    e, f = torch.meshgrid(idx, idx, indexing='ij')
    pair_mask = (e < f).unsqueeze(0).expand(B, -1, -1)
    valid_pairs = (valid_candidate.unsqueeze(1) & valid_candidate.unsqueeze(2)) & pair_mask
    
    valid_ranking = (p1 != 0).any(dim=2).float()  # (B, V)
    
    r1_e = rank1.unsqueeze(3).expand(B, V, N, N)
    r1_f = rank1.unsqueeze(2).expand(B, V, N, N)
    r2_e = rank2.unsqueeze(2).expand(B, N, N)
    r2_f = rank2.unsqueeze(1).expand(B, N, N)
    
    discordant = ((r1_e < r1_f) & (r2_e > r2_f)) | ((r1_e > r1_f) & (r2_e < r2_f))
    valid_ranking = valid_ranking.unsqueeze(2).unsqueeze(3)
    discordant = discordant * valid_ranking
    discordant_sum = discordant.sum(dim=1)
    kendall_tau_dist = (discordant_sum * valid_pairs).sum(dim=(1, 2))
    
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
                
        self.action_spec = Composite(
            w=Unbounded(
                shape=(self.batch_size[0], self.tape_size, self.tape_size),
                dtype=torch.float32,
                device=self.device
            ),
            b=Unbounded(
                shape=(self.batch_size[0], self.tape_size),
                dtype=torch.float32,
                device=self.device
            ),
            shape=(self.batch_size[0], )
        )
        
        # self.tape_spec = Unbounded(
        #     shape=(self.batch_size[0], self.tape_size),
        #     dtype=torch.float32,
        #     device=self.device
        # )
        
        self.observation_spec = Unbounded(
            shape=(self.batch_size[0], self.tape_size),
            dtype=torch.float32,
            device=self.device
        )
        
        self.reward_spec = Bounded(
            shape=(self.batch_size[0], 1),
            dtype=torch.float32,
            low=-1.0,
            high=1.0,
            device=self.device
        )
        
        self.done_spec = Bounded(
            low=torch.tensor(False),
            high=torch.tensor(True),
            shape=(self.batch_size[0], 1),
            dtype=torch.bool
        )
                
        self.halted_envs = None
        
        self.halted_kd = None
        
        self.running_time = None
        
        
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        
        self.halted_envs = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
        
        self.halted_kd = torch.zeros(self.batch_size, dtype=torch.bool).to(self.device)
        
        self.running_time = torch.zeros(self.batch_size).to(self.device)
        
        self.state = torch.zeros(self.batch_size[0], self.tape_size).to(self.device)
        
        #problem generation
        self.problem = self.generate_problems()
        
        return TensorDict(
            observation=self.state,
            problem=self.problem,
            done=torch.zeros(self.batch_size).bool(),
            device=self.device,
            batch_size=self.batch_size
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.state = torch.where((~self.halted_envs).unsqueeze(-1), torch.bmm(tensordict['w'], self.state.unsqueeze(-1)).squeeze(-1) + tensordict['b'], self.state)
        
        halt_flag = self.state[..., -1]
        
        self.halted_envs = self.halted_envs | (halt_flag > self.epsilon).to(self.device)

        candidate_flags = self.state[..., -MAX_CANDIDATES-1:-1]

        c_flags = torch.where(candidate_flags > self.epsilon, torch.ones_like(candidate_flags), torch.zeros_like(candidate_flags))
        
        kd = kendall_tau_distance_from_vectors(self.problem["V"], self.problem["X"], c_flags)

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
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic
    
    def _render(self, mode: str = "manim") -> None:
        pass
        
    def generate_problems(self) -> TensorDict:
        B = self.batch_size[0]

        n_candidates = torch.randint(MIN_CANDIDATES, MAX_CANDIDATES + 1, (B,), device=self.device)
        n_voters = torch.randint(MIN_VOTERS, MAX_VOTERS + 1, (B,), device=self.device)

        rand_scores_X = torch.rand(B, MAX_CANDIDATES, device=self.device)
        candidate_ids_X = torch.arange(1, MAX_CANDIDATES + 1, device=self.device).unsqueeze(0).expand(B, -1)
        
        mask_1_X = torch.arange(MAX_CANDIDATES, device=self.device).unsqueeze(0).expand(B, -1) >= n_candidates.unsqueeze(-1)       
        rand_scores_X[mask_1_X] = np.inf
        
        sorted_idx_X = rand_scores_X.argsort(dim=1)  # (B, MAX_CANDIDATES)
        X_full = candidate_ids_X.gather(1, sorted_idx_X)  # (B, MAX_CANDIDATES)

        cand_positions_X = torch.arange(MAX_CANDIDATES, device=self.device).unsqueeze(0)  # (1, MAX_CANDIDATES)
        mask_2_X = cand_positions_X < n_candidates.unsqueeze(1)  # (B, MAX_CANDIDATES), bool

        # X = X_full * mask_2_X.long()
        X = torch.where(mask_2_X, X_full, torch.full_like(X_full, SEQS_PADDING))


        rand_scores_V = torch.rand(B, MAX_VOTERS, MAX_CANDIDATES, device=self.device)
        candidate_ids_V = torch.arange(1, MAX_CANDIDATES + 1, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, MAX_VOTERS, -1)

        mask_1_V = torch.arange(MAX_CANDIDATES, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, MAX_VOTERS, -1) >= n_candidates.unsqueeze(-1).unsqueeze(-1).expand(-1, MAX_VOTERS, -1)
        rand_scores_V[mask_1_V] = np.inf

        sorted_idx_V = rand_scores_V.argsort(dim=2)  # (B, MAX_VOTERS, MAX_CANDIDATES)
        V_full = candidate_ids_V.gather(2, sorted_idx_V)  # (B, MAX_VOTERS, MAX_CANDIDATES)

        cand_positions_V = torch.arange(MAX_CANDIDATES, device=self.device).unsqueeze(0).unsqueeze(0)  # (1,1, MAX_CANDIDATES)
        mask_V = cand_positions_V < n_candidates.unsqueeze(1).unsqueeze(2)  # (B, MAX_VOTERS, MAX_CANDIDATES)
        
        # V = V_full * mask_V.long()
        V = torch.where(mask_V, V_full, torch.full_like(V_full, SEQS_PADDING)) # (B, MAX_VOTERS, MAX_CANDIDATES)

        mask_2_V = torch.arange(0, MAX_VOTERS, device=self.device).unsqueeze(0).expand(B, -1) < n_voters.unsqueeze(-1) # (B, MAX_VOTERS)
        mask_2_V = mask_2_V.unsqueeze(-1) # (B, MAX_VOTERS, 1)
                
        V = torch.where(mask_2_V, V, torch.full_like(V, V_PADDING))

        costs_full = torch.rand(B, MAX_CANDIDATES, device=self.device)
        costs = ((MAX_COSTS - MIN_COSTS) * costs_full + MIN_COSTS).floor().long() + 1
        costs = torch.where(mask_2_X, costs, torch.full_like(costs, COSTS_PADDING))
        
        k = (torch.rand(B, device=self.device) * ((torch.square(n_candidates) - n_candidates).float()/K_DIVIDER)).floor().unsqueeze(1).long() + 1

        td = TensorDict({
            "X": X,                # (B, MAX_CANDIDATES): each row is a permutation of [1,..., n_candidates] with zeros after.
            "V": V,                # (B, MAX_VOTERS, MAX_CANDIDATES): each voter's ranking.
            "costs": costs,        # (B, MAX_CANDIDATES)
            "k": k,                # (B, 1)
            "n_candidates": n_candidates.unsqueeze(-1),  # (B, 1)
            "n_voters": n_voters.unsqueeze(-1),          # (B, 1)
        }, 
        batch_size=(B,),
        device=self.device)

        return td
        
    
if __name__ == "__main__":
    pass