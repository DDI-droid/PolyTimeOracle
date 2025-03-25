from collections import defaultdict
from typing import Optional

import numpy as np
import torch
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
from torch.linalg import vecdot

from utils import GREEN, RED, YELLOW, BLUE, RESET, load_yaml_config
from dataclasses import dataclass
import os
import random


CONFIG_FILE = os.path.join(os.getcwd(), "envs/config", "config.yaml")

@dataclass
class Args:
    env_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    
    cuda: bool = True
    
    MIN_CANDIDATES: int = 5
    MAX_CANDIDATES: int = 10
    MIN_VOTERS: int = 7
    MAX_VOTERS: int = 100
    MIN_COSTS: int = 10
    MAX_COSTS: int = 50
    
    tape_size: int = 2048

    SEQS_PADDING: int = 0
    COSTS_PADDING: int = 0
    K_DIVIDER: float = 3.7
    V_PADDING: int = 0
    
    epsilon: float = 1e-3
    
    r_alpha: float = 0.6
    r_beta: float = 0.4
    r_gamma: float = 1.0


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
    
    rank1 = onehot_p1.argmax(dim=2)
    rank2 = onehot_p2.argmax(dim=1)
     
    candidate_valid = (onehot_p2.sum(dim=1) > 0)
    candidate_excluded = (onehot_p2 * exclude.unsqueeze(-1)).sum(dim=1) > 0
    valid_candidate = candidate_valid & ~candidate_excluded
    
    valid_pairs = (valid_candidate.unsqueeze(1) & valid_candidate.unsqueeze(2))
    
    valid_ranking = (p1 != 0).any(dim=2).long()  # (B, V)
    
    t_rank_v_1 = rank1.unsqueeze(-2).expand(-1, -1, p1.shape[-1], -1)
    t_rank_v_2 = rank1.unsqueeze(-1).expand(-1, -1, -1, p1.shape[-1])
    
    t_map_v = (t_rank_v_2 < t_rank_v_1)
    
    t_rank_x_1 = rank2.unsqueeze(-2).expand(-1, p1.shape[-1], -1)
    t_rank_x_2 = rank2.unsqueeze(-1).expand(-1, -1, p1.shape[-1])

    t_map_x = (t_rank_x_2 < t_rank_x_1)
    
    t_map = torch.logical_xor(t_map_v, t_map_x.unsqueeze(1)) & valid_pairs.unsqueeze(1)
    
    kendall_tau_dist = t_map.long().sum(dim=(2, 3)) // 2
    
    kendall_tau_dist = (kendall_tau_dist * valid_ranking).sum(1)
    
    return kendall_tau_dist.long()


class PolyTimeOracle(EnvBase):
    """
    A MDP environment from which a policy learns a polynomial time Oracle/a polynomial time algorithm for Candidate Deletion Problem See Readme for details
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.state = None
        self.problem = None
                
        self.args = Args(**load_yaml_config(CONFIG_FILE))
        

        self.action_spec = Composite(
            w=Unbounded(
                shape=(self.batch_size[0], self.args.tape_size, self.args.tape_size),
                dtype=torch.float32,
                device=self.device
            ),
            b=Unbounded(
                shape=(self.batch_size[0], self.args.tape_size),
                dtype=torch.float32,
                device=self.device
            ),
            shape=(self.batch_size[0], )
        )
        
        # self.tape_spec = Unbounded(
        #     shape=(self.batch_size[0], self.args.tape_size),
        #     dtype=torch.float32,
        #     device=self.device
        # )
        
        self.observation_spec = Unbounded(
            shape=(self.batch_size[0], self.args.tape_size),
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
        
        self.halted_envs = torch.zeros(self.batch_size, dtype=torch.bool).bool().to(self.device)
        
        self.halted_kd = torch.zeros(self.batch_size, dtype=torch.bool).bool().to(self.device)
        
        self.running_time = torch.zeros(self.batch_size).to(self.device)
        
        self.state = torch.zeros(self.batch_size[0], self.args.tape_size).to(self.device)
        
        self.cst = torch.zeros(self.batch_size[0]).to(self.device)
                
        #problem generation
        self.problem = self.generate_problems()
        
        self.ktd = kendall_tau_distance_from_vectors(self.problem['V'], self.problem['X'], torch.zeros(self.batch_size[0], self.args.MAX_CANDIDATES))

        
        return TensorDict(
            observation=self.state,
            problem=self.problem,
            done=self.halted_envs,
            device=self.device,
            batch_size=self.batch_size
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # self.state = torch.where((~self.halted_envs).unsqueeze(-1), torch.bmm(tensordict['w'], self.state.unsqueeze(-1)).squeeze(-1) + tensordict['b'], self.state)
        # Optimized state computation
        prev_state = self.state
        new_state = torch.zeros_like(self.state)
        
        neg_halted_envs = ~self.halted_envs # extra lite
        w_prime = tensordict['w'][neg_halted_envs]# extra
        prev_state_prime = prev_state.unsqueeze_(-1)[neg_halted_envs]#extra
        b_prime = tensordict['b'][neg_halted_envs]
        
        new_state[neg_halted_envs] = torch.bmm(w_prime, prev_state_prime).squeeze_(-1) + b_prime
              
        prev_state.squeeze_(-1)
        
        new_state[self.halted_envs] = prev_state[self.halted_envs]
                
        self.state = new_state
        # del prev_state
        # del neg_halted_envs
        # del w_prime
        # del prev_state_prime
        # del b_prime
        # del prev_state[self.halted_envs]
        
        
        halt_flag = self.state[..., -1]
        
        #TODO optimize logical or for halted_envs
        self.halted_envs = self.halted_envs | (halt_flag > self.args.epsilon).to(self.device)
        
        # envs specific implementation [Candidate Deletion problem]       
        candidate_flags = self.state[..., -self.args.MAX_CANDIDATES-1:-1]

        c_flags = torch.where(candidate_flags > self.args.epsilon, torch.ones_like(candidate_flags), torch.zeros_like(candidate_flags))
        
        # cst = vecdot(c_flags, self.problem['costs'].float())
        # optimized cst computation
        cst = torch.zeros_like(self.cst)
        cst[neg_halted_envs] = vecdot(c_flags[neg_halted_envs], self.problem['costs'].float()[neg_halted_envs])
        cst[self.halted_envs] = self.cst[self.halted_envs]
        
        self.cst = cst
        
        #del c_flags[neg_halted_envs]
        #del self.problem['costs'].float()[neg_halted_envs]
        #del self.cst[self.halted_envs]
        
        # ktd = kendall_tau_distance_from_vectors(self.problem['V'], self.problem['X'], c_flags)
        # optimized ktd computation
        ktd = torch.zeros_like(self.ktd)
        ktd[neg_halted_envs] = kendall_tau_distance_from_vectors(self.problem['V'][neg_halted_envs], self.problem['X'][neg_halted_envs], c_flags[neg_halted_envs])
        ktd[self.halted_envs] = self.ktd[self.halted_envs]
        
        self.ktd = ktd
        
        # del self.problem['V'][neg_halted_envs]
        # del self.problem['X'][neg_halted_envs]
        # del c_flags[neg_halted_envs]
        # del self.ktd[self.halted_envs]
        
        #TODO optimize for halted_envs
        self.halted_kd = self.halted_kd | (self.halted_envs & (self.ktd > self.problem['k'].squeeze(-1))).to(self.device)
                
        r = torch.where(self.halted_kd, -torch.ones(self.batch_size), torch.zeros(self.batch_size)).to(self.device)
        
        #TODO optimize for halted_envs
        r = torch.where(self.halted_envs & ~self.halted_kd, ((2/torch.exp(self.cst)) - 1).to(self.device), r)
        
        #TODO optimize for halted_envs
        r = torch.where(
            neg_halted_envs,
            (
                2 / (
                    1 + torch.exp(
                        self.args.r_alpha * (torch.where(self.ktd > self.problem['k'].squeeze(-1), self.ktd, torch.zeros_like(self.ktd))) 
                        + self.args.r_beta * self.cst + self.args.r_gamma * torch.ones(self.batch_size)
                    )
                ) - 1
            ).to(self.device),
            r
        )
        
        self.running_time[neg_halted_envs] += 1
                
        return TensorDict(
            observation=self.state,
            reward=r,
            done=self.halted_envs,
            info=self.running_time.unsqueeze(-1),
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
        
    #env specific implementation
    def generate_problems(self) -> TensorDict:
        B = self.batch_size[0]

        n_candidates = torch.randint(self.args.MIN_CANDIDATES, self.args.MAX_CANDIDATES + 1, (B,), device=self.device)
        n_voters = torch.randint(self.args.MIN_VOTERS, self.args.MAX_VOTERS + 1, (B,), device=self.device)

        rand_scores_X = torch.rand(B, self.args.MAX_CANDIDATES, device=self.device)
        candidate_ids_X = torch.arange(1, self.args.MAX_CANDIDATES + 1, device=self.device).unsqueeze(0).expand(B, -1)
        
        mask_1_X = torch.arange(self.args.MAX_CANDIDATES, device=self.device).unsqueeze(0).expand(B, -1) >= n_candidates.unsqueeze(-1)       
        rand_scores_X[mask_1_X] = np.inf
        
        sorted_idx_X = rand_scores_X.argsort(dim=1)  # (B, sefl.args.MAX_CANDIDATES)
        X_full = candidate_ids_X.gather(1, sorted_idx_X)  # (B, self.args.MAX_CANDIDATES)

        cand_positions_X = torch.arange(self.args.MAX_CANDIDATES, device=self.device).unsqueeze(0)  # (1, self.args.MAX_CANDIDATES)
        mask_2_X = cand_positions_X < n_candidates.unsqueeze(1)  # (B, self.args.MAX_CANDIDATES), bool

        # X = X_full * mask_2_X.long()
        X = torch.where(mask_2_X, X_full, torch.full_like(X_full, self.args.SEQS_PADDING))


        rand_scores_V = torch.rand(B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES, device=self.device)
        candidate_ids_V = torch.arange(1, self.args.MAX_CANDIDATES + 1, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, self.args.MAX_VOTERS, -1)

        mask_1_V = torch.arange(self.args.MAX_CANDIDATES, device=self.device).unsqueeze(0).unsqueeze(0).expand(B, self.args.MAX_VOTERS, -1) >= n_candidates.unsqueeze(-1).unsqueeze(-1).expand(-1, self.args.MAX_VOTERS, -1)
        rand_scores_V[mask_1_V] = np.inf

        sorted_idx_V = rand_scores_V.argsort(dim=2)  # (B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES)
        V_full = candidate_ids_V.gather(2, sorted_idx_V)  # (B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES)

        cand_positions_V = torch.arange(self.args.MAX_CANDIDATES, device=self.device).unsqueeze(0).unsqueeze(0)  # (1,1, self.args.MAX_CANDIDATES)
        mask_V = cand_positions_V < n_candidates.unsqueeze(1).unsqueeze(2)  # (B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES)
        
        # V = V_full * mask_V.long()
        V = torch.where(mask_V, V_full, torch.full_like(V_full, self.args.SEQS_PADDING)) # (B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES)

        mask_2_V = torch.arange(0, self.args.MAX_VOTERS, device=self.device).unsqueeze(0).expand(B, -1) < n_voters.unsqueeze(-1) # (B, self.args.MAX_VOTERS)
        mask_2_V = mask_2_V.unsqueeze(-1) # (B, self.args.MAX_VOTERS, 1)
                
        V = torch.where(mask_2_V, V, torch.full_like(V, self.args.V_PADDING))

        costs_full = torch.rand(B, self.args.MAX_CANDIDATES, device=self.device)
        costs = ((self.args.MAX_COSTS - self.args.MIN_COSTS) * costs_full + self.args.MIN_COSTS).floor().long() + 1
        costs = torch.where(mask_2_X, costs, torch.full_like(costs, self.args.COSTS_PADDING))
        
        k = (torch.rand(B, device=self.device) * ((torch.square(n_candidates) - n_candidates).float()/self.args.K_DIVIDER)).floor().unsqueeze(1).long() + 1

        td = TensorDict({
            'X': X,                # (B, self.args.MAX_CANDIDATES): each row is a permutation of [1,..., n_candidates] with zeros after.
            'V': V,                # (B, self.args.MAX_VOTERS, self.args.MAX_CANDIDATES): each voter's ranking.
            'costs': costs,        # (B, self.args.MAX_CANDIDATES)
            'k': k,                # (B, 1)
            'n_candidates': n_candidates.unsqueeze(-1),  # (B, 1)
            'n_voters': n_voters.unsqueeze(-1),          # (B, 1)
        }, 
        batch_size=(B,),
        device=self.device)

        return td
        
    
if __name__ == "__main__":
    pass