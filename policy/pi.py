import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tensordict import TensorDict, TensorDictBase, tensorclass
from tensordict.nn import TensorDictModule, TensorDictSequential

from utils import to_perm_mat_2, GREEN, RESET, load_yaml_config
from envs import problem, kendall_tau_distance_from_vectors

CONFIG_FILE = os.path.join(os.getcwd(), "policy/config", "config.yaml")

@dataclass
class Args:
    prb_enc_x_featr_1: int = 128
    prb_enc_v_featr_1: int = 128
    prb_enc_cost_fetr: int = 128
    
    problem_embed_dim: int = 512
    
    num_heads: int = 16
    dropout: int = 0.1
    num_diffs: int = 32
    
    pi_h_dim1: int = 2048
    pi_h_dim2: int = 1024
    
    qnet_h_dim1: int = 2048
    qnet_h_dim2: int = 1024
    


class problem_Encoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        """
        x_featr_1: int
            The size of the first feature map(X).
        v_featr_1: int
            The size of the first feature map(V).
        cost_featr: int
            The size of the feature map for costs.
        f_featr: int
            The size of the final feature
        num_heads: int
            num_heads in mhsa
        dropout: float
            duh
        num_diffs: int
            num of loops
        MAX_CANDIDATES: int
        MAX_VOTERS: int
        """
        super().__init__()
        
        self.x_encdr = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((kwargs['x_featr_1'], kwargs['x_featr_1'])),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.BatchNorm1d(kwargs['x_featr_1']**2),
            nn.Linear(kwargs['x_featr_1']**2, kwargs['f_featr']),
            nn.LeakyReLU()
        )
        
        self.v_encdr = nn.Sequential(
            nn.Conv2d(kwargs['max_voters'], 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((kwargs['v_featr_1'], kwargs['v_featr_1'])),
            nn.Conv2d(256, 1, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.BatchNorm1d(kwargs['v_featr_1']**2),
            nn.Linear(kwargs['v_featr_1']**2, kwargs['f_featr']),
            nn.LeakyReLU()
        )
        
        self.costs_encdr = nn.Sequential(
            nn.Linear(kwargs['max_candidates'], 128),
            nn.ReLU(),
            nn.Linear(128, kwargs['f_featr']),
            nn.LeakyReLU()
        )
        
        self.k_encdr = nn.Sequential(
            nn.Linear(1, kwargs['f_featr']),
            nn.ReLU(),
        )
        
        self.n_candidates_encdr = nn.Sequential(
            nn.Linear(1, kwargs['f_featr']),
            nn.ReLU()
        )
        
        self.n_voters_encdr = nn.Sequential(
            nn.Linear(1, kwargs['f_featr']),
            nn.ReLU()
        )
        
        self.mhsa = nn.MultiheadAttention(embed_dim=kwargs['f_featr'], num_heads=kwargs['num_heads'], dropout=kwargs['dropout'], add_bias_kv=True, add_zero_attn=True, kdim=None, vdim=None, batch_first=True)
        
        self.sa_mlp = nn.Sequential(
            nn.Linear(kwargs['f_featr']*4, kwargs['f_featr']*8),
            nn.LeakyReLU(),
            nn.Linear(kwargs['f_featr']*8, kwargs['f_featr']*4),
            nn.LeakyReLU()
        )
        
        self.num_diffs = kwargs['num_diffs']
    
    def forward(self, X: torch.Tensor, V: torch.Tensor, costs: torch.Tensor, k: torch.Tensor, n_candidates: torch.Tensor, n_voters: torch.Tensor) -> torch.Tensor:
        
        batch_size = X.shape[0]
        
        X = to_perm_mat_2(X)
        V = to_perm_mat_2(V)
        
        X = X.unsqueeze(1)

        X = self.x_encdr(X)
        V = self.v_encdr(V)
        costs = self.costs_encdr(costs)
        k = self.k_encdr(k)
        n_candidates = self.n_candidates_encdr(n_candidates)
        n_voters = self.n_voters_encdr(n_voters)
        
        embed = torch.cat([X, V, costs, k, n_candidates, n_voters], dim=0).view(batch_size, 6, -1)
        
        o_embed = embed.detach().clone()
        for _ in range(self.num_diffs):
            embed, _ = self.mhsa(embed, embed, embed)
            embed = embed + o_embed
            embed = embed.reshape(batch_size, -1)
            embed = self.sa_mlp(embed)
            embed = embed.view(batch_size, 4, -1)
            del o_embed
            o_embed = embed.detach().clone()
            
        return o_embed[:, -1, :]
            

class π(nn.Module):
    def __init__(self, **kwargs) -> None:
        '''
        tape_size: int
            the size of the state storing tape
        MAX_CANDIDATES: int
        MAX_VOTERS
        '''
        super().__init__()
        
        self.args = Args(**load_yaml_config(CONFIG_FILE))
        
        self.prblm_enc = TensorDictModule(problem_Encoder(x_featr_1=self.args.prb_enc_x_featr_1, v_featr_1=self.args.prb_enc_v_featr_1,\
            cost_featr=self.args.prb_enc_cost_fetr, f_featr=self.args.problem_embed_dim,\
            num_heads=self.args.num_heads, dropout=self.args.dropout, num_diffs=self.args.num_diffs,\
            max_candidates=kwargs['max_candidates'], max_voters=kwargs['max_voters']), \
            in_keys=['X', 'V', 'costs', 'k', 'n_candidates', 'n_voters'], out_keys=['embed'])
        
        self.probs = None
        
        self.tape_size = kwargs['tape_size']
        
        self.prb_embed_dim = self.args.problem_embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.prb_embed_dim + self.tape_size, self.args.pi_h_dim1),
            nn.LeakyReLU(),
            nn.Linear(self.args.pi_h_dim1, self.args.pi_h_dim2),
            nn.ReLU(),
            nn.Linear(self.args.pi_h_dim2, self.tape_size*self.tape_size + self.tape_size),
            nn.LeakyReLU()
        )
    
    def reset(self, probs: TensorDictBase) -> None:
        del self.probs
        self.prblm_enc(probs)
        self.probs = probs
        
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prb_embds = self.probs.embed
        
        tmp_embds = torch.cat([prb_embds, state], dim=1)
        
        tmp_embds = self.net(tmp_embds)
        
        w, b = tmp_embds[:, :self.tape_size*self.tape_size].reshape(-1, self.tape_size, self.tape_size), tmp_embds[:, self.tape_size*self.tape_size: ]
        
        return w, b
        
class Qnet(nn.Module):
    def __init__(self, **kwargs) -> None:
        '''
        tape_size: int
            the size of the state storing tape
        problem_encoder: nn.Module
        '''
        super().__init__()
        
        self.prblm_enc = kwargs['problem_encoder']
        
        self.probs = None
        
        self.args = Args(**load_yaml_config(CONFIG_FILE))
        
        self.tape_size = kwargs['tape_size']
        
        self.prb_embed_dim = self.args.problem_embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.prb_embed_dim + 2 * self.tape_size + self.tape_size*self.tape_size, self.args.qnet_h_dim1),
            nn.LeakyReLU(),
            nn.Linear(self.args.qnet_h_dim1, self.args.qnet_h_dim2),
            nn.ReLU(),
            nn.Linear(self.args.qnet_h_dim2, 1),
            nn.LeakyReLU()
        )
    
    def reset(self, probs: TensorDict) -> None:
        del self.probs
        self.prblm_enc(probs)
        self.probs = probs
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        prb_embds = self.probs.embed
        
        batch_size = state.shape[0]
        action = action.view(batch_size, -1)
        
        tmp_embds = torch.cat([prb_embds, state, action], dim=1)
        
        q_value = self.net(tmp_embds)
        
        return q_value   
        

if __name__ == "__main__":
    pass