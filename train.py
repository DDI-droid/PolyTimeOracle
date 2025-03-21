import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase

from envs import PolyTimeOracle, kendall_tau_distance_from_vectors
from pi import problem, problem_Encoder, Qnet, π, MAX_CANDIDATES, MAX_VOTERS
from utils import to_perm_mat_2

if __name__ == "__main__":
    # probs = problem(
    #     X=torch.zeros(32, MAX_CANDIDATES),
    #     V=torch.zeros(32, MAX_VOTERS, MAX_CANDIDATES),
    #     costs=torch.zeros(32, MAX_CANDIDATES),
    #     k=torch.zeros(32, 1)
    # )
    
    # π_ = π(tape_size=1024, prb_embed_dim=512, h_dim_1=2048, h_dim_2=1024)
    # π_.reset(probs)
    
    # pt = PolyTimeOracle(tape_size=1024, batch_size=(32, ))
    # t_dct = pt.reset(problem=probs.embed)

    # w, b = π_(t_dct['observation'])
    
    # action = TensorDict(
    #     w=w,
    #     b=b,
    #     batch_size=w.shape[0]
    # )
    
    # t_dct = pt.step(action)
    ranks = torch.tensor([0, 1, -1, -1])
    x = torch.tensor([1, 0, -1, -1])
    
    print(kendall_tau_distance_from_vectors(ranks.unsqueeze(0), x.unsqueeze(0), exclude=torch.zeros_like(ranks[0, ...]).unsqueeze(0))) 