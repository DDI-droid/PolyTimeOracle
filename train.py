import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase

from envs import PolyTimeOracle, kendall_tau_distance_from_vectors
from pi import Qnet, π
from utils import to_perm_mat_2

if __name__ == "__main__":
    pass
    # pt = PolyTimeOracle(tape_size=1024, batch_size=(32, ))
    # t_dct = pt.reset()    
    
    # pt.step(pt.action_spec.sample())
    # π_ = π(tape_size=pt.tape_size, prb_embed_dim=512, h_dim_1=2048, h_dim_2=1024)
    # π_.reset(t_dct['problem'])

    # w, b = π_(t_dct['observation'])
    
    # action = TensorDict(
    #     w=w,
    #     b=b,
    #     batch_size=w.shape[0]
    # )
    
    #t_dct = pt.step(action)
    
    
    # ranks = torch.tensor([[[0, 0, 0, 0], [2, 3, 1, 4]], [[1, 2, 3, 0], [2, 3, 1, 0]]])
    # x = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0] ])
        
    # print(kendall_tau_distance_from_vectors(ranks, x, exclude=torch.tensor([0, 0, 0, 0]).unsqueeze(0).expand(2, -1))) 