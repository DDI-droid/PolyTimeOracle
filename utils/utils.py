import torch
import torch.nn as nn

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

# def tensordict_module(in_keys, out_keys):
#     def wrapper(cls):
#         if not issubclass(cls, nn.Module):
#             raise TypeError(f"Expected a subclass of nn.Module, got {cls.__name__}")
        
#         class WrappedTDModule(cls):
#             def __init__(self, *args, **kwargs):
#                 super().__init__(*args, **kwargs)
#                 self.td_module = TensorDictModule(
#                     module=self,
#                     in_keys=in_keys,
#                     out_keys=out_keys
#                 )

#             def forward(self, tensordict):
#                 return self.td_module(tensordict)

#         return WrappedTDModule
#     return wrapper

def to_perm_mat_1(rankings: torch.Tensor):
    """Converts a batched ranking to a batched permutation matrix."""
    
    if len(rankings.shape) not in  (2, 3):
        raise ValueError(f"Expected a 2D or 3D tensor, got {rankings.shape}")
    
    batch_size, n = rankings.shape[0], rankings.shape[-1]
    
    eye = torch.eye(n, device=rankings.device)
    
    perm_matrices = eye[rankings]
    
    return perm_matrices

def to_perm_mat_2(rankings: torch.Tensor, pad_value: int = -1):
    """
    Converts a batched ranking to a batched permutation matrix, handling end-padding.
    
    For valid rankings, expected values are in 0, 1, ..., n-1.
    If a ranking is padded with pad_value at the end, that position will yield a row of zeros 
    in the corresponding permutation matrix.
    
    Args:
        rankings (torch.Tensor): A 2D tensor of shape (voters, n) or a 3D tensor of shape (batch, voters, n).
        pad_value (int): The value used for padding (default: -1).
    
    Returns:
        torch.Tensor: A permutation matrix tensor with an added dimension.
            For a 2D input (voters, n), the output is (voters, n, n).
            For a 3D input (batch, voters, n), the output is (batch, voters, n, n).
    """
    if len(rankings.shape) not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D tensor, got {rankings.shape}")
    
    n = rankings.shape[-1]
    
    eye_extended = torch.cat([torch.eye(n, device=rankings.device), 
                              torch.zeros(1, n, device=rankings.device)], dim=0)
    
    rankings_fixed = rankings.clone()
    rankings_fixed[rankings_fixed == pad_value] = n
    
    rankings_fixed = rankings_fixed.long()
    
    perm_matrices = eye_extended[rankings_fixed]
    return perm_matrices

if __name__ == "__main__":
    ranks = torch.tensor([[[0, 1, 2], [1, 0, -1]], [[0, 1, 2], [0, -1, -1]]])
    print(to_perm_mat_2(ranks))