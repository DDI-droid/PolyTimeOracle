import time
from sys import argv
import torch
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase

from envs import PolyTimeOracle, kendall_tau_distance_from_vectors
from policy import Qnet, π
from utils import to_perm_mat_2, GREEN, RED, RESET

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    batch_size = (int(argv[1]), )
    env = PolyTimeOracle(batch_size=batch_size)
    env.to(device)

    # GPU warmup
    _ = env.reset()
    num_warmup_steps = 10
    for _ in range(num_warmup_steps):
        _ = env.step(env.action_spec.sample())
    
    num_steps = int(argv[2])
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_steps):
        _ = env.step(env.action_spec.sample()) 
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    elapsed_time = end_time - start_time
    steps_per_second = num_steps / elapsed_time
    print(f"Max steps per second (batched on {device}): {batch_size[0] * steps_per_second:.2f}")