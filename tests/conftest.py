import torch

# Test models are tiny; extra threads only add contention on a shared CPU.
torch.set_num_threads(1)
