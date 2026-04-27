# https://github.com/pytorch/pytorch/issues/173574
# expect: SIGFPE (Floating Point Exception)
import torch

out = torch.full((10,), 2, dtype=torch.int64)
torch.arange(2.0, 7.0, 0.5, out=out)
