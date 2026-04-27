# https://github.com/pytorch/pytorch/issues/173574
# expect: SIGFPE  (variant: negative range, fractional step, int64 out)
import torch

out = torch.zeros((20,), dtype=torch.int64)
torch.arange(-2.0, 2.0, 0.5, out=out)
