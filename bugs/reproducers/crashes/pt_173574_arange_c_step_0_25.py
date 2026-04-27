# https://github.com/pytorch/pytorch/issues/173574
# expect: SIGFPE  (variant: step=0.25 into int64 out)
import torch

out = torch.zeros((20,), dtype=torch.int64)
torch.arange(0.0, 1.0, 0.25, out=out)
