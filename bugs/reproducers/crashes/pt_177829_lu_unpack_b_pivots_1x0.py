# https://github.com/pytorch/pytorch/issues/177829
# expect: SIGSEGV  (variant: empty (1,0) pivots; (0,0) does NOT crash)
import torch

LU_data = torch.tensor([[2., 3., 1.], [0.5, 1., 2.], [0.25, 0.5, 1.]])
LU_pivots = torch.empty((1, 0), dtype=torch.int32)
torch.lu_unpack(LU_data, LU_pivots)
