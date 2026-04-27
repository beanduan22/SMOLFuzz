# https://github.com/pytorch/pytorch/issues/177829
# expect: SIGSEGV
import torch

LU_data = torch.tensor([[2., 3., 1.], [0.5, 1., 2.], [0.25, 0.5, 1.]])
LU_pivots = torch.tensor([], dtype=torch.int32)
torch.lu_unpack(LU_data, LU_pivots)
