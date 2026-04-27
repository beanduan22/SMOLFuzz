# https://github.com/pytorch/pytorch/issues/177829
# expect: SIGSEGV  (variant: batched 2x3x3 LU_data with empty int32 pivots)
import torch

LU_data = torch.stack([torch.eye(3)] * 2)
LU_pivots = torch.tensor([], dtype=torch.int32)
torch.lu_unpack(LU_data, LU_pivots)
