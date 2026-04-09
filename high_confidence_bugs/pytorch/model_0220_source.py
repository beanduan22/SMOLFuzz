# SMOLFuzz model 220 | llm=llama3.3:70b | attempts=2 | apis=6 | error=no
# USED_APIS_SELECTED = ['torch.nn.functional.margin_ranking_loss', 'torch.nn.functional.upsample', 'torch.Tensor.masked_fill_', 'torch.tanh', 'torch.Tensor.unfold', 'torch.BFloat16Storage', 'torch.as_tensor', 'torch.Tensor.sinh_', 'torch.fft.rfft2', 'torch.Tensor.divide_', 'torch.ceil', 'torch.Tensor.frexp', 'torch.Tensor.lu_solve', 'torch.nn.utils.rnn.pad_packed_sequence', 'torch.set_warn_always', 'torch.complex', 'torch.Tensor.element_size', 'torch.Tensor.index_fill_', 'torch.Tensor.diagflat', 'torch.Tensor.i0_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = F.relu(x)
            z = torch.tanh(y)
            w = self.fc1(z)
            v = torch.sin(w) * torch.cos(w)
            u = self.fc2(v)
            out = torch.fft.rfft2(u)
        return out

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = [
    "torch.nn.Linear", 
    "torch.tanh", 
    "torch.sin", 
    "torch.cos", 
    "torch.enable_grad", 
    "torch.fft.rfft2"
]