# SMOLFuzz model 404 | llm=llama3.3:70b | attempts=1 | apis=7 | error=no
# USED_APIS_SELECTED = ['torch.distributions.constraints.Constraint', 'torch.nn.Unfold', 'torch.nn.Sequential', 'torch.Tensor.le_', 'torch.Tensor.fliplr', 'torch.frombuffer', 'torch.Tensor.size', 'torch.IntStorage', 'torch.Tensor.bmm', 'torch.maximum', 'torch.Tensor.sub', 'torch.set_num_threads', 'torch.nn.init.kaiming_normal_', 'torch.orgqr', 'torch.compiled_with_cxx11_abi', 'torch.nn.UpsamplingBilinear2d', 'torch.Tensor.as_subclass', 'torch.bitwise_right_shift', 'torch.quantize_per_channel', 'torch.Tensor.igamma_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = torch.maximum(y, torch.sin(y))
            w = torch.sub(z, torch.cos(y))
            v = torch.bmm(w.unsqueeze(2), w.unsqueeze(1)).squeeze()
            u = self.fc2(v)
        return u

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.maximum", "torch.sin", 
             "torch.cos", "torch.sub", "torch.bmm", "torch.enable_grad"]