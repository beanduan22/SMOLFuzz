# SMOLFuzz model 192 | llm=llama3.3:70b | attempts=2 | apis=7 | error=no
# USED_APIS_SELECTED = ['torch.distributions.transforms.IndependentTransform', 'torch.rand', 'torch.nn.functional.pixel_unshuffle', 'torch.Tensor.geqrf', 'torch.narrow', 'torch.Tensor.expand_as', 'torch.Tensor.short', 'torch.frombuffer', 'torch.Tensor.addcdiv_', 'torch.logsumexp', 'torch.log2', 'torch.Tensor.rad2deg', 'torch.cdist', 'torch.Tensor.igamma_', 'torch.Tensor.put_', 'torch.nn.utils.weight_norm', 'torch.copysign', 'torch.isin', 'torch.set_warn_always', 'torch.nn.parameter.UninitializedParameter', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = torch.sin(y) * torch.cos(y)
            w = torch.logsumexp(z, dim=1, keepdim=True)
            v = torch.copysign(w, z)
            u = v + 0.5 * (0.3 / (0.2 + 1e-6))
            t = torch.log2(u)
        return t

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.sin", "torch.cos",
             "torch.logsumexp", "torch.copysign", "torch.log2", "torch.enable_grad"]