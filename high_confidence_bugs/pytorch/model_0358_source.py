# SMOLFuzz model 358 | llm=llama3.3:70b | attempts=3 | apis=9 | error=no
# USED_APIS_SELECTED = ['torch.nn.PixelUnshuffle', 'torch.nn.functional.softplus', 'torch.geqrf', 'torch.Tensor.ravel', 'torch.Tensor.view_as', 'torch.QInt8Storage', 'torch.Tensor.is_cuda', 'torch.Tensor.cuda', 'torch.mv', 'torch.Tensor.diag', 'torch.Tensor.arccosh_', 'torch.can_cast', 'torch.Tensor.q_scale', 'torch.Tensor.igamma_', 'torch.unique', 'torch.set_default_dtype', 'torch.nn.init.xavier_uniform_', 'torch.use_deterministic_algorithms', 'torch.diagonal', 'torch.nn.init.constant_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = F.softplus(y)
            w = torch.sin(z) * torch.cos(z)
            v = torch.diag(w.sum(dim=0))
            u = torch.geqrf(v)[0]
            t = u.t()
            s = torch.mv(t, x.sum(dim=0)).view(1, 8).expand(x.size(0), 8)
            r = torch.unique(s, dim=0)
            q = self.fc2(r.float())
        return q

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.functional.softplus",
             "torch.sin", "torch.cos", "torch.diag",
             "torch.geqrf", "torch.mv", "torch.unique",
             "torch.enable_grad"]