# SMOLFuzz model 191 | llm=qwen2.5-coder:32b | attempts=2 | apis=9 | error=no
# USED_APIS_SELECTED = ['torch.distributions.constraints.Constraint', 'torch.randint', 'torch.nn.functional.adaptive_avg_pool3d', 'torch.le', 'torch.Tensor.split', 'torch.Tensor.diag_embed', 'torch.FloatStorage', 'torch.ShortStorage', 'torch.Tensor.arccos', 'torch.Tensor.t_', 'torch.Tensor.acosh_', 'torch.Tensor.arcsin_', 'torch.set_flush_denormal', 'torch.cholesky', 'torch.Tensor.record_stream', 'torch.set_warn_always', 'torch.frexp', 'torch.Tensor.put_', 'torch.pca_lowrank', 'torch.Tensor.kthvalue', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # Gradient-tracking scope changes semantics of subsequent ops
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.fc1(x)
            z = self.bn(y)
            w = torch.sin(z)
            v = self.fc2(w)
            u = self.drop(v)
            t = torch.arccos(torch.clamp(u, -0.999, 0.999))
        
        # Mode switch affects BatchNorm and Dropout
        self.train()
        a = self.bn(t)
        self.eval()
        b = self.fc2(a)
        
        # More operations with gradient tracking
        c = torch.acosh(torch.ones_like(b) + 1e-6)
        d, _ = b.kthvalue(1, dim=1, keepdim=True)
        e = d.expand_as(b)
        f = b - e
        
        return f

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.sin",
             "torch.arccos", "torch.clamp", "torch.nn.Dropout", 
             "torch.acosh", "torch.Tensor.expand_as", "torch.enable_grad"]