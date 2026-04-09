# SMOLFuzz model 241 | llm=qwen2.5-coder:32b | attempts=1 | apis=9 | error=no
# USED_APIS_SELECTED = ['torch.distributions.constraints.Constraint', 'torch.nn.ReLU', 'torch.nn.functional.relu', 'torch.Tensor.get_device', 'torch.tanh', 'torch.QInt32Storage', 'torch.Tensor.is_pinned', 'torch.Tensor.multinomial', 'torch.Tensor.logdet', 'torch.Tensor.tan_', 'torch.bmm', 'torch.stft', 'torch.set_flush_denormal', 'torch.Tensor.i0_', 'torch.nanquantile', 'torch.Tensor.cholesky_solve', 'torch.nn.LogSoftmax', 'torch.Tensor.sparse_mask', 'torch.range', 'torch.vsplit', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.relu(x)
            x = self.logsoftmax(x)
            x = self.fc2(x)
            y = torch.tanh(x)
            z = torch.multinomial(y.exp(), num_samples=1).float()
            w = torch.bmm(z.unsqueeze(2), y.unsqueeze(1))
            v = w.squeeze()
        return v

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.ReLU", "torch.nn.functional.relu", 
             "torch.Tensor.requires_grad_", "torch.enable_grad", 
             "torch.nn.LogSoftmax", "torch.tanh", "torch.Tensor.multinomial", 
             "torch.bmm"]