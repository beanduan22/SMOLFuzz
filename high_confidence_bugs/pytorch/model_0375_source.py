# SMOLFuzz model 375 | llm=qwen2.5-coder:32b | attempts=1 | apis=7 | error=no
# USED_APIS_SELECTED = ['torch.random.fork_rng', 'torch.nn.functional.dropout', 'torch.nn.functional.fractional_max_pool3d', 'torch.Tensor.geometric_', 'torch.tanh', 'torch.ComplexFloatStorage', 'torch.ComplexDoubleStorage', 'torch.QInt8Storage', 'torch.div', 'torch.Tensor.sinc_', 'torch.linalg.norm', 'torch.quantize_per_tensor', 'torch.Tensor.scatter_add_', 'torch.pinverse', 'torch.Tensor.istft', 'torch.Tensor.true_divide_', 'torch.Tensor.lu_solve', 'torch.Tensor.igamma', 'torch.Tensor.fix', 'torch.nn.init.xavier_normal_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            x = F.tanh(self.fc1(x))
            x = self.bn(x)
            x = self.drop(x)
            x = self.fc2(x)
        
        # Switching modes
        self.train()
        x = self.bn(x)
        self.eval()
        x = self.bn(x)
        
        # Using dropout in eval mode (identity)
        x = self.drop(x)
        
        # Applying sinc_
        x.sinc_()
        
        # Linear algebra operations
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm + 1e-5)
        
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.nn.functional.dropout",
             "torch.tanh", "torch.Tensor.sinc_", "torch.linalg.norm", "torch.div"]