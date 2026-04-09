# SMOLFuzz model 147 | llm=qwen2.5-coder:32b | attempts=1 | apis=8 | error=no
# USED_APIS_SELECTED = ['torch.distributions.transforms.ReshapeTransform', 'torch.kaiser_window', 'torch.nn.PixelShuffle', 'torch.all', 'torch.testing.assert_close', 'torch.HalfStorage', 'torch.full', 'torch.Tensor.subtract', 'torch.Tensor.std', 'torch.conj', 'torch.Tensor.expm1', 'torch.Tensor.polygamma_', 'torch.nn.init.kaiming_normal_', 'torch.Tensor.qscheme', 'torch.atleast_3d', 'torch.sigmoid', 'torch.dsplit', 'torch.cov', 'torch.nn.utils.skip_init', 'torch.Tensor.igamma', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.bn(x)
            x = torch.sigmoid(x)
            x = self.fc2(x)
            x = x.expm1()
            x = x.polygamma_(1)
            x = x.std(dim=0, keepdim=True)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.sigmoid", 
             "torch.Tensor.expm1", "torch.Tensor.polygamma_", "torch.Tensor.std", 
             "torch.enable_grad", "torch.Tensor.requires_grad_"]