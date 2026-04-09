# SMOLFuzz model 295 | llm=qwen2.5-coder:32b | attempts=1 | apis=7 | error=no
# USED_APIS_SELECTED = ['torch.distributions.transforms.ReshapeTransform', 'torch.nn.Softshrink', 'torch.equal', 'torch.trapz', 'torch.Tensor.to_mkldnn', 'torch.HalfStorage', 'torch.Tensor.clone', 'torch.linalg.householder_product', 'torch.special.xlogy', 'torch.Tensor.remainder', 'torch.Tensor.sspaddmm', 'torch.Tensor.qscheme', 'torch.set_default_tensor_type', 'torch.dequantize', 'torch.Tensor.scatter_add_', 'torch.Tensor.histc', 'torch.Tensor.device', 'torch.Tensor.histogram', 'torch.Tensor.stft', 'torch.nn.init.constant_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 4)
        self.softshrink = nn.Softshrink(lambd=0.5)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.linear1(x)
            x = torch.sin(x)
            x = self.softshrink(x)
            x = self.linear2(x)
            self.train()
            x = self.bn(x)
            x = self.bn(x)  # Reuse BN to demonstrate mode change
            self.eval()
        x = x.remainder(1.0)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.sin", "torch.nn.Softshrink",
             "torch.nn.BatchNorm1d", "torch.Tensor.requires_grad_",
             "torch.enable_grad", "torch.Tensor.remainder"]