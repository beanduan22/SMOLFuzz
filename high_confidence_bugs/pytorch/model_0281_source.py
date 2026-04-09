# SMOLFuzz model 281 | llm=qwen2.5-coder:32b | attempts=2 | apis=10 | error=no
# USED_APIS_SELECTED = ['torch.is_grad_enabled', 'torch.nn.functional.multilabel_soft_margin_loss', 'torch.Tensor.ne_', 'torch.ravel', 'torch.Tensor.is_floating_point', 'torch.ByteStorage', 'torch.BoolStorage', 'torch.mul', 'torch.Tensor.addcmul', 'torch.Tensor.inverse', 'torch.nn.init.uniform_', 'torch.Tensor.element_size', 'torch.set_num_threads', 'torch.Tensor.triangular_solve', 'torch.set_printoptions', 'torch.Tensor.frexp', 'torch.quantize_per_tensor', 'torch.Tensor.bitwise_and', 'torch.Tensor.copysign_', 'torch.Tensor.fill_diagonal_', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = torch.sin(x)
            x = self.fc2(x)
            x = self.bn(x)
            loss = torch.nn.functional.multilabel_soft_margin_loss(x, torch.ones_like(x))
        with torch.no_grad():
            x = torch.mul(loss.unsqueeze(-1), x)  # Adjusted to maintain shape consistency
            x = self.fc2(x)
        x = torch.Tensor.ne_(x, torch.zeros_like(x)).float()
        x = torch.nn.init.uniform_(x, -1.0, 1.0)
        x = torch.Tensor.addcmul(torch.ones_like(x), x, x)
        x = torch.Tensor.fill_diagonal_(x, 5.0)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.sin", "torch.nn.functional.multilabel_soft_margin_loss",
             "torch.mul", "torch.Tensor.ne_", "torch.nn.init.uniform_",
             "torch.Tensor.addcmul", "torch.Tensor.fill_diagonal_", "torch.enable_grad", "torch.no_grad"]