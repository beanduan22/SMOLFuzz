# SMOLFuzz model 467 | llm=qwen2.5-coder:32b | attempts=2 | apis=6 | error=no
# USED_APIS_SELECTED = ['torch.distributions.constraints.Constraint', 'torch.Tensor.normal_', 'torch.nn.functional.dropout', 'torch.nn.functional.adaptive_avg_pool3d', 'torch.geqrf', 'torch.Tensor.ravel', 'torch.Generator', 'torch.as_tensor', 'torch.Tensor.char', 'torch.arctan', 'torch.Tensor.matrix_exp', 'torch.utils.model_zoo.load_url', 'torch.Tensor.retains_grad', 'torch.set_printoptions', 'torch.isposinf', 'torch.Tensor.element_size', 'torch.is_tensor', 'torch.Tensor.greater_equal', 'torch.Tensor.solve', 'torch.Tensor.erfc', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.bn(x)
            x = self.drop(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.fc2(x)
            x = torch.arctan(x)
            x = torch.matrix_exp(x)
            x = x.greater_equal(0).float()
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.nn.functional.dropout",
             "torch.arctan", "torch.matrix_exp", "torch.Tensor.greater_equal"]