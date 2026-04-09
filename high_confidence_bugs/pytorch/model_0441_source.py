# SMOLFuzz model 441 | llm=qwen2.5-coder:32b | attempts=2 | apis=8 | error=no
# USED_APIS_SELECTED = ['torch.distributions.transforms.AbsTransform', 'torch.multinomial', 'torch.nn.functional.adaptive_avg_pool3d', 'torch.nn.ReLU', 'torch.Tensor.allclose', 'torch.Tensor.index_add', 'torch.Tensor.to_mkldnn', 'torch.CharStorage', 'torch.QInt32Storage', 'torch.Tensor.sigmoid_', 'torch.Tensor.logcumsumexp', 'torch.Tensor.corrcoef', 'torch.Tensor.cholesky', 'torch._assert', 'torch.Tensor.is_contiguous', 'torch.Tensor.erf', 'torch.Tensor.q_per_channel_scales', 'torch.nn.Unflatten', 'torch.Tensor.as_subclass', 'torch.utils.model_zoo.load_url', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch; import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = torch.sigmoid_(x)
            x = torch.logcumsumexp(x, dim=1)
            x = torch.nn.functional.adaptive_avg_pool1d(x, 8)
            x = x.corrcoef()
            x = self.relu(x)
            x = x.cholesky()
            x = x.sum(dim=1)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.ReLU", "torch.Tensor.sigmoid_", 
             "torch.logcumsumexp", "torch.nn.functional.adaptive_avg_pool1d",
             "torch.Tensor.corrcoef", "torch.Tensor.cholesky", "torch.enable_grad"]