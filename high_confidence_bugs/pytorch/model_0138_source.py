# SMOLFuzz model 138 | llm=llama3.3:70b | attempts=1 | apis=8 | error=no
# USED_APIS_SELECTED = ['torch.distributions.transforms.ReshapeTransform', 'torch.nn.PixelShuffle', 'torch.equal', 'torch.chunk', 'torch.Tensor.resize_', 'torch.empty', 'torch.Tensor.int_repr', 'torch.Tensor.subtract', 'torch.Tensor.argmin', 'torch.Tensor.logit_', 'torch.Tensor.sign_', 'torch.fft.rfft', 'torch.nn.NLLLoss', 'torch.atleast_2d', 'torch.cholesky_solve', 'torch.Tensor.index_put_', 'torch.nn.Identity', 'torch.nn.utils.skip_init', 'torch.eig', 'torch.Tensor.sparse_mask', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.identity = nn.Identity()

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.sin(x)
            z = self.fc1(y)
            w = torch.cos(z)
            v = self.fc2(w)
            u = torch.tanh(v)
            jacobian = torch.autograd.functional.jacobian(lambda t: torch.sin(self.fc1(t)), x)
            loss = nn.NLLLoss()(u, torch.empty_like(u).argmin(dim=1))
            with torch.no_grad():
                cholesky_solve_result = torch.cholesky_solve(torch.eye(8), torch.eye(8))
        return u + jacobian.sum() + loss + cholesky_solve_result.sum()

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = [
    "torch.nn.Linear",
    "torch.sin",
    "torch.cos",
    "torch.tanh",
    "torch.autograd.functional.jacobian",
    "torch.enable_grad",
    "nn.NLLLoss",
    "torch.cholesky_solve"
]