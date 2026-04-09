# SMOLFuzz model 346 | llm=llama3.3:70b | attempts=5 | apis=8 | error=no
# USED_APIS_SELECTED = ['torch.hann_window', 'torch.nn.functional.mse_loss', 'torch.nn.RNNBase', 'torch.Tensor.lerp_', 'torch.Tensor.new_empty', 'torch.Tensor.ravel', 'torch.Tensor.cuda', 'torch.FloatStorage', 'torch.Tensor.double', 'torch.linalg.svd', 'torch.Tensor.cumprod_', 'torch.Tensor.copysign_', 'torch.Tensor.lu_solve', 'torch.Tensor.isposinf', 'torch.Tensor.bitwise_and', 'torch.Tensor.erf_', 'torch.Tensor.cholesky_solve', 'torch.utils.dlpack.from_dlpack', 'torch.digamma', 'torch.Tensor.erfinv', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.hann_window(8).to(x.device)
            z = torch.sin(x) * torch.cos(y.unsqueeze(0))
            w = self.fc1(z)
            v = torch.erf(w)
            u = torch.cumprod(v, dim=1)
            t = self.fc2(u)
            s = nn.functional.mse_loss(t, x)
        return s

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = [
    "torch.hann_window",
    "torch.nn.Linear",
    "torch.sin",
    "torch.cos",
    "torch.erf",
    "torch.cumprod",
    "torch.nn.functional.mse_loss",
    "torch.enable_grad"
]

if __name__ == "__main__":
    model = Model()
    inputs = make_inputs()
    output = model(inputs[0])
    print(output)