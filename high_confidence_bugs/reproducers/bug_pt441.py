#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 441
Bug Type: crash
Root Cause: corrcoef() → cholesky() pipeline: scale_small mutation makes correlation
            matrix non-positive-definite on CPU (LAPACK raises RuntimeError), but
            cuSOLVER on GPU tolerates it and returns a result.
Detail  : CPU-only crash — RuntimeError from cholesky() on CPU, GPU succeeds
APIs    : torch.nn.Linear, torch.nn.ReLU, torch.Tensor.sigmoid_, torch.logcumsumexp,
          torch.nn.functional.adaptive_avg_pool1d, torch.Tensor.corrcoef, torch.Tensor.cholesky
Mutation: scale_small

Run: python3 bug_pt441.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ─────────────────────────────────────────────────────────────────────
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
            x = F.adaptive_avg_pool1d(x, 8)
            x = x.corrcoef()
            x = self.relu(x)
            x = x.cholesky()
            x = x.sum(dim=1)
        return x

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact scale_small mutated inputs that triggered the crash (embedded from saved .pt file)
    inp0 = torch.tensor([[-0.025408625602722168, -0.00732085295021534, 0.01323717087507248, -0.007933376356959343, -0.0036197020672261715, -0.01284149568527937, 0.026837142184376717, -0.023954732343554497], [-0.02601183019578457, -0.021634312346577644, -0.01964784413576126, 0.015940474346280098, 0.0033350379671901464, 0.022145528346300125, 0.008589751087129116, -0.03545458987355232], [0.006427152547985315, -0.03092769905924797, 0.02781722880899906, -0.013294767588376999, -0.006846986711025238, 0.017567157745361328, -0.01650809496641159, -0.0011404029792174697], [-0.009468961507081985, -0.018325502052903175, -0.0006618089973926544, -0.004749863874167204, 0.0038775105495005846, 0.006596426013857126, 0.019777396693825722, 0.024068189784884453]], dtype=torch.float32)
    inputs = [inp0]

    torch.manual_seed(42)
    cpu_model = Model()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gpu_model = Model().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())

    # GPU succeeds
    try:
        gpu_out = gpu_model(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs])
        print(f"GPU output: {gpu_out.detach().cpu()}")
        gpu_ok = True
    except Exception as e:
        print(f"GPU crashed: {e}")
        gpu_ok = False

    # CPU crashes
    try:
        cpu_out = cpu_model(*inputs)
        print(f"CPU output: {cpu_out.detach()}")
        cpu_ok = True
    except Exception as e:
        print(f"CPU crashed: {e}")
        cpu_ok = False

    if gpu_ok and not cpu_ok:
        print("BUG CONFIRMED: GPU succeeds but CPU crashes on the same model and input")
    elif cpu_ok and not gpu_ok:
        print("BUG CONFIRMED: CPU succeeds but GPU crashes on the same model and input")
    elif cpu_ok and gpu_ok:
        print("Both succeeded — crash not reproduced on this hardware/driver version")
    else:
        print("Both crashed — differential crash not reproduced")


if __name__ == "__main__":
    run()
