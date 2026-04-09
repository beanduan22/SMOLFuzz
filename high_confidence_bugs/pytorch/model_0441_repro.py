#!/usr/bin/env python3
"""
SMOLFuzz bug reproducer  —  CRASH
Model : 441
Mutation: scale_small
Detail  : CPU-only crash: Traceback (most recent call last):
  File "/home/binduan/myspace/smolfuzz/results/run_500_pytorch/full/w
APIs    : torch.nn.Linear, torch.nn.ReLU, torch.Tensor.sigmoid_, torch.logcumsumexp, torch.nn.functional.adaptive_avg_pool1d, torch.Tensor.corrcoef, torch.Tensor.cholesky, torch.enable_grad

Run:  python3 bug_crash_m0441_mut2_20260410_043056.repro.py
Requires: PyTorch with CUDA available.
"""
import torch
import torch.nn as nn
from pathlib import Path

# ── Model definition (LLM-generated) ──────────────────────────
 
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

# ── Load buggy inputs ─────────────────────────────────────────
_here = Path(__file__).parent
inputs = torch.load(_here / 'bug_crash_m0441_mut2_20260410_043056.inputs.pt', weights_only=False)

# ── Build models with identical weights ───────────────────────
torch.manual_seed(42)
cpu_model = Model().cpu()
gpu_model = Model().cuda()
gpu_model.load_state_dict(cpu_model.state_dict())  # identical weights

# ── CUDA succeeds ────────────────────────────────────────
ok_inputs = [x.to('cuda') if isinstance(x, torch.Tensor) else x for x in inputs]
ok_out = cpu_model(*ok_inputs)
print(f'CUDA output: {ok_out}')

# ── CPU crashes ───────────────────────────────────────
bad_inputs = [x.to('cpu') if isinstance(x, torch.Tensor) else x for x in inputs]
bad_out = gpu_model(*bad_inputs)  # <── expect crash here
print(f'CPU output: {bad_out}')
