#!/usr/bin/env python3
"""
SMOLFuzz bug reproducer  —  INCONSISTENT
Model : 191
Mutation: add_noise
Detail  : output[0]: l2=1.5129e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.sin, torch.arccos, torch.clamp, torch.nn.Dropout, torch.acosh, torch.Tensor.expand_as, torch.enable_grad

Run:  python3 bug_inconsistent_m0191_mut1_20260409_224510.repro.py
Requires: PyTorch with CUDA available.
"""
import torch
import torch.nn as nn
from pathlib import Path

# ── Model definition (LLM-generated) ──────────────────────────

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # Gradient-tracking scope changes semantics of subsequent ops
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.fc1(x)
            z = self.bn(y)
            w = torch.sin(z)
            v = self.fc2(w)
            u = self.drop(v)
            t = torch.arccos(torch.clamp(u, -0.999, 0.999))
        
        # Mode switch affects BatchNorm and Dropout
        self.train()
        a = self.bn(t)
        self.eval()
        b = self.fc2(a)
        
        # More operations with gradient tracking
        c = torch.acosh(torch.ones_like(b) + 1e-6)
        d, _ = b.kthvalue(1, dim=1, keepdim=True)
        e = d.expand_as(b)
        f = b - e
        
        return f

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.sin",
             "torch.arccos", "torch.clamp", "torch.nn.Dropout", 
             "torch.acosh", "torch.Tensor.expand_as", "torch.enable_grad"]

# ── Load buggy inputs ─────────────────────────────────────────
_here = Path(__file__).parent
inputs = torch.load(_here / 'bug_inconsistent_m0191_mut1_20260409_224510.inputs.pt', weights_only=False)

# ── Build models with identical weights ───────────────────────
torch.manual_seed(42)
cpu_model = Model().cpu()
gpu_model = Model().cuda()
gpu_model.load_state_dict(cpu_model.state_dict())  # identical weights

# ── CPU run ───────────────────────────────────────────────────
cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
cpu_out = cpu_model(*cpu_inputs)
cpu_outs = [cpu_out] if isinstance(cpu_out, torch.Tensor) else list(cpu_out)

# ── GPU run ───────────────────────────────────────────────────
gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
gpu_out = gpu_model(*gpu_inputs)
gpu_outs = [gpu_out] if isinstance(gpu_out, torch.Tensor) else list(gpu_out)

# ── Compare (L2 norm on finite positions) ─────────────────────
for i, (c, g) in enumerate(zip(cpu_outs, gpu_outs)):
    if isinstance(c, torch.Tensor) and isinstance(g, torch.Tensor):
        c_f, g_f = c.float(), g.float().cpu()
        both_finite = ~(torch.isnan(c_f)|torch.isnan(g_f)|torch.isinf(c_f)|torch.isinf(g_f))
        asym_nan = ((torch.isnan(c_f)&~torch.isnan(g_f)) | (torch.isnan(g_f)&~torch.isnan(c_f))).any()
        if asym_nan:
            print(f'output[{i}]: asymmetric NaN  cpu_nan={torch.isnan(c_f).any()}  gpu_nan={torch.isnan(g_f).any()}')
        elif both_finite.any():
            l2 = (c_f[both_finite] - g_f[both_finite]).pow(2).sum().sqrt().item()
            print(f'output[{i}]: L2={l2:.4e}  shape={list(c.shape)}  (threshold=1e-3)')
        else:
            print(f'output[{i}]: all non-finite (symmetric — expected behavior)')
