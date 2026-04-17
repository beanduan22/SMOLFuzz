#!/usr/bin/env python3
"""
SMOLFuzz bug reproducer — INCONSISTENT
Model : 46
Mutation: scale_large
Detail  : output[0]: rel_err=1.631e+02 > 1.0 dtype=torch.float32 shape=[4, 4] rtol=1e-04 atol=1e-05 nondet_floor=0.000e+00
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.Mish, torch.nn.functional.hardswish, torch.nn.functional.threshold_, torch.fmod, torch.exp, torch.sin, torch.enable_grad

Run:  python3 bug_inconsistent_m0046_mut5_20260416_185555.repro.py
Requires: PyTorch with CUDA available.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from pathlib import Path

# Determinism settings (match fuzzer runner).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass

# ── Model (LLM-generated) ────────────────────────────
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.mish = nn.Mish()
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.fc1(x)
        with torch.enable_grad():
            x.requires_grad_(True)
            x = self.bn(x)
            x = self.mish(x)
            y = torch.nn.functional.hardswish(x)
            z = torch.nn.functional.threshold_(y.clone(), 0.5, 0.0)
            w = torch.fmod(z, 2.0)
            v = torch.exp(w)
            u = torch.sin(v)
        self.train()
        x = self.fc2(u)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.nn.BatchNorm1d", "torch.nn.Mish",
             "torch.nn.functional.hardswish", "torch.nn.functional.threshold_",
             "torch.fmod", "torch.exp", "torch.sin", "torch.enable_grad"]

# ── Load buggy inputs ─────────────────────────────
_here = Path(__file__).parent
inputs = torch.load(_here / 'bug_inconsistent_m0046_mut5_20260416_185555.inputs.pt', weights_only=False)

# ── Build two models with identical weights ───────
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
cpu_model = Model().cpu().eval()
gpu_model = Model().cuda().eval()
gpu_model.load_state_dict(cpu_model.state_dict())

cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
with torch.set_grad_enabled(True):
    cpu_out = cpu_model(*cpu_inputs)
    gpu_out = gpu_model(*gpu_inputs)
cpu_outs = [cpu_out] if isinstance(cpu_out, torch.Tensor) else list(cpu_out)
gpu_outs = [gpu_out] if isinstance(gpu_out, torch.Tensor) else list(gpu_out)

TOL = {torch.float64:(1e-5,1e-8), torch.float32:(1e-4,1e-5),
       torch.bfloat16:(1e-2,1e-2), torch.float16:(1e-2,1e-3)}
for i,(c,g) in enumerate(zip(cpu_outs, gpu_outs)):
    if isinstance(c, torch.Tensor) and isinstance(g, torch.Tensor):
        rtol, atol = TOL.get(c.dtype, (1e-4, 1e-5))
        c_f, g_f = c.float(), g.float().cpu()
        finite = torch.isfinite(c_f) & torch.isfinite(g_f)
        if finite.any():
            rel = ((c_f[finite]-g_f[finite]).abs() / (atol + rtol * c_f[finite].abs())).max().item()
            print(f'output[{i}] dtype={c.dtype} rel_err={rel:.3e} shape={list(c.shape)}')
        else:
            print(f'output[{i}]: all non-finite')
