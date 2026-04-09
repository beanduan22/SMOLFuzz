#!/usr/bin/env python3
"""
SMOLFuzz bug reproducer  —  INCONSISTENT
Model : 202
Mutation: scale_large
Detail  : output[0]: l2=1.7321e+00 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.special.log_softmax, torch.sin, torch.cos, torch.enable_grad, torch.floor, torch.cumsum

Run:  python3 bug_inconsistent_m0202_mut5_20260409_225842.repro.py
Requires: PyTorch with CUDA available.
"""
import torch
import torch.nn as nn
from pathlib import Path

# ── Model definition (LLM-generated) ──────────────────────────
import torch.special

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = torch.special.log_softmax(y, dim=1)
            w = torch.sin(z) * torch.cos(z)
            v = torch.floor(w)
            u = torch.cumsum(v, 1)
        return u

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.special.log_softmax", 
             "torch.sin", "torch.cos", "torch.enable_grad", 
             "torch.floor", "torch.cumsum"]

# Test the model
model = Model()
inputs = make_inputs()
output = model(*inputs)
print("Output shape:", output.shape)

# Test tensor methods
tensor = torch.randn(4, 8)
print("Tensor grad:", tensor.grad)
print("Tensor nelement:", tensor.nelement())
print("Tensor lt:", tensor.lt(tensor))
print("Unbind tensor:", list(torch.unbind(tensor)))
print("Ravel tensor:", tensor.ravel())

# Test complex double storage and char storage
complex_storage = torch.ComplexDoubleStorage()
char_storage = torch.CharStorage()

# Test result type
result_type = torch.result_type(tensor, tensor)

# Test mode
mode = torch.mode(tensor)[0]

# Set default dtype and tensor type
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

# Test is set to
is_set_to = tensor.is_set_to(tensor)

# Test frexp
frexp = tensor.frexp()

# Test linalg matrix rank
matrix_rank = torch.linalg.matrix_rank(tensor)

print("USED_APIS:", USED_APIS)

# ── Load buggy inputs ─────────────────────────────────────────
_here = Path(__file__).parent
inputs = torch.load(_here / 'bug_inconsistent_m0202_mut5_20260409_225842.inputs.pt', weights_only=False)

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
