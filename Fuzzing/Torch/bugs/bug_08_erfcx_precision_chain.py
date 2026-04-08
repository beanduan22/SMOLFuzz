"""
Bug #8 — INCONSISTENT: precision chain sqrt → erfcx → LogSigmoid at baseline
Cluster: 2 models (31, 103)
Representative: model_0031
Trigger: baseline (no mutation — the architecture itself is numerically sensitive)

Root cause: torch.special.erfcx is a scaled complementary error function:
erfcx(x) = e^(x²) * erfc(x). It is computed differently on CPU (using Cephes
library) vs GPU (using CUDA math intrinsics). Each step introduces a ULP-level
error, and the LogSigmoid at the end converts very large/small erfcx values
to log-probabilities, squashing errors in a non-uniform way. Divergence is
visible even at standard (unscaled) inputs.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
bn  = nn.BatchNorm1d(8)
fc2 = nn.Linear(8, 8)
logsigmoid = nn.LogSigmoid()

def forward(x):
    x = fc1(x)
    x = torch.nn.functional.hardshrink(x)
    x = bn(x)
    x = fc2(x)
    x = torch.sqrt(x.abs())          # ensure non-negative for sqrt
    x = torch.special.erfcx(x)       # scaled complementary erf — CPU/GPU differ
    return logsigmoid(x)

fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
bn_g  = nn.BatchNorm1d(8).cuda(); bn_g.load_state_dict(bn.state_dict())
fc2_g = nn.Linear(8, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
ls_g  = nn.LogSigmoid().cuda()

def forward_gpu(x):
    x = fc1_g(x)
    x = torch.nn.functional.hardshrink(x)
    x = bn_g(x)
    x = fc2_g(x)
    x = torch.sqrt(x.abs())
    x = torch.special.erfcx(x)
    return ls_g(x)

x = torch.randn(4, 8)
cpu_out = forward(x)
gpu_out = forward_gpu(x.cuda()).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")

# Minimal 2-line version:
print("\nMinimal:")
v = torch.rand(4, 8) * 3          # erfcx is most sensitive in [0, 3]
print(f"erfcx cpu/gpu diff: {(torch.special.erfcx(v) - torch.special.erfcx(v.cuda()).cpu()).abs().max():.4e}")
