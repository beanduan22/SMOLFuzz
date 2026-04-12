"""
Bug: torch.nn.GroupNorm on extreme-magnitude float32 inputs.
CPU uses sequential Welford variance, GPU uses parallel reduction.
At extreme scales (1e18+), variance computation can diverge.

Similar to pt_layernorm_extreme: CPU=NaN, GPU=finite (or vice versa).
"""
import torch
import torch.nn as nn
import numpy as np

print("=== PyTorch GroupNorm extreme-magnitude CPU vs GPU divergence ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

torch.manual_seed(42)
rng = np.random.default_rng(42)

def test_groupnorm(scale, C, G, batch_size=4, spatial=8):
    """Test GroupNorm CPU vs GPU for extreme-magnitude inputs."""
    gn = nn.GroupNorm(G, C, dtype=torch.float32)
    gn_gpu = nn.GroupNorm(G, C, dtype=torch.float32).cuda()
    gn_gpu.load_state_dict(gn.state_dict())

    x = torch.randn(batch_size, C, spatial, spatial, dtype=torch.float32) * scale
    with torch.no_grad():
        cpu_out = gn(x)
        gpu_out = gn_gpu(x.cuda()).cpu()

    cpu_nan = torch.isnan(cpu_out).any().item()
    cpu_inf = torch.isinf(cpu_out).any().item()
    gpu_nan = torch.isnan(gpu_out).any().item()
    gpu_inf = torch.isinf(gpu_out).any().item()

    diff = (cpu_out.float() - gpu_out.float()).abs().max().item() if not (cpu_nan or gpu_nan or cpu_inf or gpu_inf) else float('nan')
    return cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff

print("\n--- float32 GroupNorm(C=64, G=8): various scales ---")
for scale in [1e10, 1e15, 1e18, 1e19, 1e20, 1e25]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_groupnorm(scale, C=64, G=8)
        flag = ""
        if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif (cpu_nan or cpu_inf) and (not gpu_nan and not gpu_inf):
            flag = "*** BUG: CPU=NaN/Inf, GPU=finite ***"
        elif not (cpu_nan or gpu_nan) and diff == diff and diff > 1e-3:
            flag = f"*** DIVERGENCE: {diff:.3e} ***"
        print(f"scale={scale:.0e}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"scale={scale:.0e}: ERROR: {e}")

# Test with different group sizes
print("\n--- GroupNorm with 1 group (equiv to InstanceNorm) ---")
for scale in [1e18, 1e19, 1e20, 1e25]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_groupnorm(scale, C=32, G=1)
        flag = ""
        if (cpu_nan or cpu_inf) != (gpu_nan or gpu_inf):
            flag = f"*** BUG: CPU(nan={cpu_nan},inf={cpu_inf}) vs GPU(nan={gpu_nan},inf={gpu_inf}) ***"
        print(f"G=1 scale={scale:.0e}: CPU(nan={cpu_nan}) GPU(nan={gpu_nan}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"G=1 scale={scale:.0e}: ERROR: {e}")

# Wider channels
print("\n--- Large channel GroupNorm (scale=1e19) ---")
for C, G in [(32, 4), (64, 8), (128, 16), (256, 32), (512, 64)]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_groupnorm(1e19, C=C, G=G)
        flag = ""
        if (cpu_nan or cpu_inf) and not (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=NaN/Inf, GPU=finite ***"
        elif (gpu_nan or gpu_inf) and not (cpu_nan or cpu_inf):
            flag = "*** BUG: GPU=NaN/Inf, CPU=finite ***"
        print(f"C={C},G={G}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"C={C},G={G}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
scale = 1e20
try:
    cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_groupnorm(scale, C=64, G=8)
    print(f"GroupNorm(C=64, G=8), float32, scale={scale:.0e}:")
    print(f"CPU: nan={cpu_nan}, inf={cpu_inf}")
    print(f"GPU: nan={gpu_nan}, inf={gpu_inf}")
    if (cpu_nan or cpu_inf) and not (gpu_nan or gpu_inf):
        print("*** BUG CONFIRMED: CPU=NaN/Inf, GPU=finite ***")
    elif (gpu_nan or gpu_inf) and not (cpu_nan or cpu_inf):
        print("*** BUG CONFIRMED: GPU=NaN/Inf, CPU=finite ***")
    elif diff > 1e-3:
        print(f"*** SIGNIFICANT DIVERGENCE: {diff:.3e} ***")
    else:
        print("No bug at this scale")
except Exception as e:
    print(f"ERROR: {e}")
