"""
Bug: torch.nn.LayerNorm on extreme-magnitude float32 inputs — GPU returns inf
where CPU returns finite (or vice versa).

Source: Related to PyTorch issue #173885 (LayerNorm Inductor CPU float16 Inf→NaN)
Extended to test: float32 large inputs, float16 inputs, CPU vs GPU divergence.

LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
GPU: uses fused CUDA kernel with parallel reduction
CPU: uses sequential Welford algorithm

For very large inputs (scale ≈ 1e19+), the variance computation can overflow
on GPU (x^2 approach) while Welford stays stable on CPU.
"""
import torch
import torch.nn as nn
import numpy as np

print("=== PyTorch LayerNorm extreme-magnitude CPU vs GPU divergence ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA")
    exit(0)

torch.manual_seed(42)

def test_layernorm(scale, normalized_shape, dtype, batch_size=4):
    """Test LayerNorm CPU vs GPU for extreme-magnitude inputs."""
    ln = nn.LayerNorm(normalized_shape, dtype=dtype if dtype != torch.float16 else torch.float32)
    if dtype == torch.float16:
        ln = ln.half()
    ln_gpu = nn.LayerNorm(normalized_shape, dtype=dtype if dtype != torch.float16 else torch.float32).cuda()
    if dtype == torch.float16:
        ln_gpu = ln_gpu.half().cuda()
    ln_gpu.load_state_dict(ln.state_dict())

    x = torch.randn(batch_size, normalized_shape, dtype=dtype) * scale
    with torch.no_grad():
        cpu_out = ln(x)
        gpu_out = ln_gpu(x.cuda()).cpu()

    cpu_nan = torch.isnan(cpu_out).any().item()
    cpu_inf = torch.isinf(cpu_out).any().item()
    gpu_nan = torch.isnan(gpu_out).any().item()
    gpu_inf = torch.isinf(gpu_out).any().item()

    diff = (cpu_out.float() - gpu_out.float()).abs().max().item() if not (cpu_nan or gpu_nan or cpu_inf or gpu_inf) else float('nan')
    return cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff

print("\n--- float32 LayerNorm: various scales ---")
for scale in [1e10, 1e15, 1e18, 1e19, 1e20]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_layernorm(scale, 768, torch.float32)
        flag = ""
        if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif (cpu_nan or cpu_inf) and (not gpu_nan and not gpu_inf):
            flag = "*** BUG: CPU=NaN/Inf, GPU=finite ***"
        elif not (cpu_nan or gpu_nan) and diff > 1e-3:
            flag = f"*** DIVERGENCE: {diff:.3e} ***"
        print(f"scale={scale:.0e}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"scale={scale:.0e}: ERROR: {e}")

# float16 LayerNorm
print("\n--- float16 LayerNorm: extreme-magnitude inputs ---")
for scale in [1e2, 5e2, 1e3, 2e3, 5e3]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_layernorm(scale, 256, torch.float16)
        flag = ""
        if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif (cpu_nan or cpu_inf) and (not gpu_nan and not gpu_inf):
            flag = "*** BUG: CPU=NaN/Inf, GPU=finite ***"
        print(f"f16 scale={scale:.0e}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"f16 scale={scale:.0e}: ERROR: {e}")

# Large hidden size
print("\n--- Large normalized_shape (scale=1e19) ---")
for hidden in [256, 512, 1024, 2048, 4096]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_layernorm(1e19, hidden, torch.float32)
        flag = ""
        if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif (cpu_nan or cpu_inf) and (not gpu_nan and not gpu_inf):
            flag = "*** BUG ***"
        elif not (cpu_nan or gpu_nan) and diff > 0.1:
            flag = f"*** DIVERGENCE: {diff:.3e} ***"
        print(f"hidden={hidden}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"hidden={hidden}: ERROR: {e}")

# bfloat16 LayerNorm
print("\n--- bfloat16 LayerNorm ---")
for scale in [1e2, 5e3, 1e4]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_layernorm(scale, 256, torch.bfloat16)
        flag = ""
        if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif not (cpu_nan or gpu_nan) and diff > 1.0:
            flag = f"*** DIVERGENCE: {diff:.3e} ***"
        print(f"bf16 scale={scale:.0e}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) diff={diff:.3e} {flag}")
    except Exception as e:
        print(f"bf16 scale={scale:.0e}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
# Best reproducer
scale = 1e19
cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_layernorm(scale, 768, torch.float32)
print(f"LayerNorm(768), float32, scale={scale:.0e}:")
print(f"CPU: nan={cpu_nan}, inf={cpu_inf}")
print(f"GPU: nan={gpu_nan}, inf={gpu_inf}")
if diff == diff:
    print(f"Max diff: {diff:.3e}")
if (not cpu_nan and not cpu_inf) and (gpu_nan or gpu_inf):
    print("*** BUG CONFIRMED: CPU=finite, GPU=NaN/Inf ***")
elif (cpu_nan or cpu_inf) and (not gpu_nan and not gpu_inf):
    print("*** BUG CONFIRMED: CPU=NaN/Inf, GPU=finite ***")
elif diff > 1e-3:
    print(f"*** SIGNIFICANT DIVERGENCE: {diff:.3e} ***")
else:
    print("No bug at this scale — try larger scale")
