"""
Bug: torch.nn.InstanceNorm2d on extreme-magnitude float32 inputs.
CPU uses sequential Welford variance, GPU uses parallel reduction.
At extreme scales (1e18+), CPU=NaN while GPU=finite.

InstanceNorm normalizes each feature map individually (no shared stats).
This is related to pt_layernorm_extreme and pt_group_norm_extreme.
"""
import torch
import torch.nn as nn
import numpy as np

print("=== PyTorch InstanceNorm extreme-magnitude CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

torch.manual_seed(42)

def test_instancenorm(scale, C, H, W, batch_size=4):
    inst = nn.InstanceNorm2d(C, affine=True, dtype=torch.float32)
    inst_gpu = nn.InstanceNorm2d(C, affine=True, dtype=torch.float32).cuda()
    inst_gpu.load_state_dict(inst.state_dict())

    x = torch.randn(batch_size, C, H, W, dtype=torch.float32) * scale
    with torch.no_grad():
        cpu_out = inst(x)
        gpu_out = inst_gpu(x.cuda()).cpu()

    cpu_nan = torch.isnan(cpu_out).any().item()
    cpu_inf = torch.isinf(cpu_out).any().item()
    gpu_nan = torch.isnan(gpu_out).any().item()
    gpu_inf = torch.isinf(gpu_out).any().item()
    diff = (cpu_out.float() - gpu_out.float()).abs().max().item() if not (cpu_nan or gpu_nan or cpu_inf or gpu_inf) else float('nan')
    return cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff

print("\n--- float32 InstanceNorm2d(C=64): various scales ---")
for scale in [1e10, 1e15, 1e18, 1e19, 1e20, 1e25]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_instancenorm(scale, C=64, H=8, W=8)
        flag = ""
        if (cpu_nan or cpu_inf) and not (gpu_nan or gpu_inf):
            flag = "*** BUG: CPU=NaN/Inf, GPU=finite ***"
        elif (gpu_nan or gpu_inf) and not (cpu_nan or cpu_inf):
            flag = "*** BUG: CPU=finite, GPU=NaN/Inf ***"
        elif not (cpu_nan or gpu_nan) and diff == diff and diff > 1e-3:
            flag = f"*** DIVERGENCE: {diff:.3e} ***"
        print(f"scale={scale:.0e}: CPU(nan={cpu_nan},inf={cpu_inf}) GPU(nan={gpu_nan},inf={gpu_inf}) {flag}")
    except Exception as e:
        print(f"scale={scale:.0e}: ERROR: {e}")

# Different spatial sizes
print("\n--- InstanceNorm2d: various spatial sizes (scale=1e20) ---")
for C, H in [(16, 4), (32, 8), (64, 16), (128, 32)]:
    try:
        cpu_nan, cpu_inf, gpu_nan, gpu_inf, diff = test_instancenorm(1e20, C=C, H=H, W=H)
        flag = ""
        if (cpu_nan or cpu_inf) != (gpu_nan or gpu_inf):
            flag = f"*** BUG ***"
        print(f"C={C},H={H}: CPU(nan={cpu_nan}) GPU(nan={gpu_nan}) {flag}")
    except Exception as e:
        print(f"C={C},H={H}: ERROR: {e}")

# 1D and 3D variants
print("\n--- InstanceNorm1d extreme scale ---")
for scale in [1e18, 1e19, 1e20]:
    try:
        inst1d = nn.InstanceNorm1d(64, affine=True)
        inst1d_gpu = nn.InstanceNorm1d(64, affine=True).cuda()
        inst1d_gpu.load_state_dict(inst1d.state_dict())
        x = torch.randn(4, 64, 128) * scale
        with torch.no_grad():
            cpu_out = inst1d(x)
            gpu_out = inst1d_gpu(x.cuda()).cpu()
        cpu_nan = torch.isnan(cpu_out).any().item()
        gpu_nan = torch.isnan(gpu_out).any().item()
        flag = "*** BUG ***" if cpu_nan != gpu_nan else ""
        print(f"1D scale={scale:.0e}: CPU_nan={cpu_nan}, GPU_nan={gpu_nan} {flag}")
    except Exception as e:
        print(f"1D scale={scale:.0e}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
scale = 1e20
import numpy as _np
_rng2 = _np.random.default_rng(42)
N, C, H, W = 8, 32, 32, 32
x_np2 = (_rng2.standard_normal((N, C, H, W)) * scale).astype(_np.float32)
x_cpu2 = torch.tensor(x_np2)
x_gpu2 = x_cpu2.cuda()
out_cpu2 = torch.nn.functional.instance_norm(x_cpu2)
out_gpu2 = torch.nn.functional.instance_norm(x_gpu2).cpu()
cpu_nan2 = out_cpu2.isnan().any().item()
gpu_nan2 = out_gpu2.isnan().any().item()
print(f"instance_norm(float32, scale={scale:.0e}, shape={list(x_cpu2.shape)}):")
print(f"CPU: nan={cpu_nan2}")
print(f"GPU: nan={gpu_nan2}")
if cpu_nan2 != gpu_nan2:
    print(f"*** BUG CONFIRMED: CPU produces nan={cpu_nan2} but GPU produces nan={gpu_nan2} ***")
else:
    print("No significant divergence found")
