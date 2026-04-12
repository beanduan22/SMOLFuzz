"""
Bug: torch.Tensor.scatter_add_ with duplicate indices produces different values
on CPU vs GPU due to GPU atomic addition order being non-deterministic.

CPU: serial summation, deterministic, correct
GPU: parallel atomicAdd in arbitrary order → floating-point non-associativity
     causes different sums, sometimes 100x+ worse than CPU

Source: Known GPU non-determinism issue; cross-framework test vs TF scatter.
For float16 or bfloat16, the non-associativity is exacerbated.
"""
import torch
import numpy as np

print("=== PyTorch scatter_add_ CPU vs GPU non-determinism ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA available")
    exit(0)

torch.manual_seed(42)

# Test 1: float32 scatter_add with many duplicates
print("\n--- Test 1: float32 scatter_add with heavy duplicate indices ---")
for N, M in [(1_000_000, 100), (500_000, 10), (2_000_000, 1000)]:
    src = torch.randn(N, dtype=torch.float32)
    idx = torch.randint(0, M, (N,), dtype=torch.long)

    out_cpu = torch.zeros(M, dtype=torch.float32)
    out_cpu.scatter_add_(0, idx, src)

    out_gpu = torch.zeros(M, dtype=torch.float32, device='cuda')
    out_gpu.scatter_add_(0, idx.cuda(), src.cuda())
    out_gpu = out_gpu.cpu()

    # Reference: float64
    out_ref = torch.zeros(M, dtype=torch.float64)
    out_ref.scatter_add_(0, idx, src.double())

    diff_cpu = (out_cpu.double() - out_ref).abs().max().item()
    diff_gpu = (out_gpu.double() - out_ref).abs().max().item()
    cpu_gpu_diff = (out_cpu - out_gpu).abs().max().item()
    ratio = diff_gpu / (diff_cpu + 1e-30)

    print(f"N={N}, M={M}: CPU_err={diff_cpu:.3e}, GPU_err={diff_gpu:.3e}, "
          f"CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")
    if cpu_gpu_diff > 1e-3:
        print(f"  *** SIGNIFICANT CPU/GPU DIVERGENCE ***")

# Test 2: float16 scatter_add — worse due to lower precision
print("\n--- Test 2: float16 scatter_add ---")
for N, M in [(100_000, 50), (500_000, 100)]:
    src = torch.randn(N, dtype=torch.float16) * 100
    idx = torch.randint(0, M, (N,), dtype=torch.long)

    out_cpu = torch.zeros(M, dtype=torch.float16)
    out_cpu.scatter_add_(0, idx, src)

    out_gpu = torch.zeros(M, dtype=torch.float16, device='cuda')
    out_gpu.scatter_add_(0, idx.cuda(), src.cuda())
    out_gpu = out_gpu.cpu()

    out_ref = torch.zeros(M, dtype=torch.float32)
    out_ref.scatter_add_(0, idx, src.float())

    diff_cpu = (out_cpu.float() - out_ref).abs().max().item()
    diff_gpu = (out_gpu.float() - out_ref).abs().max().item()
    cpu_gpu_diff = (out_cpu.float() - out_gpu.float()).abs().max().item()
    ratio = diff_gpu / (diff_cpu + 1e-30)

    print(f"N={N}, M={M}: CPU_err={diff_cpu:.3e}, GPU_err={diff_gpu:.3e}, "
          f"CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")
    if cpu_gpu_diff > 0.1:
        print(f"  *** SIGNIFICANT CPU/GPU DIVERGENCE ***")

# Test 3: bfloat16 — most affected by non-associativity
print("\n--- Test 3: bfloat16 scatter_add ---")
N, M = 200_000, 50
src = torch.randn(N, dtype=torch.bfloat16) * 1000
idx = torch.randint(0, M, (N,), dtype=torch.long)

out_cpu = torch.zeros(M, dtype=torch.bfloat16)
out_cpu.scatter_add_(0, idx, src)
out_gpu = torch.zeros(M, dtype=torch.bfloat16, device='cuda')
out_gpu.scatter_add_(0, idx.cuda(), src.cuda())
out_gpu = out_gpu.cpu()
out_ref = torch.zeros(M, dtype=torch.float32)
out_ref.scatter_add_(0, idx, src.float())

diff_cpu = (out_cpu.float() - out_ref).abs().max().item()
diff_gpu = (out_gpu.float() - out_ref).abs().max().item()
cpu_gpu_diff = (out_cpu.float() - out_gpu.float()).abs().max().item()
ratio = diff_gpu / (diff_cpu + 1e-30)

print(f"N={N}, M={M}: CPU_err={diff_cpu:.3e}, GPU_err={diff_gpu:.3e}, "
      f"CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")

print(f"\n=== BUG SUMMARY ===")
# Run determinism test: same GPU call twice
torch.manual_seed(42)
N, M = 1_000_000, 100
src = torch.randn(N, dtype=torch.float32)
idx = torch.randint(0, M, (N,), dtype=torch.long)

out1 = torch.zeros(M, device='cuda')
out1.scatter_add_(0, idx.cuda(), src.cuda())
out2 = torch.zeros(M, device='cuda')
out2.scatter_add_(0, idx.cuda(), src.cuda())

gpu_deterministic = torch.allclose(out1, out2)
out_cpu = torch.zeros(M)
out_cpu.scatter_add_(0, idx, src)
cpu_gpu_match = torch.allclose(out_cpu, out1.cpu())

print(f"N={N}, M={M}, float32 with {N//M:.0f} duplicates per bucket:")
print(f"GPU deterministic (run1 == run2): {gpu_deterministic}")
print(f"CPU == GPU: {cpu_gpu_match}")
if not cpu_gpu_match:
    diff = (out_cpu - out1.cpu()).abs().max().item()
    print(f"Max CPU vs GPU diff: {diff:.3e}")
