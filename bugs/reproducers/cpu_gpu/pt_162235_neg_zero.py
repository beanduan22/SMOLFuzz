"""
Bug   : Inconsistent handling of `-0.0` between CPU and CUDA for many operators
Issue : https://github.com/pytorch/pytorch/issues/162235
Class : cpu_vs_gpu  (silent wrong result)
Repro : float CPU preserves the -0.0 sign bit; CUDA normalises to +0.0,
        which also flips argsort / sort order.

Variants exercise the same root cause across:
  - different ops:     maximum, relu, relu6, hardtanh, amin, amax, argsort, sort
  - different dtypes:  float32, float64, float16, bfloat16
  - different shapes:  scalar / 1-D / 2-D / broadcast pair
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(name: str, cpu_out: torch.Tensor, gpu_out: torch.Tensor) -> None:
    cpu = cpu_out.detach().cpu()
    gpu = gpu_out.detach().cpu()
    if cpu.dtype != gpu.dtype:
        print(f"  [{name}] DTYPE MISMATCH cpu={cpu.dtype} gpu={gpu.dtype}")
        return
    if cpu.is_floating_point():
        cpu_b = cpu.to(torch.float64)
        gpu_b = gpu.to(torch.float64)
        diff = (cpu_b - gpu_b).abs()
        sign_diff = (torch.signbit(cpu) != torch.signbit(gpu)).any().item()
        print(f"  [{name}] max|diff|={diff.max().item():.3e}  signbit_disagreement={sign_diff}")
    else:
        eq = torch.equal(cpu, gpu)
        print(f"  [{name}] equal={eq}  cpu={cpu.tolist()}  gpu={gpu.tolist()}")


def variant_1_original_unary_ops() -> None:
    """Issue's own minimal repro — float32, 1-D inputs."""
    print("variant_1_original_unary_ops")
    x_single = torch.tensor([-0.0], dtype=torch.float32)
    x_pair = torch.tensor([-0.0, 0.0], dtype=torch.float32)
    g = "cuda"
    _check("maximum",
           torch.maximum(x_single, torch.tensor([0.0])),
           torch.maximum(x_single.to(g), torch.tensor([0.0], device=g)))
    _check("relu", F.relu(x_single), F.relu(x_single.to(g)))
    _check("relu6", F.relu6(x_single), F.relu6(x_single.to(g)))
    _check("hardtanh",
           F.hardtanh(x_single, 0.0, 1.0),
           F.hardtanh(x_single.to(g), 0.0, 1.0))
    _check("argsort", torch.argsort(x_pair), torch.argsort(x_pair.to(g)))
    cpu_v, cpu_i = torch.sort(x_pair)
    gpu_v, gpu_i = torch.sort(x_pair.to(g))
    _check("sort.values", cpu_v, gpu_v)
    _check("sort.indices", cpu_i, gpu_i)
    _check("amin", torch.amin(x_pair, dim=0), torch.amin(x_pair.to(g), dim=0))
    _check("amax", torch.amax(x_pair, dim=0), torch.amax(x_pair.to(g), dim=0))


def variant_2_dtype_sweep() -> None:
    """Same -0.0 input, different floating dtypes."""
    print("variant_2_dtype_sweep")
    for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        if dtype is torch.bfloat16 and not torch.cuda.is_available():
            continue
        x = torch.tensor([-0.0], dtype=dtype)
        ref = torch.tensor([0.0], dtype=dtype)
        cpu_out = torch.maximum(x, ref)
        gpu_out = torch.maximum(x.cuda(), ref.cuda())
        _check(f"maximum/{dtype}", cpu_out, gpu_out)


def variant_3_shape_sweep() -> None:
    """Promote to higher-rank tensors and broadcast."""
    print("variant_3_shape_sweep")
    shapes = [(8,), (4, 4), (2, 3, 5), (1, 1, 16, 16)]
    for shp in shapes:
        x = torch.full(shp, -0.0, dtype=torch.float32)
        cpu = torch.amax(x.flatten(), dim=0)
        gpu = torch.amax(x.cuda().flatten(), dim=0)
        _check(f"amax/shape={shp}", cpu, gpu)


def variant_4_argsort_pair_grids() -> None:
    """Pair grids exercising argsort + sort permutation flip."""
    print("variant_4_argsort_pair_grids")
    rows = torch.tensor([[-0.0, 0.0],
                         [0.0, -0.0],
                         [-0.0, -0.0, 0.0]], dtype=torch.float32) if False else None
    cases = [
        torch.tensor([-0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, -0.0], dtype=torch.float32),
        torch.tensor([-0.0, 0.0, -0.0, 0.0], dtype=torch.float32),
        torch.tensor([[-0.0, 0.0],
                      [0.0, -0.0]], dtype=torch.float32),
    ]
    for i, x in enumerate(cases):
        cpu = torch.argsort(x, dim=-1)
        gpu = torch.argsort(x.cuda(), dim=-1)
        _check(f"argsort/case{i}", cpu, gpu)


def main() -> None:
    if not torch.cuda.is_available():
        print("[skip] CUDA not available; this bug requires a CUDA device")
        return
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
    variant_1_original_unary_ops()
    variant_2_dtype_sweep()
    variant_3_shape_sweep()
    variant_4_argsort_pair_grids()


if __name__ == "__main__":
    main()
