"""
Bug: torch.nn.functional.softplus with large beta returns inf instead of correct value.
Source: PyTorch GitHub issue #171249
Cross-framework: Confirms PyTorch bug; compare with TF/JAX behavior.

softplus(x, beta) = log(1 + exp(beta * x)) / beta
For large beta and positive x: beta*x overflows float32, giving inf/beta = inf
Stable formula: x + log1p(exp(-beta*x))/beta
"""
import torch
import torch.nn.functional as F
import numpy as np

print("=== PyTorch softplus large-beta overflow ===")
print(f"PyTorch version: {torch.__version__}")

x = torch.tensor([0.5, 1.0, 2.0, -1.0, 0.0], dtype=torch.float32)

# reference: softplus(x, beta→∞) → relu(x) = max(0, x)
reference = torch.relu(x)
print(f"\nReference (ReLU limit): {reference.tolist()}")

betas = [1e10, 1e20, 1e30]
for beta in betas:
    cpu_out = F.softplus(x, beta=beta, threshold=float('inf'))
    print(f"\nbeta={beta:.0e}:")
    print(f"  CPU: {cpu_out.tolist()}")
    if torch.cuda.is_available():
        gpu_out = F.softplus(x.cuda(), beta=beta, threshold=float('inf')).cpu()
        print(f"  GPU: {gpu_out.tolist()}")
    print(f"  Reference (ReLU): {reference.tolist()}")

# Confirm bug: CPU should return inf for large beta, reference is finite
beta = 1e30
cpu_out = F.softplus(x, beta=beta, threshold=float('inf'))
has_inf_cpu = torch.isinf(cpu_out).any().item()
has_inf_ref = torch.isinf(reference).any().item()

print(f"\n=== BUG SUMMARY ===")
print(f"beta=1e30, x=[0.5, 1.0, 2.0, -1.0, 0.0]")
print(f"CPU output:  {cpu_out.tolist()}")
print(f"Reference:   {reference.tolist()}")
print(f"CPU has inf: {has_inf_cpu}  (BUG: should be False)")

if torch.cuda.is_available():
    gpu_out = F.softplus(x.cuda(), beta=beta, threshold=float('inf')).cpu()
    has_inf_gpu = torch.isinf(gpu_out).any().item()
    print(f"GPU output:  {gpu_out.tolist()}")
    print(f"GPU has inf: {has_inf_gpu}")
    print(f"CPU==GPU: {torch.allclose(cpu_out, gpu_out, equal_nan=True)}")
