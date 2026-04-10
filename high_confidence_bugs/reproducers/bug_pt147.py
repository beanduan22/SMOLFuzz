#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 147
Bug Type: inconsistent
Root Cause: BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence
Detail  : output[0]: l2=8.3500e+01 > threshold=1e-03 finite_elements=8 shape=[1, 8]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.sigmoid, torch.Tensor.expm1, torch.Tensor.polygamma_
Mutation: mask

Run: python3 bug_pt147.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.bn(x)
            x = torch.sigmoid(x)
            x = self.fc2(x)
            x = x.expm1()
            x = x.polygamma_(1)
            x = x.std(dim=0, keepdim=True)
        return x

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
    inp0 = torch.tensor([[1.2801445722579956, 2.084859848022461, 0.3327067494392395, -0.4003010094165802, 0.0, 0.16465674340724945, 0.5738624930381775, 0.0060699633322656155], [-0.514781653881073, -1.83219575881958, 1.3614412546157837, 1.6753082275390625, 0.0, 0.0, 0.48791050910949707, 0.7413539886474609], [0.669989824295044, 0.0, 0.0, 1.5798181295394897, 0.5415620803833008, 0.0, 0.0, -0.21595177054405212], [-0.8319048881530762, -0.11799944937229156, 1.7658331394195557, -1.221530556678772, 0.0, 0.0, 0.09160400182008743, 0.0]], dtype=torch.float32)
    inputs = [inp0]

    torch.manual_seed(42)
    cpu_model = Model()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gpu_model = Model().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    # NOTE: keep training mode — BatchNorm uses batch statistics (source of the bug)

    
    cpu_out = cpu_model(*inputs)
    gpu_out = gpu_model(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs])
    

    def to_list(o):
        if isinstance(o, torch.Tensor):
            return [o.detach()]
        return [x.detach() for x in (o if isinstance(o, (list, tuple)) else [])
                if isinstance(x, torch.Tensor)]

    found = False
    for i, (c, g) in enumerate(zip(to_list(cpu_out), to_list(gpu_out))):
        g = g.cpu()
        if c.shape != g.shape:
            print(f"output[{i}]: shape mismatch cpu={c.shape} gpu={g.shape}")
            found = True
            continue
        cf, gf = c.float(), g.float()
        asym = (torch.isnan(cf) & ~torch.isnan(gf)) | (torch.isnan(gf) & ~torch.isnan(cf))
        asinf = (torch.isinf(cf) & ~torch.isinf(gf)) | (torch.isinf(gf) & ~torch.isinf(cf))
        nan_inf = (torch.isnan(cf) & torch.isinf(gf)) | (torch.isinf(cf) & torch.isnan(gf))
        if (asym | asinf | nan_inf).any():
            n = int((asym | asinf | nan_inf).sum().item())
            print(f"output[{i}]: ASYMMETRIC NaN/Inf — "
                  f"cpu_nan={torch.isnan(cf).sum().item()} "
                  f"gpu_nan={torch.isnan(gf).sum().item()} asym={n}")
            found = True
        else:
            fin = ~(torch.isnan(cf) | torch.isinf(cf) | torch.isnan(gf) | torch.isinf(gf))
            if fin.any():
                l2 = float((cf[fin] - gf[fin]).pow(2).sum().sqrt().item())
                if l2 > 1e-3:
                    print(f"output[{i}]: L2={l2:.4e}  shape={list(c.shape)}")
                    found = True

    if found:
        print("BUG CONFIRMED: CPU and GPU produce different results for the same model and input")
    else:
        print("Inconsistency not reproduced on this hardware/driver version")


if __name__ == "__main__":
    run()
