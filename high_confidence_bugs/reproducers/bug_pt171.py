#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 171
Bug Type: inconsistent
Detail  : output[0]: l2=1.1689e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.nn.functional.poisson_nll_loss, torch.Tensor.cumprod, torch.nn.BatchNorm1d
Mutation: mask

Run: python3 bug_pt171.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = F.poisson_nll_loss(x, torch.ones_like(x), reduction='none')
            x = x.cumprod(dim=1)
            x = self.bn(x)
            self.train()
            x = self.drop(x)
            self.eval()
            x = self.bn(x)
            x = self.fc2(x)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

# ── Reproducer ─────────────────────────────────────────────────
def run():
    torch.manual_seed(42)
    cpu_model = Model()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gpu_model = Model().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    cpu_model.eval()
    gpu_model.eval()

    # Embedded inputs (mutation: mask, already applied)
    x = torch.tensor([[0.5893014669418335, 1.4324951171875, -0.01360425166785717, 1.3221226930618286, 0.0, 0.049943193793296814, -0.45243170857429504, 1.269170880317688], [-0.30174121260643005, 0.8906669020652771, 0.4243488013744354, -1.5894263982772827, 1.6798499822616577, 0.0, -1.031320571899414, 2.070213556289673], [2.595740556716919, -0.5057842135429382, -0.27941036224365234, -1.4813755750656128, -0.9861884117126465, 1.3755278587341309, 0.3283268213272095, 0.6493554711341858], [1.4629944562911987, -0.13590571284294128, 0.5994144678115845, -0.6495959758758545, -0.8551895022392273, 0.0, 0.0, -0.3249739408493042]], dtype=torch.float32)
    inputs = [x]

    with torch.no_grad():
        cpu_out = cpu_model(*inputs)
        gpu_out = gpu_model(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs])

    if isinstance(cpu_out, torch.Tensor):
        cpu_outs, gpu_outs = [cpu_out], [gpu_out.cpu()]
    else:
        cpu_outs = [o for o in cpu_out if isinstance(o, torch.Tensor)]
        gpu_outs = [o.cpu() for o in gpu_out if isinstance(o, torch.Tensor)]

    found = False
    for i, (c, g) in enumerate(zip(cpu_outs, gpu_outs)):
        g = g.cpu()
        if c.shape != g.shape:
            print(f"output[{i}]: shape mismatch cpu={c.shape} gpu={g.shape}")
            found = True
            continue
        cf, gf = c.float(), g.float()
        fin = ~(torch.isnan(cf) | torch.isinf(cf) | torch.isnan(gf) | torch.isinf(gf))
        asym = ((torch.isnan(cf) & ~torch.isnan(gf)) | (torch.isnan(gf) & ~torch.isnan(cf)) |
                (torch.isinf(cf) & ~torch.isinf(gf)) | (torch.isinf(gf) & ~torch.isinf(cf)) |
                (torch.isnan(cf) & torch.isinf(gf)) | (torch.isinf(cf) & torch.isnan(gf)))
        if asym.any():
            print(f"output[{i}]: ASYMMETRIC NaN/Inf — cpu_nan={torch.isnan(cf).sum().item()} gpu_nan={torch.isnan(gf).sum().item()} "
                  f"cpu_inf={torch.isinf(cf).sum().item()} gpu_inf={torch.isinf(gf).sum().item()} asym={asym.sum().item()}")
            found = True
        elif fin.any():
            l2 = (cf[fin] - gf[fin]).pow(2).sum().sqrt().item()
            if l2 > 1e-3:
                print(f"output[{i}]: L2={l2:.4e}  shape={list(c.shape)}")
                found = True

    if not found:
        print("No inconsistency detected (may be hardware/driver dependent)")
    else:
        print("BUG CONFIRMED: CPU and GPU produce different results for the same model and input")

if __name__ == "__main__":
    run()
