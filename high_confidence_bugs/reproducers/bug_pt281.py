#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 281
Bug Type: inconsistent
Detail  : output[0]: l2=2.3222e+00 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.sin, torch.nn.functional.multilabel_soft_margin_loss, torch.mul
Mutation: mask

Run: python3 bug_pt281.py
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
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = torch.sin(x)
            x = self.fc2(x)
            x = self.bn(x)
            loss = torch.nn.functional.multilabel_soft_margin_loss(x, torch.ones_like(x))
        with torch.no_grad():
            x = torch.mul(loss.unsqueeze(-1), x)  # Adjusted to maintain shape consistency
            x = self.fc2(x)
        x = torch.Tensor.ne_(x, torch.zeros_like(x)).float()
        x = torch.nn.init.uniform_(x, -1.0, 1.0)
        x = torch.Tensor.addcmul(torch.ones_like(x), x, x)
        x = torch.Tensor.fill_diagonal_(x, 5.0)
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
    x = torch.tensor([[1.1116291284561157, 0.0, 0.0, 0.0, 0.0, 0.24364304542541504, -1.1565853357315063, 0.0], [-0.6280121207237244, 0.0, 0.0, -0.45204347372055054, 0.0, 0.9478051066398621, -0.074477918446064, 1.1162607669830322], [0.0, -0.24085357785224915, 0.0, 1.166702151298523, -1.4397724866867065, 0.0, 2.227046251296997, 0.0], [-1.4434592723846436, -0.18051756918430328, 0.0, 0.9693029522895813, 0.8661069273948669, 0.0, 1.916176676750183, 0.0]], dtype=torch.float32)
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
