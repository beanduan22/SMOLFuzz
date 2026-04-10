#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 409
Bug Type: inconsistent
Detail  : output[0]: l2=1.7038e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.expm1, torch.sin, torch.nn.BatchNorm1d
Mutation: scale_small

Run: python3 bug_pt409.py
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
            x = torch.expm1(x)
            x = torch.sin(x)
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

    # Embedded inputs (mutation: scale_small, already applied)
    x = torch.tensor([[-0.04334306716918945, 0.0780133455991745, -0.1689128875732422, 0.0715419203042984, 0.04317104071378708, -0.05295079946517944, 0.0921449214220047, 0.02798382192850113], [-0.09291644394397736, 0.06660287827253342, 0.036893781274557114, -0.10693350434303284, -0.10811633616685867, 0.05611750856041908, -0.07805923372507095, 0.05734654515981674], [-0.027038728818297386, -0.018592622131109238, 0.02806687355041504, 0.036147356033325195, -0.11488638073205948, 0.01537808496505022, -0.07378794252872467, 0.035230278968811035], [-0.051428187638521194, 0.04042334854602814, -0.06358738243579865, -0.12130557000637054, -0.028172988444566727, 0.046928416937589645, -0.004977411590516567, -0.05934349447488785]], dtype=torch.float32)
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
