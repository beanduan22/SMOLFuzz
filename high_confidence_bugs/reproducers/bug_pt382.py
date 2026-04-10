#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 382
Bug Type: inconsistent
Detail  : output[0]: l2=7.1658e-03 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.nn.functional.dropout2d, torch.sin, torch.cos
Mutation: scale_small

Run: python3 bug_pt382.py
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

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.nn.functional.dropout2d(x[:, None, :, None], p=0.5)[:, 0, :, 0]
            z = self.fc1(y)
            w = self.fc2(z)
            u = torch.sin(w) * torch.cos(w)
        v = u.detach_().cpu()
        return torch.nn.Softmin(dim=-1)(v)

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
    x = torch.tensor([[-0.07450681924819946, -0.0249410942196846, 0.037409357726573944, -0.007298925891518593, -0.04464515671133995, -0.04565056785941124, 0.017812194302678108, -0.024272607639431953], [-0.07131846994161606, 0.08488379418849945, 0.03552514314651489, -0.04301527142524719, -0.04417037591338158, -0.005990826990455389, -0.034650735557079315, -0.027025535702705383], [0.022847089916467667, 0.05152927339076996, 0.016975007951259613, -0.017264962196350098, 0.03705437853932381, -0.01749545708298683, -0.05126110836863518, 0.009082949720323086], [0.013071794994175434, 0.018717288970947266, 0.039267949759960175, 0.002976001938804984, -0.07201249897480011, -0.05630429461598396, -0.044120658189058304, -0.043149434030056]], dtype=torch.float32)
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
