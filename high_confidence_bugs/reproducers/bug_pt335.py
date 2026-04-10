#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 335
Bug Type: inconsistent
Detail  : output[0]: l2=2.9577e-01 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.Dropout, torch.nn.functional.pad
Mutation: add_noise

Run: python3 bug_pt335.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.transforms as transforms

# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(4, 8)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = F.pad(x, (0, 8))[:, :8]  
            x = self.fc1(x)
            x = self.bn(x)
            self.train()
            x = self.drop(x)
            self.eval()
            x = self.bn(x)
            x = torch.sin(x)
            u, indices = x.topk(k=4, dim=1)
            v = u.clone().lgamma_()
            v = v.repeat_interleave(4, dim=1)  # Changed this line
            transform = transforms.SigmoidTransform()
            z = torch.less_equal(x, torch.max(x, dim=1).values.unsqueeze(1).repeat_interleave(16, dim=1)).float()
            w = transform(z)
            x = x + w
            x = u  # Take topk values for linear layer input
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

    # Embedded inputs (mutation: add_noise, already applied)
    x = torch.tensor([[0.9296867251396179, 2.208510637283325, 0.6901650428771973, -0.998125433921814, 1.8454164266586304, -0.4518145024776459, 0.0746079534292221, 1.7498817443847656], [-0.4104006886482239, 0.09852416813373566, 1.2658077478408813, 1.9157612323760986, -0.7603945732116699, -1.8289883136749268, -0.4822019040584564, -0.3121579587459564], [1.3941164016723633, 0.8707928657531738, -1.2530170679092407, 2.7377307415008545, -1.0357056856155396, 0.44822603464126587, -0.27351686358451843, -1.3920104503631592], [1.3150577545166016, -0.40529248118400574, 0.4912153482437134, -0.6456960439682007, 0.17139215767383575, -1.0919092893600464, 0.01453128270804882, 1.7698695659637451]], dtype=torch.float32)
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
