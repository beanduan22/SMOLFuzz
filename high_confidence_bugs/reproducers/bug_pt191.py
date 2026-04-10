#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 191
Bug Type: inconsistent
Root Cause: BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence
Detail  : output[0]: l2=1.5129e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.sin, torch.arccos, torch.clamp
Mutation: add_noise

Run: python3 bug_pt191.py
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
        self.fc1 = nn.Linear(8, 4)
        self.bn = nn.BatchNorm1d(4)
        self.fc2 = nn.Linear(4, 4)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        # Gradient-tracking scope changes semantics of subsequent ops
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.fc1(x)
            z = self.bn(y)
            w = torch.sin(z)
            v = self.fc2(w)
            u = self.drop(v)
            t = torch.arccos(torch.clamp(u, -0.999, 0.999))
        
        # Mode switch affects BatchNorm and Dropout
        self.train()
        a = self.bn(t)
        self.eval()
        b = self.fc2(a)
        
        # More operations with gradient tracking
        c = torch.acosh(torch.ones_like(b) + 1e-6)
        d, _ = b.kthvalue(1, dim=1, keepdim=True)
        e = d.expand_as(b)
        f = b - e
        
        return f

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
    inp0 = torch.tensor([[-1.0565217733383179, 0.9817373752593994, 1.1849403381347656, 1.8239014148712158, 0.27914267778396606, 0.45316168665885925, -0.3436286151409149, 0.5453031063079834], [-1.0722485780715942, -1.4581655263900757, 0.4105833172798157, 0.6555752754211426, -0.5836853981018066, -1.0135717391967773, -1.0637238025665283, -2.0168917179107666], [1.3470723628997803, 0.05939164757728577, 0.5588117837905884, -0.01303786039352417, 0.9860228300094604, -0.8505029678344727, 0.10746737569570541, -0.07155431807041168], [-3.5278499126434326e-06, -1.147844910621643, 0.09884875267744064, 0.321972131729126, -0.456988126039505, -0.9627467393875122, -0.45523443818092346, 2.530625820159912]], dtype=torch.float32)
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
