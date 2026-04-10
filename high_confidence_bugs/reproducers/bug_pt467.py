#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 467
Bug Type: inconsistent
Root Cause: BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence
Detail  : output[0]: l2=1.7321e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.functional.dropout, torch.arctan, torch.matrix_exp
Mutation: scale_large

Run: python3 bug_pt467.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ─────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.bn(x)
            x = self.drop(x)
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.fc2(x)
            x = torch.arctan(x)
            x = torch.matrix_exp(x)
            x = x.greater_equal(0).float()
        return x

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
    inp0 = torch.tensor([[21766.7109375, 388724.625, 303715.03125, 306408.0625, -193268.90625, -69401.328125, -111206.9609375, 25280.720703125], [-113252.2109375, 85485.6015625, 163806.78125, 211680.109375, 37558.15625, 22505.21484375, 241715.3125, -178150.328125], [207439.125, -148052.828125, 16160.2939453125, -573633.5, 174184.84375, 255326.390625, 292608.1875, -9382.595703125], [-244177.46875, 1650.1563720703125, -31021.7734375, -583626.375, -44514.70703125, 254758.109375, 480061.46875, -199234.03125]], dtype=torch.float32)
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
