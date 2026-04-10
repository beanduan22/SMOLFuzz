#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 357
Bug Type: inconsistent
Root Cause: BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence
Detail  : output[0]: l2=5.7845e-01 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.functional.affine_grid, torch.index_select, torch.Tensor.ge_, torch.nn.BatchNorm1d, torch.nn.Dropout
Mutation: mask

Run: python3 bug_pt357.py
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
        self.fc1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(8, 8)
        self.kldiv_loss = nn.KLDivLoss(reduction='batchmean')
        self.weight_norm_fc = nn.utils.weight_norm(nn.Linear(8, 8))

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = F.relu(x)
            grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).expand(4, -1, -1), (4, 1, 8, 8))
            selected = torch.index_select(grid.reshape(4, -1, 2), 1, torch.tensor([0]))
            x = x.ge_(torch.mean(x)).float()
            x = self.bn(x)
            x = self.drop(x)
            x = self.fc2(x)
            target = torch.full_like(x, 0.5)
            loss = self.kldiv_loss(F.log_softmax(x, dim=1), F.softmax(target, dim=1))
            x = x.renorm_(p=2, dim=1, maxnorm=1.0) # changed dim from 0 to 1
            _, indices = torch.mode(x, dim=1)
            x = torch.sin(x)
            x = self.weight_norm_fc(x)
            x = torch.sqrt(torch.clamp(x, min=0)) # clamp values to prevent NaNs
        return x

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
    inp0 = torch.tensor([[0.0, 0.8028251528739929, -0.2810384929180145, -1.217956304550171, 0.0, 0.0, 0.26621419191360474, 0.15644356608390808], [0.0, -0.43585026264190674, 0.0, 0.0, -1.5145453214645386, -0.14186717569828033, 0.1686246544122696, -1.2342407703399658], [-0.5872898697853088, 0.0, 1.2242895364761353, 0.0, -0.32154181599617004, -0.9924722909927368, -1.043931007385254, 0.5522363185882568], [0.0, 0.0, -0.4151269495487213, -1.0463557243347168, 1.0163828134536743, 0.0, 0.201784148812294, 0.0]], dtype=torch.float32)
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
