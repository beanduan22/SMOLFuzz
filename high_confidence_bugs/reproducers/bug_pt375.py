#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 375
Bug Type: inconsistent
Root Cause: BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence
Detail  : output[0]: l2=2.4601e+00 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.functional.dropout, torch.tanh, torch.Tensor.sinc_
Mutation: add_noise

Run: python3 bug_pt375.py
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

    def forward(self, x):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            x = F.tanh(self.fc1(x))
            x = self.bn(x)
            x = self.drop(x)
            x = self.fc2(x)
        
        # Switching modes
        self.train()
        x = self.bn(x)
        self.eval()
        x = self.bn(x)
        
        # Using dropout in eval mode (identity)
        x = self.drop(x)
        
        # Applying sinc_
        x.sinc_()
        
        # Linear algebra operations
        norm = torch.linalg.norm(x, dim=1, keepdim=True)
        x = torch.div(x, norm + 1e-5)
        
        return x

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
    inp0 = torch.tensor([[-1.487096905708313, 0.828857421875, 1.2989463806152344, -0.5988630652427673, 1.5388692617416382, 0.3135000169277191, 0.9483519792556763, -0.15384162962436676], [-1.0513911247253418, 0.11044211685657501, 0.19087624549865723, -0.6315414905548096, -0.24875497817993164, -0.46867620944976807, -1.0669728517532349, -1.0768659114837646], [-1.2317456007003784, 0.16817203164100647, -0.5135955810546875, 0.7820791006088257, 1.0378071069717407, -1.2233976125717163, -1.2107113599777222, -0.749923586845398], [0.5333338379859924, -0.43717601895332336, 0.3146974742412567, 0.08150574564933777, -0.7573711276054382, 0.5895403623580933, 0.7765327095985413, -0.7340806722640991]], dtype=torch.float32)
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
