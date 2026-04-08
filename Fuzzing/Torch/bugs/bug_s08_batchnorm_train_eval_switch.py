"""
Bug S08 — INCONSISTENT: BatchNorm train/eval switch mid-forward causes stat divergence (model_0277)
Trigger: add_noise mutation

Root cause: The model calls self.bn(x) in eval mode (uses stored running_mean/var),
then calls self.train() and self.bn(x) again in training mode (uses batch stats).
Switching training mode mid-forward updates running_mean/running_var on CPU and
GPU using different parallel-reduction orderings, causing the stored statistics
to diverge. The subsequent acos_ of a clamped norm is sensitive to these differences.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(8, 8)
        self.bn   = nn.BatchNorm1d(8)
        self.sigm = nn.Sigmoid()
        self.fc2  = nn.Linear(8, 4)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.fc1(x)
            x = self.bn(x)                # eval mode: uses running stats
            x = self.sigm(x)
            self.train()                   # switch to train mid-forward
            x = self.bn(x)                # train mode: updates running stats
            self.eval()
            x = self.fc2(x)
            y = torch.log2(x.abs() + 1e-8)
            u = x.norm(dim=1, keepdim=True)
            v = torch.acos_(torch.clamp(u / (u.max() + 1e-8), -1+1e-7, 1-1e-7))
        return v

model = Model().eval()

# GPU version with identical weights
model_g = Model().eval().cuda()
model_g.load_state_dict(model.state_dict())

x = torch.randn(4, 8) + torch.randn(4, 8) * 0.05  # add_noise

cpu_out = model(x)
gpu_out = model_g(x.cuda()).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
print(f"CPU: {cpu_out.flatten()}")
print(f"GPU: {gpu_out.flatten()}")
