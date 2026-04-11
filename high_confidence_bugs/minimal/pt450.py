import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.silu(self.fc1(torch.sin(x)))
        u = torch.outer(w[0], w[0]).unsqueeze(0)
        v = torch.cholesky_inverse(u)
        t = self.fc2(v.squeeze(0))
        return nn.functional.huber_loss(t, x[0])

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.full((4, 8), 1.7730573415756226)

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
