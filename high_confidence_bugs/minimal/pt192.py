import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        w = torch.logsumexp(z, dim=1, keepdim=True)
        v = torch.copysign(w, z)
        u = v + 0.5 * (0.3 / (0.2 + 1e-6))
        return torch.log2(u)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [191291.3906, -203984.0156, -813554.8125, -133981.0156, -634194.25, 1800681.875, 733763.0625, 789282.3125],
    [436030.0625, -289938.0938, 939114.5625, 1230863.5, -1602146.125, -327240.1563, 120323.5703, 475272.0313],
    [209243.5, -664976.4375, -1178874.375, 1711560.75, -806346.6875, -1759401.875, -1441975.25, 100767.875],
    [399113.8438, 534722.8125, -721668.125, -820032.625, -429900.5625, -1028356.9375, -148990.5, -497775.5938],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU nan count:", torch.isnan(cpu).sum().item())
print("GPU nan count:", torch.isnan(gpu).sum().item())
print("Asymmetric NaN positions:", ((torch.isnan(cpu) != torch.isnan(gpu)).sum().item()))
