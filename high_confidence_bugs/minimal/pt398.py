import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import ifft2

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc1(x)
        z = F.hardsigmoid(y) * F.hinge_embedding_loss(y, torch.ones_like(y))
        w = ifft2(z)
        v = torch.logcumsumexp(w.real, 1)
        return torch.sin(v) * torch.cos(v)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [702997.75, -278614.9375, -1230909.375, -217134.6563, -43574.2578, 225041.2031, -941563.5, -334651.875],
    [-732716.75, 408498.7813, 208513.4219, -1657058.875, 572058.1875, 421530.0625, 1338393.25, -162130.7031],
    [1370748.625, 1195530.875, -487297.625, -751929.0625, 285215.7813, 122924.5938, -93846.0703, -1419654.875],
    [1378770.25, 362684.2813, -530503.875, 643936.0, 1771128.625, -514009.2188, -1075299.75, 433106.4063],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
