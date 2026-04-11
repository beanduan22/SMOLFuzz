import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc1(x)
        z = torch.fmod(y, 0.5) * torch.sgn(y)
        w = self.fc2(z)
        return torch.sin(nn.SmoothL1Loss()(w, y))

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-1645617.375, -396535.1875, 640078.6875, 382541.6563, -1730108.5, 687192.75, 608672.9375, 1196564.375],
    [190381.1563, 106980.6172, 779027.5625, 1296586.5, 1399299.75, -1821212.0, 2036436.375, -250166.2031],
    [-837925.0, 2509467.5, -1177072.5, -194206.4063, -662927.25, -295987.875, -1034618.875, -484003.375],
    [516603.75, -1334311.75, 12325.3174, 643731.875, 869192.8125, 1028689.5625, -145303.8281, 466193.5625],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
