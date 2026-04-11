import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        y = self.gelu(self.fc1(x))
        z = torch.sin(y) * torch.cos(y)
        w = self.fc2(z)
        d = torch.special.expm1(w).true_divide_(w)
        return torch.aminmax(d, dim=0)[0]

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-586343.9375, -1134628.875, -1039412.3125, -486660.0, 911998.875, -39699.2813, -1014223.3125, -486190.8125],
    [114514.5156, -210469.0156, 508707.4688, -488173.7188, 1144397.0, 55448.6953, 1246786.25, -761446.8125],
    [-903962.4375, 991748.6875, -833011.875, -225015.2031, -253670.8594, 225024.6875, -9477.999, 300967.9063],
    [354369.6563, -1006495.3125, 865711.3125, 1399779.375, 357127.25, -509767.0, 424651.2188, -628146.5625],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
