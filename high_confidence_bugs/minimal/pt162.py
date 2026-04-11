import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(9, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        w = torch.special.erfc(z)
        v = torch.exp(w)
        u = torch.nansum(v, dim=1).unsqueeze(1)
        return self.fc2(torch.concat([u, v], dim=1))

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [166837.4844, -119079.75, -46291.418, 199545.8438, -111630.0, -273368.5313, 188674.1094, 347940.9688],
    [147105.9375, 184394.7969, -252079.25, -118651.8359, 168386.0156, 26887.3203, -113678.2266, -81572.3125],
    [178314.6719, -130999.4063, -87117.1094, -337177.9688, -26907.75, 195786.0313, -328342.0938, -596075.3125],
    [-3371.9485, 167403.9844, 213312.4844, 55590.4531, -193117.1875, 251031.2813, 128132.2578, -357162.3125],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
