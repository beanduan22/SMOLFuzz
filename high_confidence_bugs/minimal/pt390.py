import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        u = torch.clip(z, min=-0.5, max=0.5)
        v = torch.log_(u.abs() + 1e-6)
        return nn.Softmin(dim=1)(self.fc2(v))

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [45819.9492, 435.0102, -70678.9922, -93307.0078, 139226.0938, 128386.9141, -21955.8145, 164057.4844],
    [81980.7266, 21167.3945, -106764.3984, 9196.9414, 75262.8359, 7774.1152, 15653.3037, 12470.6504],
    [48229.6719, -10403.791, -32509.8457, 38698.3711, -50140.5938, 5599.1812, -168723.4531, 135946.1094],
    [45057.293, 137189.7031, -52317.7617, 108110.4219, 33284.9805, -14167.207, -33159.5625, 2439.3909],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
