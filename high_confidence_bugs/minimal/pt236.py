import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        u = torch.mean(z)
        return torch.sin(self.fc2(z)) + u

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [670890.0625, -689509.9375, 1145859.125, -685675.6875, -1672820.625, 104331.5156, 597883.25, -901974.3125],
    [138407.9375, -391745.4375, -233220.5625, 470946.5625, 1236021.25, 449253.4375, -405428.9688, 973444.875],
    [602300.3125, -879032.75, -451999.875, -560600.5, -1200804.625, -761568.3125, -283827.25, 488784.7188],
    [-297777.4063, -428206.3125, -894782.375, 658649.4375, 88165.2031, 1363669.0, -511385.0313, -534593.25],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
