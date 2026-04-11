import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        self.buffer = nn.Parameter(torch.randn(4, 8))

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = self.fc2(x).logaddexp(self.buffer)
        return x.atan_().frac()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-333221.1563, -25957.5039, 34894.7383, 470963.9063, -189701.5938, -128185.4141, 398118.9063, 23546.2109],
    [293434.9688, 237614.8125, -335975.5625, -251305.5313, 131968.0938, 43010.5039, 2339.8271, 140569.8281],
    [-78501.5391, -42987.5117, -229267.2344, 305464.5625, -107468.5, 763593.4375, 84853.2188, -27106.752],
    [-337913.1563, -559285.3125, -199205.1563, -163815.7344, 144009.9688, 447394.0313, 514169.5938, 35510.7383],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
