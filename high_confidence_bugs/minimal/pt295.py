import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)
        self.softshrink = nn.Softshrink(lambd=0.5)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = self.softshrink(x)
        x = self.fc2(x)
        self.train(); x = self.bn(x); x = self.bn(x); self.eval()
        return x.remainder(1.0)

# Note: self.train()/eval() inside forward triggers different BatchNorm paths
torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.full((4, 8), 1.4817914962768555)

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
