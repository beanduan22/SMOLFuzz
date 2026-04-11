import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        w = torch.sin(self.gelu(self.fc1(x))) * torch.cos(self.gelu(self.fc1(x)))
        v = self.fc2(w)
        jac = torch.autograd.functional.jacobian(lambda t: torch.sin(self.fc1(t)), x)
        return v + jac.sum()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [417695.625, -413094.7813, 822908.375, 84662.8672, 676017.3125, 671627.625, -1991231.25, 557041.75],
    [-266216.5313, 141292.9219, -274570.875, -281652.8125, 531370.375, -993285.25, 630516.3125, -609379.75],
    [-784808.3125, -703746.5625, 1149344.375, 914229.875, -813672.3125, -247406.7031, -491951.4063, 957009.6875],
    [-637863.75, 282156.9375, -923531.875, -970033.625, -151085.2031, 2149866.25, 845727.5625, -69333.9609],
])

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
