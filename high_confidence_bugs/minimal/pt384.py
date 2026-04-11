import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        jac = torch.autograd.functional.jacobian(
            lambda t: torch.sin(self.fc2(t)), x
        )
        return jac.sum()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [432001.4063, -178943.8281, 256427.6719, 108605.0313, 622762.5, 254057.5469, -475429.4375, -316443.875],
    [1751907.375, -486254.625, 332341.2813, 38548.3477, 1476846.75, 145677.0, 670375.875, -256612.7969],
    [-329658.375, -190186.0938, 829730.125, -541971.375, -258808.1094, 26393.6113, 937689.9375, -442783.5625],
    [-57529.7188, 1021103.0, 439405.7813, 417694.6563, 58830.5938, -934694.0, 309999.1875, 103314.0781],
])

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
