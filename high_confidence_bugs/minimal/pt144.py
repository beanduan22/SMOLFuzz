import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        u = torch.median(z, dim=1, keepdim=True)[0]
        v = torch.addcmul(torch.zeros_like(u), u, torch.tensor(2.0))
        w = torch.mvlgamma(v + 1.5, 8)
        return torch.digamma(w)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-9155.5869, 185314.8125, -162802.8281, 59676.7344, -18955.5234, -34882.8906, -21867.2344, 28905.1133],
    [-94904.2422, -52308.6289, 139624.6719, -32638.7871, -127087.9609, 115765.9063, -18739.6016, -32148.4004],
    [30900.4473, 88146.7188, 63134.3867, 106574.8047, 27316.957, -29811.2207, 15049.8057, -39792.5391],
    [-21032.4434, 94214.5078, -7887.436, 134287.875, -73570.3125, -135061.0625, -48483.9766, -49524.3555],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
