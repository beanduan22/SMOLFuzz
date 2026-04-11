import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        y = self.fc1(x)
        v = torch.arccosh(y + 2)
        u = self.fc2(v)
        return torch.logdet(u), u.reciprocal_()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.full((4, 8), 0.4033372700214386)

with torch.no_grad():
    cpu_log, cpu_u = m_cpu(x)
    gpu_log, gpu_u = [o.cpu() for o in m_gpu(x.cuda())]

print("CPU logdet:", cpu_log.item())
print("GPU logdet:", gpu_log.item())
print("CPU is inf:", torch.isinf(cpu_log).item(), "GPU is nan:", torch.isnan(gpu_log).item())
