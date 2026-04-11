import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(F.softplus(self.fc1(x))) * torch.cos(F.softplus(self.fc1(x)))
        v = torch.diag(z.sum(dim=0))
        u = torch.geqrf(v)[0]
        r = torch.unique(torch.mv(u.t(), x.sum(dim=0)).view(1, 8).expand(x.size(0), 8), dim=0)
        return self.fc2(r.float())

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [58334.1133, 18151.2148, -131812.8438, -48904.7305, 47871.0117, 11011.9102, -122171.1875, 62493.5664],
    [9485.8232, 60270.4141, 176497.2656, 17260.3379, -75749.1875, 195830.5156, -100473.0625, 68490.0859],
    [94649.8594, 9531.7012, -120367.6484, 135898.5938, 181354.6563, 48062.8242, 32151.4531, 62332.4648],
    [34571.5156, 9817.3008, -61054.7969, -31538.1035, -183182.5156, -54603.9688, -47099.6328, 25398.4219],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
