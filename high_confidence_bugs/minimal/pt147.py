import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.bn(self.fc1(x))
        x = torch.sigmoid(x)
        x = self.fc2(x).expm1()
        x = x.polygamma_(1)
        return x.std(dim=0, keepdim=True)

# training mode intentional — BatchNorm uses batch stats (source of CPU/GPU divergence)
torch.manual_seed(42)
m_cpu = Model()
torch.manual_seed(42)
m_gpu = Model().cuda()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [1.2801445722579956, 2.084859848022461, 0.3327067494392395, -0.4003010094165802, 0.0, 0.16465674340724945, 0.5738624930381775, 0.0060699633322656155],
    [-0.514781653881073, -1.83219575881958, 1.3614412546157837, 1.6753082275390625, 0.0, 0.0, 0.48791050910949707, 0.7413539886474609],
    [0.669989824295044, 0.0, 0.0, 1.5798181295394897, 0.5415620803833008, 0.0, 0.0, -0.21595177054405212],
    [-0.8319048881530762, -0.11799944937229156, 1.7658331394195557, -1.221530556678772, 0.0, 0.0, 0.09160400182008743, 0.0],
], dtype=torch.float32)

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
