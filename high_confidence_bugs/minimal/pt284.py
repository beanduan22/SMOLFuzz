import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        u = z.hardshrink(0.5).fmin(z)
        return self.fc2(u.true_divide_(2))

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [610511.375, -33056.3516, -604191.125, -331113.4375, 221792.7969, 85566.3359, 105654.5156, 232062.4219],
    [-429645.5625, 628693.6875, 594652.875, -459170.7813, -651905.8125, -529320.0625, -342601.5313, -1164186.875],
    [523613.75, 91171.4141, -159862.6094, -305621.2813, 90744.2656, -22588.5371, 185172.2969, -48455.4922],
    [-747211.75, 187488.1563, -294479.5, 179506.4531, 412394.2813, -175376.9844, -118936.7891, 788830.5625],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
