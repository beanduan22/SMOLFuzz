import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        y = self.fc1(x)
        z = torch.maximum(y, torch.sin(y)) - torch.cos(y)
        v = torch.bmm(z.unsqueeze(2), z.unsqueeze(1)).squeeze()
        return self.fc2(v)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [83945.0859, -229131.4375, -609619.6875, 988758.6875, -533363.5625, 152816.25, 184772.5313, -231754.3125],
    [-329318.2188, -100712.3125, 990370.5625, 512704.0, -475620.6563, -24123.5547, 128549.7734, -43233.9766],
    [-125001.6641, 344256.0, 368509.125, -371031.7188, 406933.5625, -329429.4688, 94081.375, -593497.125],
    [236846.2344, -17574.7441, -183436.4531, 739010.3125, 348209.9063, 713365.0, 161792.9531, -455819.8438],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
