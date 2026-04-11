import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        z = torch.special.log_softmax(self.fc1(x), dim=1)
        w = torch.sin(z) * torch.cos(z)
        return torch.cumsum(torch.floor(w), dim=1)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-182593.6094, 97438.2422, -602814.4375, 377043.1563, 517407.2813, 225786.9375, -1122366.75, -729344.125],
    [-556154.5, 195211.4219, -55275.9648, -515341.9688, -748593.5625, -327557.4688, -522184.5313, -652825.375],
    [-355279.7188, 664743.5, 673445.1875, -608570.0625, -40333.2617, -737681.375, 1795011.125, -1025.5217],
    [-161870.5313, 1263365.875, -307726.3125, 1439017.0, -358743.9063, -293719.3125, 845251.9375, -269744.3125],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
