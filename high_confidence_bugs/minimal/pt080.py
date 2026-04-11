import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(4, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        z = torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))
        w = self.softmax(z)
        u = torch.nanmean(w, dim=1, keepdim=True)
        v = torch.atan2(u, w)
        t = torch.mm(v, w.t())
        nn.utils.weight_norm(self.fc2, name='weight', dim=0)
        return self.fc2(t)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-1107004.0, 914321.25, -1587441.125, -722095.1875, 47154.9258, 11018.3096, -697898.5625, -409708.8438],
    [479886.6563, -61289.3672, -68860.7109, -964705.8125, -749825.125, -593089.6875, 628875.1875, 124264.625],
    [341519.8125, 417857.9063, -1813769.875, -1160927.625, 527203.625, -2106610.25, -1226500.5, -755559.625],
    [-1208173.875, -70516.0234, 868239.1875, 465324.6563, -753960.5, -371414.5313, -63667.4414, -1069246.625],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
