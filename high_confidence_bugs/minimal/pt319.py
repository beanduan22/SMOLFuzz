import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.up(x.unflatten(1, (4, 8))).flatten(1)
        return self.fc2(x)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-508597.4688, -338682.2813, -28296.8633, 196718.6094, -196383.0313, -746003.4375, 343720.9375, -37572.3828],
    [-391198.4375, 739655.4375, -90969.8047, -543350.5625, -599631.8125, 753509.4375, 104946.5078, 626955.75],
    [-82534.6172, -742640.9375, 404341.4688, -549422.3125, -473503.75, -597453.5625, -331676.3438, 382906.5625],
    [-1226559.5, -267103.4063, 13368.7568, 562068.375, -37062.5156, 143350.0469, 760769.9375, -728.4443],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu)
print("GPU:", gpu)
print(f"L2: {(cpu.float() - gpu.float()).norm().item():.4e}")
