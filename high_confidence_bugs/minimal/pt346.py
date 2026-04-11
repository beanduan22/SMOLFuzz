import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        w = torch.hann_window(8, device=x.device)
        z = torch.sin(x) * torch.cos(w.unsqueeze(0))
        u = torch.cumprod(torch.erf(self.fc1(z)), dim=1)
        return nn.functional.mse_loss(self.fc2(u), x)

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-393721.5, 686425.4375, 43296.7461, 740778.375, 495242.2188, -41800.8242, 378486.1875, -226442.7344],
    [-475392.0938, -14079.9189, -416967.0625, -39561.4531, -461622.5625, -562811.875, 360509.9688, 586939.625],
    [-209398.3906, -164781.5469, 1370588.375, -268644.1563, 510658.9375, -674704.5625, -211712.7344, -725142.875],
    [-286453.25, 321865.2813, -465922.1563, 1221045.375, -1242109.75, -264.6772, 147065.3125, 405030.625],
])

with torch.no_grad():
    cpu = m_cpu(x)
    gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
