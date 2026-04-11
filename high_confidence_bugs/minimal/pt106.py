import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        y = self.fc1(x)  # jacobian is evaluated at fc1(x), not x
        jac = torch.autograd.functional.jacobian(
            lambda t: torch.sin(self.fc1(t)), y
        )
        return jac.sum()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [119823.8906, -454772.75, 622313.875, 156369.2031, -92904.0156, 339488.0938, -284196.9688, 274293.8125],
    [164817.3125, -69768.0, -231042.875, 286670.7813, -460369.875, 76860.1406, -1131659.875, -973842.875],
    [215890.8281, -7169.6035, -560793.3125, -393479.5625, 1604054.375, -836846.5, -413339.8438, 99432.7969],
    [-562458.4375, -671573.8125, 500004.0938, -1478818.625, 262910.5938, 590900.375, 196897.0938, -153577.1094],
])

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
