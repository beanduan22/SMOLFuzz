import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)

    def forward(self, x):
        jac = torch.autograd.functional.jacobian(
            lambda t: torch.sin(self.fc1(t)), x
        )
        return jac.sum()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [22876.0684, 60837.1875, 71575.6406, -123327.6406, -66152.9375, -39915.9492, 85655.5, 44349.5664],
    [-24698.4844, -35356.1016, -21055.3418, -74739.9688, 40536.3086, -59745.0078, 145119.3906, 41017.4453],
    [-111954.4375, 23978.2383, 41092.5781, 14050.9063, 107100.1953, -114077.375, 33377.1211, -151744.5781],
    [-31536.0059, -91353.0078, -13304.6396, -60389.8125, -44722.0039, -148689.1875, -31318.2148, -3824.1902],
])

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
