import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        jac = torch.autograd.functional.jacobian(
            lambda t: torch.sin(self.fc2(torch.tanh(torch.neg(torch.log1p(torch.cos(torch.sin(self.fc1(t)))))))), x
        )
        return jac.sum()

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-56649.0703, 44708.1367, 20531.2422, 196653.2188, 275958.9688, -44015.2773, -350452.0, 388331.625],
    [213176.6563, -363866.7188, 261445.1875, -494134.0625, -77968.8281, 33906.2031, 367492.375, -241145.2344],
    [106441.25, -356663.0938, 202312.7656, 252443.5, -142810.1406, 55002.3477, -32485.3086, 269925.4375],
    [-128967.2656, 96447.0391, 50964.5938, 109961.8516, -288699.2813, -256372.5781, 340626.4688, -322019.4688],
])

cpu = m_cpu(x)
gpu = m_gpu(x.cuda()).cpu()

print("CPU:", cpu.item())
print("GPU:", gpu.item())
print(f"L2: {abs(cpu.item() - gpu.item()):.4e}")
