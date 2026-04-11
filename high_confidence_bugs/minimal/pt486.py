import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, x):
        x = self.fc2(self.tanhshrink(torch.sin(self.fc1(x)) * torch.cos(self.fc1(x))))
        jac = torch.autograd.grad(x.var(dim=0).sum(), x, allow_unused=True)[0]
        return x, jac

torch.manual_seed(42)
m_cpu = Model().eval()
torch.manual_seed(42)
m_gpu = Model().cuda().eval()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [294037.3125, 414175.4063, -65990.6328, -923997.625, -1039047.125, -640162.6875, -727251.5625, -100772.7969],
    [265938.4375, -287175.125, -151044.0469, 382098.4375, 99202.1797, -348049.7813, 758885.625, 9890.415],
    [-158432.6406, -224531.6094, 468144.875, -23151.5156, -356240.0625, -227377.2344, 99491.8203, -332768.3438],
    [-284873.9688, 1061222.0, 820382.3125, -850360.4375, 202728.375, -132002.2813, -755424.5625, -826808.875],
])
x.requires_grad_(True)

cpu_x, cpu_j = m_cpu(x)
gpu_x, gpu_j = m_gpu(x.cuda())
gpu_x, gpu_j = gpu_x.cpu(), gpu_j.cpu()

print(f"output L2: {(cpu_x.float() - gpu_x.float()).norm().item():.4e}")
print(f"grad L2:   {(cpu_j.float() - gpu_j.float()).norm().item():.4e}")
