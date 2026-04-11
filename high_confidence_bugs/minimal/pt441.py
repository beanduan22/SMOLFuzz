import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid_(self.fc2(x))
        x = F.adaptive_avg_pool1d(torch.logcumsumexp(x, dim=1), 8)
        x = self.relu(x.corrcoef())
        return x.cholesky().sum(dim=1)

torch.manual_seed(42)
m_cpu = Model()
torch.manual_seed(42)
m_gpu = Model().cuda()
m_gpu.load_state_dict(m_cpu.state_dict())

x = torch.tensor([
    [-0.025408625602722168, -0.00732085295021534, 0.01323717087507248, -0.007933376356959343, -0.0036197020672261715, -0.01284149568527937, 0.026837142184376717, -0.023954732343554497],
    [-0.02601183019578457, -0.021634312346577644, -0.01964784413576126, 0.015940474346280098, 0.0033350379671901464, 0.022145528346300125, 0.008589751087129116, -0.03545458987355232],
    [0.006427152547985315, -0.03092769905924797, 0.02781722880899906, -0.013294767588376999, -0.006846986711025238, 0.017567157745361328, -0.01650809496641159, -0.0011404029792174697],
    [-0.009468961507081985, -0.018325502052903175, -0.0006618089973926544, -0.004749863874167204, 0.0038775105495005846, 0.006596426013857126, 0.019777396693825722, 0.024068189784884453],
], dtype=torch.float32)

try:
    print("GPU:", m_gpu(x.cuda()).detach().cpu())
except Exception as e:
    print("GPU crash:", e)

try:
    print("CPU:", m_cpu(x).detach())
except Exception as e:
    print("CPU crash:", e)
