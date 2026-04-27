# https://github.com/pytorch/pytorch/issues/162235
import torch
import torch.nn.functional as F

x_single = torch.tensor([-0.0])
x_pair = torch.tensor([-0.0, 0.0])

print("cpu maximum :", torch.maximum(x_single, torch.tensor([0.0])))
print("gpu maximum :", torch.maximum(x_single.cuda(), torch.tensor([0.0]).cuda()))

print("cpu relu    :", F.relu(x_single))
print("gpu relu    :", F.relu(x_single.cuda()))

print("cpu argsort :", torch.argsort(x_pair))
print("gpu argsort :", torch.argsort(x_pair.cuda()))

print("cpu amin    :", torch.amin(x_pair, dim=0))
print("gpu amin    :", torch.amin(x_pair.cuda(), dim=0))

for dt in (torch.float64, torch.float32, torch.float16):
    x = torch.tensor([-0.0], dtype=dt)
    print(f"cpu max/{dt}:", torch.maximum(x, torch.tensor([0.0], dtype=dt)))
    print(f"gpu max/{dt}:", torch.maximum(x.cuda(), torch.tensor([0.0], dtype=dt).cuda()))
