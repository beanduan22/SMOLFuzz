# https://github.com/pytorch/pytorch/issues/52241
import torch
import torch.nn as nn

torch.manual_seed(0)
V, T = 2, 3
params = nn.functional.log_softmax(torch.randn(T, 1, V + 1), dim=-1).double()
params.requires_grad = True
labels = torch.tensor([[1]])
input_lengths = torch.tensor([T])
label_lengths = torch.tensor([1])
torch.autograd.gradcheck(nn.CTCLoss(), [params, labels, input_lengths, label_lengths])

torch.manual_seed(1)
V, T = 5, 8
params = nn.functional.log_softmax(torch.randn(T, 2, V + 1), dim=-1).double()
params.requires_grad = True
labels = torch.tensor([[1, 2], [3, 4]])
torch.autograd.gradcheck(
    nn.CTCLoss(reduction="sum"),
    [params, labels, torch.tensor([T, T]), torch.tensor([2, 2])],
)
