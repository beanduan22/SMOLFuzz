# SMOLFuzz model 202 | llm=llama3.3:70b | attempts=3 | apis=7 | error=no
# USED_APIS_SELECTED = ['torch.Tensor.grad', 'torch.nn.functional.adaptive_avg_pool3d', 'torch.Tensor.nelement', 'torch.Tensor.lt', 'torch.unbind', 'torch.Tensor.ravel', 'torch.ComplexDoubleStorage', 'torch.CharStorage', 'torch.Tensor.floor_', 'torch.special.log_softmax', 'torch.Tensor.cumsum_', 'torch.linalg.matrix_rank', 'torch.result_type', 'torch.Tensor.istft', 'torch.mode', 'torch.set_default_dtype', 'torch.Tensor.is_set_to', 'torch.Tensor.erfinv_', 'torch.set_default_tensor_type', 'torch.Tensor.frexp', 'torch.nn.Linear', 'torch.sin', 'torch.Tensor.requires_grad_', 'torch.enable_grad']

import torch
import torch.nn as nn
import torch.special

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = torch.special.log_softmax(y, dim=1)
            w = torch.sin(z) * torch.cos(z)
            v = torch.floor(w)
            u = torch.cumsum(v, 1)
        return u

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.Linear", "torch.special.log_softmax", 
             "torch.sin", "torch.cos", "torch.enable_grad", 
             "torch.floor", "torch.cumsum"]

# Test the model
model = Model()
inputs = make_inputs()
output = model(*inputs)
print("Output shape:", output.shape)

# Test tensor methods
tensor = torch.randn(4, 8)
print("Tensor grad:", tensor.grad)
print("Tensor nelement:", tensor.nelement())
print("Tensor lt:", tensor.lt(tensor))
print("Unbind tensor:", list(torch.unbind(tensor)))
print("Ravel tensor:", tensor.ravel())

# Test complex double storage and char storage
complex_storage = torch.ComplexDoubleStorage()
char_storage = torch.CharStorage()

# Test result type
result_type = torch.result_type(tensor, tensor)

# Test mode
mode = torch.mode(tensor)[0]

# Set default dtype and tensor type
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

# Test is set to
is_set_to = tensor.is_set_to(tensor)

# Test frexp
frexp = tensor.frexp()

# Test linalg matrix rank
matrix_rank = torch.linalg.matrix_rank(tensor)

print("USED_APIS:", USED_APIS)