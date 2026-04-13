"""Bug 52: Cross-framework — TF handles large beta correctly, PyTorch does not.
Source: PyTorch GitHub issue #171249
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch
import torch.nn.functional as F

x = [0.5, 1.0, 2.0]

pt_out = F.softplus(torch.tensor(x), beta=1e30, threshold=float('inf'))
tf_out = tf.nn.softplus(tf.constant(x) * 1e30) / 1e30

print("PyTorch:", pt_out.tolist())    # BUG: [inf, inf, inf]
print("TF:     ", tf_out.numpy())     # correct: [0.5, 1.0, 2.0]
print("Expected: [0.5, 1.0, 2.0]")
assert not any(float('inf') == v for v in tf_out.numpy()), "TF should be finite"
print("BUG CONFIRMED: PyTorch overflows, TF returns correct result")
