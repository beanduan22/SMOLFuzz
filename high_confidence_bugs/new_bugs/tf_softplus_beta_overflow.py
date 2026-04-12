"""
Bug: tf.nn.softplus with large beta — cross-framework check of PyTorch #171249.
softplus(x) in TF is defined as log(1 + exp(x)), no beta parameter.
But we can replicate: softplus(beta*x)/beta = log(1+exp(beta*x))/beta

Test: does TF's manual beta-softplus also produce inf for large beta?
CPU vs GPU comparison inside TF.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow softplus large-beta overflow ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

x = tf.constant([0.5, 1.0, 2.0, -1.0, 0.0], dtype=tf.float32)
reference = tf.nn.relu(x)  # limit of softplus(beta*x)/beta as beta→∞
print(f"\nReference (ReLU limit): {reference.numpy().tolist()}")

def tf_softplus_beta(x, beta):
    """Manual beta-softplus: log(1+exp(beta*x))/beta"""
    return tf.math.softplus(beta * x) / beta

def tf_softplus_beta_stable(x, beta):
    """Numerically stable: x + log1p(exp(-beta*x))/beta for x>0"""
    return tf.where(beta * x > 0,
                    x + tf.math.softplus(-beta * tf.abs(x)) / beta,
                    tf.math.softplus(beta * x) / beta)

betas = [1e10, 1e20, 1e30, 1e38]
for beta in betas:
    cpu_out = tf_softplus_beta(x, beta)
    print(f"\nbeta={beta:.0e}:")
    print(f"  TF (naive):  {cpu_out.numpy().tolist()}")
    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf_softplus_beta(x, beta)
        print(f"  TF GPU:      {gpu_out.numpy().tolist()}")

# Stable variant comparison
beta = 1e30
naive = tf_softplus_beta(x, beta)
stable = tf_softplus_beta_stable(x, beta)
print(f"\n=== BUG SUMMARY ===")
print(f"beta=1e30, x=[0.5, 1.0, 2.0, -1.0, 0.0]")
print(f"TF naive:   {naive.numpy().tolist()}")
print(f"TF stable:  {stable.numpy().tolist()}")
print(f"Reference:  {reference.numpy().tolist()}")
print(f"Naive has inf:  {np.any(np.isinf(naive.numpy()))}  (BUG)")
print(f"Stable has inf: {np.any(np.isinf(stable.numpy()))}")

if gpus:
    with tf.device('/GPU:0'):
        gpu_naive = tf_softplus_beta(x, beta)
        gpu_stable = tf_softplus_beta_stable(x, beta)
    print(f"\nGPU naive:  {gpu_naive.numpy().tolist()}")
    print(f"GPU stable: {gpu_stable.numpy().tolist()}")
    cpu_gpu_diff = np.max(np.abs(naive.numpy() - gpu_naive.numpy()))
    print(f"CPU vs GPU max diff: {cpu_gpu_diff:.3e}")
