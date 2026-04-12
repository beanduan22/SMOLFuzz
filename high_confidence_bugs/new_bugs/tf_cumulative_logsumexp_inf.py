"""
Bug: tf.math.cumulative_logsumexp on all-inf input returns NaN at index 1+.
Source: TensorFlow GitHub issue #115091
Root cause: logsumexp(a, b) = max(a,b) + log1p(exp(min-max))
When both a=inf, b=inf: inf + log1p(exp(inf-inf)) = inf + log1p(exp(NaN)) = inf + NaN = NaN

Expected: cumulative_logsumexp([inf, inf, inf]) = [inf, inf, inf]
Bug: Returns [inf, NaN, NaN]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.math.cumulative_logsumexp inf→NaN bug ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

# Test 1: All-inf vector
print("\n--- Test 1: All-inf input ---")
test_cases = [
    tf.constant([[float('inf')], [float('inf')], [float('inf')]], dtype=tf.float32),
    tf.constant([float('inf'), float('inf'), float('inf'), float('inf')], dtype=tf.float32),
    tf.constant([[float('inf'), float('inf')], [float('inf'), float('inf')]], dtype=tf.float32),
    tf.constant([float('inf'), 0.0, float('inf'), 0.0], dtype=tf.float32),
    tf.constant([0.0, float('inf'), float('inf'), 0.0], dtype=tf.float32),
]

for i, x in enumerate(test_cases):
    try:
        cpu_result = tf.math.cumulative_logsumexp(x, axis=0).numpy()
        cpu_has_nan = np.any(np.isnan(cpu_result))
        print(f"Case {i+1}: input shape={x.shape}")
        print(f"  CPU result: {cpu_result.ravel().tolist()}")
        print(f"  CPU has NaN: {cpu_has_nan}", end="")
        if cpu_has_nan:
            print(f"  *** BUG: NaN from valid inf inputs ***", end="")
        print()

        if gpus:
            with tf.device('/GPU:0'):
                gpu_result = tf.math.cumulative_logsumexp(x, axis=0).numpy()
            gpu_has_nan = np.any(np.isnan(gpu_result))
            print(f"  GPU result: {gpu_result.ravel().tolist()}")
            print(f"  GPU has NaN: {gpu_has_nan}", end="")
            if gpu_has_nan:
                print(f"  *** BUG ***", end="")
            print()
            if cpu_has_nan != gpu_has_nan:
                print(f"  *** CPU/GPU DIVERGENCE: CPU_nan={cpu_has_nan}, GPU_nan={gpu_has_nan} ***")
    except Exception as e:
        print(f"Case {i+1}: ERROR: {e}")

# Test 2: Large finite values (near overflow)
print("\n--- Test 2: Large finite values (near log-overflow) ---")
x_large = tf.constant([85.0, 85.0, 85.0, 85.0], dtype=tf.float32)  # exp(85) ~ FLT_MAX
cpu_large = tf.math.cumulative_logsumexp(x_large, axis=0).numpy()
print(f"Input: [85, 85, 85, 85], CPU result: {cpu_large.tolist()}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_large = tf.math.cumulative_logsumexp(x_large, axis=0).numpy()
    print(f"GPU result: {gpu_large.tolist()}")
    diff = np.max(np.abs(cpu_large - gpu_large))
    print(f"CPU vs GPU diff: {diff:.3e}")

# Test 3: Mixed inf and -inf
print("\n--- Test 3: Mix of inf, -inf, nan ---")
x_mixed = tf.constant([float('inf'), float('-inf'), float('inf'), float('nan')], dtype=tf.float32)
try:
    cpu_mixed = tf.math.cumulative_logsumexp(x_mixed, axis=0).numpy()
    print(f"Input: [inf, -inf, inf, nan]")
    print(f"CPU: {cpu_mixed.tolist()}")
    if gpus:
        with tf.device('/GPU:0'):
            gpu_mixed = tf.math.cumulative_logsumexp(x_mixed, axis=0).numpy()
        print(f"GPU: {gpu_mixed.tolist()}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: 2D cumulative with all-inf column
print("\n--- Test 4: 2D input with inf column ---")
x_2d = tf.constant([[float('inf'), 1.0], [float('inf'), 2.0], [float('inf'), 3.0]], dtype=tf.float32)
cpu_2d = tf.math.cumulative_logsumexp(x_2d, axis=0).numpy()
print(f"Input shape (3,2), col0=all-inf, col1=[1,2,3]:")
print(f"CPU: {cpu_2d.tolist()}")
cpu_nan_col0 = np.any(np.isnan(cpu_2d[:, 0]))
print(f"NaN in inf-column: {cpu_nan_col0}", end="")
if cpu_nan_col0:
    print(" *** BUG ***", end="")
print()
if gpus:
    with tf.device('/GPU:0'):
        gpu_2d = tf.math.cumulative_logsumexp(x_2d, axis=0).numpy()
    print(f"GPU: {gpu_2d.tolist()}")
    gpu_nan_col0 = np.any(np.isnan(gpu_2d[:, 0]))
    if cpu_nan_col0 != gpu_nan_col0:
        print(f"*** CPU/GPU DIVERGENCE ***")

print(f"\n=== BUG SUMMARY ===")
x = tf.constant([[float('inf')], [float('inf')], [float('inf')]], dtype=tf.float32)
result = tf.math.cumulative_logsumexp(x, axis=0).numpy()
print(f"Input: [[inf], [inf], [inf]]")
print(f"CPU result: {result.ravel().tolist()}")
expected = [float('inf'), float('inf'), float('inf')]
print(f"Expected:   {expected}")
if np.any(np.isnan(result)):
    print("*** BUG CONFIRMED: NaN from valid inf inputs ***")
if gpus:
    with tf.device('/GPU:0'):
        gpu_result = tf.math.cumulative_logsumexp(x, axis=0).numpy()
    print(f"GPU result: {gpu_result.ravel().tolist()}")
    if np.any(np.isnan(gpu_result)) and not np.any(np.isnan(result)):
        print("*** BUG: GPU produces NaN, CPU does not ***")
    elif np.any(np.isnan(result)) and not np.any(np.isnan(gpu_result)):
        print("*** BUG: CPU produces NaN, GPU does not ***")
