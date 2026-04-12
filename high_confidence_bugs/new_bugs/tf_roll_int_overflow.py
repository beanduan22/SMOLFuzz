"""
Bug: tf.roll (and tf.raw_ops.Roll) with INT32_MAX shift causes illegal GPU memory access.
Source: TensorFlow GitHub issue #109520
Cross-framework: Same test as pt_roll_int_overflow.py but in TensorFlow.

Expected: tf.roll(x, shift=N, axis=0) == tf.roll(x, shift=N % len(x), axis=0)
Bug: INT32_MAX shift causes GPU kernel to compute out-of-bounds index.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.roll large shift overflow ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

size = 1000
x = tf.cast(tf.range(size), tf.float32)

shifts_to_test = [
    2**31 - 1,   # INT32_MAX
    2**31,       # overflow boundary
    2**32 - 1,   # UINT32_MAX
]

print("\n--- 1D tf.roll large shift tests ---")
for shift in shifts_to_test:
    expected_shift = int(shift) % size
    ref = tf.roll(x, shift=expected_shift, axis=0)

    try:
        cpu_out = tf.roll(x, shift=shift, axis=0)
        cpu_correct = np.allclose(cpu_out.numpy(), ref.numpy())
        print(f"shift={shift}: CPU correct={cpu_correct}")

        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.roll(x, shift=shift, axis=0)
            gpu_correct = np.allclose(gpu_out.numpy(), ref.numpy())
            cpu_gpu_same = np.allclose(cpu_out.numpy(), gpu_out.numpy())
            print(f"  GPU correct={gpu_correct}, CPU==GPU: {cpu_gpu_same}")
            if not cpu_gpu_same:
                diff = np.max(np.abs(cpu_out.numpy() - gpu_out.numpy()))
                print(f"  MAX DIFF: {diff:.3e}  *** BUG ***")
    except Exception as e:
        print(f"shift={shift}: EXCEPTION — {type(e).__name__}: {e}")

# Test 2: 2D roll axis=-1 (exact match of TF #109520 setup, smaller size)
print("\n--- 2D tf.roll: axis=-1, INT32_MAX shift ---")
rows, cols = 500, 500
x2d = tf.random.normal([rows, cols], dtype=tf.float32, seed=42)
shift = 2**31 - 1
expected_shift = shift % cols
ref2d = tf.roll(x2d, shift=expected_shift, axis=-1)

try:
    cpu_out = tf.roll(x2d, shift=shift, axis=-1)
    cpu_correct = np.allclose(cpu_out.numpy(), ref2d.numpy())
    print(f"2D ({rows}x{cols}), shift=INT32_MAX: CPU correct={cpu_correct}")

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.roll(x2d, shift=shift, axis=-1)
        gpu_correct = np.allclose(gpu_out.numpy(), ref2d.numpy())
        cpu_gpu_same = np.allclose(cpu_out.numpy(), gpu_out.numpy())
        print(f"  GPU correct={gpu_correct}")
        print(f"  CPU == GPU: {cpu_gpu_same}")
        if not cpu_gpu_same:
            diff = np.max(np.abs(cpu_out.numpy() - gpu_out.numpy()))
            wrong = np.sum(cpu_out.numpy() != gpu_out.numpy())
            print(f"  MAX DIFF: {diff:.3e}, WRONG ELEMENTS: {wrong}  *** BUG ***")

    print(f"\n=== BUG SUMMARY ===")
    print(f"tf.roll(shape={rows}x{cols}, shift=INT32_MAX, axis=-1)")
    print(f"CPU correct: {cpu_correct}")
    if gpus:
        print(f"GPU correct: {gpu_correct}")
        print(f"CPU == GPU:  {cpu_gpu_same}")
    else:
        print("No GPU available to test")
except Exception as e:
    print(f"EXCEPTION (may indicate crash/corruption): {type(e).__name__}: {e}")
