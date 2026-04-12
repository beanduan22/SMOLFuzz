"""
Bug: tf.math.unsorted_segment_prod with float16 — GPU uses non-deterministic atomicMul
while CPU sequentially multiplies values within each segment.
For many values in a segment, float16 precision loss compounds differently.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow unsorted_segment_prod float16: CPU vs GPU ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

rng = np.random.default_rng(42)

print("\n--- float16 unsorted_segment_prod ---")
for M, N in [(10000, 50), (50000, 100), (100000, 200)]:
    # Use values near 1.0 to avoid immediate overflow
    src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int32)

    # Reference (float64)
    ref = np.ones(N, dtype=np.float64)
    for i in range(M):
        ref[idx_np[i]] *= float(src_np[i])

    src = tf.constant(src_np, dtype=tf.float16)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy()
        cpu_np = cpu_out.astype(np.float64)

        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy()
            gpu_np = gpu_out.astype(np.float64)

            cpu_err = float(np.max(np.abs(cpu_np - ref) / (np.abs(ref) + 1e-30)))
            gpu_err = float(np.max(np.abs(gpu_np - ref) / (np.abs(ref) + 1e-30)))
            diff = float(np.max(np.abs(cpu_np - gpu_np)))

            print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
            if diff > 0.01:
                print(f"  *** DIVERGENCE ***", end="")
            print()
    except Exception as e:
        print(f"M={M}, N={N}: ERROR: {e}")

# Non-determinism check
print("\n--- GPU non-determinism: unsorted_segment_prod float16 ---")
M, N = 100000, 200
src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int32)
src = tf.constant(src_np, dtype=tf.float16)
idx = tf.constant(idx_np, dtype=tf.int32)

if gpus:
    runs = []
    for _ in range(3):
        with tf.device('/GPU:0'):
            out = tf.math.unsorted_segment_prod(src, idx, N).numpy()
        runs.append(out.astype(np.float64))

    diff01 = float(np.max(np.abs(runs[0] - runs[1])))
    diff02 = float(np.max(np.abs(runs[0] - runs[2])))
    print(f"M={M}, N={N}: run0_vs_run1={diff01:.3e}, run0_vs_run2={diff02:.3e}", end="")
    if diff01 > 0 or diff02 > 0:
        print("  *** GPU NON-DETERMINISTIC ***", end="")
    print()

# float32 comparison
print("\n--- float32 unsorted_segment_prod (reference) ---")
for M, N in [(10000, 50), (50000, 100)]:
    src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float32)
    idx_np = rng.integers(0, N, size=M).astype(np.int32)
    src = tf.constant(src_np, dtype=tf.float32)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy()
        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy()
            diff = float(np.max(np.abs(cpu_out.astype(np.float64) - gpu_out.astype(np.float64))))
            print(f"f32 M={M}, N={N}: diff={diff:.3e}")
    except Exception as e:
        print(f"f32 M={M}, N={N}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
M, N = 50000, 100
src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int32)
src = tf.constant(src_np, dtype=tf.float16)
idx = tf.constant(idx_np, dtype=tf.int32)

cpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)
print(f"unsorted_segment_prod(float16, M={M}, N={N}):")
if gpus:
    with tf.device('/GPU:0'):
        gpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)
    diff = float(np.max(np.abs(cpu_out - gpu_out)))
    print(f"CPU vs GPU max diff: {diff:.3e}")
    if diff > 0.01:
        print(f"*** BUG: CPU and GPU produce different results for float16 segment_prod ***")
    else:
        # Check non-determinism
        with tf.device('/GPU:0'):
            gpu2 = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)
        nd = float(np.max(np.abs(gpu_out - gpu2)))
        print(f"GPU non-determinism: {nd:.3e}")
        if nd > 0:
            print(f"*** BUG: GPU non-deterministic (run diff={nd:.3e}) ***")
        else:
            print("No significant divergence")
