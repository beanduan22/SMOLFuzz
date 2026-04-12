"""
Bug: tf.math.unsorted_segment_mean with float16 — GPU uses non-deterministic atomicAdd
for accumulation before dividing by count. CPU uses sequential summation then divides.
This leads to both non-determinism and CPU/GPU divergence for float16.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow unsorted_segment_mean float16: CPU vs GPU ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

rng = np.random.default_rng(42)

print("\n--- float16 unsorted_segment_mean ---")
for M, N in [(10000, 50), (100000, 200), (500000, 500)]:
    src_np = rng.standard_normal(M).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int32)

    # Reference (float64 mean)
    ref_sum = np.zeros(N, dtype=np.float64)
    ref_count = np.zeros(N, dtype=np.int64)
    for i in range(M):
        ref_sum[idx_np[i]] += float(src_np[i])
        ref_count[idx_np[i]] += 1
    ref = ref_sum / np.maximum(ref_count, 1)

    src = tf.constant(src_np, dtype=tf.float16)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)

        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)

            cpu_err = float(np.max(np.abs(cpu_out - ref)))
            gpu_err = float(np.max(np.abs(gpu_out - ref)))
            diff = float(np.max(np.abs(cpu_out - gpu_out)))

            print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
            if diff > 0.01:
                print(f"  *** CPU/GPU DIVERGENCE ***", end="")
            print()
    except Exception as e:
        print(f"M={M}, N={N}: ERROR: {e}")

# GPU non-determinism check
print("\n--- GPU non-determinism: unsorted_segment_mean float16 ---")
M, N = 500000, 500
src_np = rng.standard_normal(M).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int32)
src = tf.constant(src_np, dtype=tf.float16)
idx = tf.constant(idx_np, dtype=tf.int32)

if gpus:
    runs = []
    for _ in range(3):
        with tf.device('/GPU:0'):
            out = tf.math.unsorted_segment_mean(src, idx, N).numpy()
        runs.append(out.astype(np.float64))

    diff01 = float(np.max(np.abs(runs[0] - runs[1])))
    diff02 = float(np.max(np.abs(runs[0] - runs[2])))
    print(f"M={M}, N={N}: run0_vs_run1={diff01:.3e}, run0_vs_run2={diff02:.3e}", end="")
    if diff01 > 0 or diff02 > 0:
        print("  *** GPU NON-DETERMINISTIC ***", end="")
    print()

# bfloat16
print("\n--- bfloat16 unsorted_segment_mean ---")
for M, N in [(100000, 200), (500000, 500)]:
    src_np = rng.standard_normal(M).astype(np.float32)  # bf16 needs float32 source
    idx_np = rng.integers(0, N, size=M).astype(np.int32)
    src = tf.constant(src_np, dtype=tf.bfloat16)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)
        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)
            diff = float(np.max(np.abs(cpu_out - gpu_out)))
            print(f"bf16 M={M}, N={N}: diff={diff:.3e}", end="")
            if diff > 0.01:
                print(f"  *** DIVERGENCE ***", end="")
            print()
    except Exception as e:
        print(f"bf16 M={M}, N={N}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
M, N = 100000, 200
src_np = rng.standard_normal(M).astype(np.float16)
idx_np = rng.integers(0, N, size=M).astype(np.int32)
src = tf.constant(src_np, dtype=tf.float16)
idx = tf.constant(idx_np, dtype=tf.int32)

cpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)
print(f"unsorted_segment_mean(float16, M={M}, N={N}):")
if gpus:
    with tf.device('/GPU:0'):
        gpu_out = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)
    diff = float(np.max(np.abs(cpu_out - gpu_out)))
    print(f"CPU vs GPU max diff: {diff:.3e}")
    if diff > 0.01:
        print(f"*** BUG: CPU and GPU segment_mean differ by {diff:.3e} ***")
    else:
        with tf.device('/GPU:0'):
            gpu2 = tf.math.unsorted_segment_mean(src, idx, N).numpy().astype(np.float64)
        nd = float(np.max(np.abs(gpu_out - gpu2)))
        print(f"GPU non-determinism: {nd:.3e}")
        if nd > 0:
            print(f"*** BUG: GPU non-deterministic (diff={nd:.3e}) ***")
        else:
            print("No significant divergence")
