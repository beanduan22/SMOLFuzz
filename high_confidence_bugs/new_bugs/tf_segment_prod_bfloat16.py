"""
Bug: tf.math.unsorted_segment_prod with bfloat16 — GPU uses non-deterministic
atomicMul for accumulation. CPU uses sequential multiplication. For bfloat16,
this leads to both non-determinism and CPU/GPU divergence.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow unsorted_segment_prod bfloat16: CPU vs GPU ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

rng = np.random.default_rng(42)

print("\n--- bfloat16 unsorted_segment_prod ---")
for M, N in [(10000, 50), (50000, 100), (100000, 200), (500000, 500)]:
    # Values near 1.0 to prevent overflow
    src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float32)
    idx_np = rng.integers(0, N, size=M).astype(np.int32)

    # Reference float64
    ref = np.ones(N, dtype=np.float64)
    count = np.zeros(N, dtype=np.int64)
    for i in range(M):
        ref[idx_np[i]] *= float(src_np[i])
        count[idx_np[i]] += 1

    src = tf.constant(src_np, dtype=tf.bfloat16)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)

        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)

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
print("\n--- GPU non-determinism: segment_prod bfloat16 ---")
M, N = 100000, 200
src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float32)
idx_np = rng.integers(0, N, size=M).astype(np.int32)
src = tf.constant(src_np, dtype=tf.bfloat16)
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

# Also test float16
print("\n--- float16 unsorted_segment_prod ---")
for M, N in [(50000, 100), (100000, 200)]:
    src_np = (rng.standard_normal(M) * 0.01 + 1.0).astype(np.float16)
    idx_np = rng.integers(0, N, size=M).astype(np.int32)
    src = tf.constant(src_np, dtype=tf.float16)
    idx = tf.constant(idx_np, dtype=tf.int32)

    try:
        cpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)
        if gpus:
            with tf.device('/GPU:0'):
                gpu_out = tf.math.unsorted_segment_prod(src, idx, N).numpy().astype(np.float64)
            diff = float(np.max(np.abs(cpu_out - gpu_out)))
            print(f"f16 M={M}, N={N}: diff={diff:.3e}", end="")
            if diff > 0.01:
                print(f"  *** DIVERGENCE ***", end="")
            print()
    except Exception as e:
        print(f"f16 M={M}, N={N}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
M, N = 100000, 200
src_np2 = (np.random.default_rng(99).standard_normal(M) * 0.01 + 1.0).astype(np.float32)
idx_np2 = np.random.default_rng(99).integers(0, N, size=M).astype(np.int32)
src2 = tf.constant(src_np2, dtype=tf.bfloat16)
idx2 = tf.constant(idx_np2, dtype=tf.int32)

cpu_out2 = tf.math.unsorted_segment_prod(src2, idx2, N).numpy().astype(np.float64)
print(f"unsorted_segment_prod(bfloat16, M={M}, N={N}):")
if gpus:
    with tf.device('/GPU:0'):
        gpu_out2 = tf.math.unsorted_segment_prod(src2, idx2, N).numpy().astype(np.float64)
        gpu_out2b = tf.math.unsorted_segment_prod(src2, idx2, N).numpy().astype(np.float64)
    diff_cg = float(np.max(np.abs(cpu_out2 - gpu_out2)))
    nd = float(np.max(np.abs(gpu_out2 - gpu_out2b)))
    print(f"CPU vs GPU max diff: {diff_cg:.3e}")
    print(f"GPU non-determinism: {nd:.3e}")
    if diff_cg > 0.01:
        print(f"*** BUG: CPU and GPU segment_prod differ by {diff_cg:.3e} ***")
    elif nd > 0:
        print(f"*** BUG: GPU non-deterministic (diff={nd:.3e}) ***")
    else:
        print("No significant divergence")
