"""
Bug: tf.math.segment_sum / tf.math.unsorted_segment_sum with float16.
GPU uses non-deterministic atomicAdd (similar to TF scatter_add_nondeterministic),
CPU is sequential and more accurate.

Also tests bfloat16 and different segment sizes.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow segment_sum float16: CPU vs GPU accuracy ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

rng = np.random.default_rng(42)

print("\n--- float16 unsorted_segment_sum accuracy ---")
for M, N in [(1000, 50), (10000, 100), (100000, 500), (500000, 1000)]:
    data_np = rng.standard_normal(M).astype(np.float16)
    seg_ids = rng.integers(0, N, size=M).astype(np.int32)

    ref_np = np.zeros(N, dtype=np.float64)
    for i in range(M):
        ref_np[seg_ids[i]] += data_np[i]

    data_f16 = tf.constant(data_np, dtype=tf.float16)
    seg = tf.constant(seg_ids)

    cpu_out = tf.math.unsorted_segment_sum(data_f16, seg, N).numpy().astype(np.float64)
    cpu_err = float(np.max(np.abs(cpu_out - ref_np)))

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.math.unsorted_segment_sum(data_f16, seg, N).numpy().astype(np.float64)
        gpu_err = float(np.max(np.abs(gpu_out - ref_np)))
        diff = float(np.max(np.abs(cpu_out - gpu_out)))
        ratio = gpu_err / (cpu_err + 1e-30)

        print(f"M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={diff:.3e}", end="")
        if ratio > 5 and diff > 0.5:
            print(f"  *** GPU {ratio:.1f}x worse ***", end="")
        elif diff > 1.0:
            print(f"  *** SIGNIFICANT DIVERGENCE ***", end="")
        print()

# bfloat16
print("\n--- bfloat16 unsorted_segment_sum ---")
for M, N in [(10000, 100), (100000, 500)]:
    data_np = rng.standard_normal(M).astype(np.float32)
    seg_ids = rng.integers(0, N, size=M).astype(np.int32)
    ref_np = np.zeros(N, dtype=np.float64)
    for i in range(M):
        ref_np[seg_ids[i]] += data_np[i]

    data_bf16 = tf.constant(data_np, dtype=tf.bfloat16)
    seg = tf.constant(seg_ids)
    cpu_out = tf.math.unsorted_segment_sum(data_bf16, seg, N).numpy().astype(np.float64)
    cpu_err = float(np.max(np.abs(cpu_out - ref_np)))

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.math.unsorted_segment_sum(data_bf16, seg, N).numpy().astype(np.float64)
        gpu_err = float(np.max(np.abs(gpu_out - ref_np)))
        diff = float(np.max(np.abs(cpu_out - gpu_out)))
        ratio = gpu_err / (cpu_err + 1e-30)
        print(f"bf16 M={M}, N={N}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, diff={diff:.3e}", end="")
        if ratio > 5:
            print(f"  GPU {ratio:.1f}x worse", end="")
        elif diff > 1.0:
            print(f"  *** SIGNIFICANT ***", end="")
        print()

# GPU non-determinism check
print("\n--- GPU non-determinism check ---")
if gpus:
    for M, N in [(100000, 500), (500000, 1000)]:
        data_np = rng.standard_normal(M).astype(np.float16)
        seg_ids = rng.integers(0, N, size=M).astype(np.int32)
        data_f16 = tf.constant(data_np, dtype=tf.float16)
        seg = tf.constant(seg_ids)

        runs = []
        for _ in range(3):
            with tf.device('/GPU:0'):
                out = tf.math.unsorted_segment_sum(data_f16, seg, N).numpy()
            runs.append(out)

        diff01 = float(np.max(np.abs(runs[0].astype(np.float32) - runs[1].astype(np.float32))))
        diff02 = float(np.max(np.abs(runs[0].astype(np.float32) - runs[2].astype(np.float32))))
        print(f"M={M}, N={N}: run0_vs_run1={diff01:.3e}, run0_vs_run2={diff02:.3e}", end="")
        if diff01 > 0 or diff02 > 0:
            print("  *** GPU NON-DETERMINISTIC ***", end="")
        print()

print(f"\n=== BUG SUMMARY ===")
M, N = 100000, 500
data_np = rng.standard_normal(M).astype(np.float16)
seg_ids = rng.integers(0, N, size=M).astype(np.int32)
ref_np = np.zeros(N, dtype=np.float64)
for i in range(M):
    ref_np[seg_ids[i]] += data_np[i]

data_f16 = tf.constant(data_np, dtype=tf.float16)
seg = tf.constant(seg_ids)
cpu_out = tf.math.unsorted_segment_sum(data_f16, seg, N).numpy().astype(np.float64)
cpu_err = float(np.max(np.abs(cpu_out - ref_np)))
print(f"float16 unsorted_segment_sum M={M}, N={N}: CPU_err={cpu_err:.3e}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_out = tf.math.unsorted_segment_sum(data_f16, seg, N).numpy().astype(np.float64)
    gpu_err = float(np.max(np.abs(gpu_out - ref_np)))
    diff = float(np.max(np.abs(cpu_out - gpu_out)))
    ratio = gpu_err / (cpu_err + 1e-30)
    print(f"GPU_err={gpu_err:.3e}, CPU_vs_GPU={diff:.3e}")
    if ratio > 5 and diff > 1.0:
        print(f"*** GPU {ratio:.1f}x less accurate ***")
    elif diff > 1.0:
        print(f"*** SIGNIFICANT CPU/GPU DIVERGENCE ***")
    else:
        print("No significant divergence")
