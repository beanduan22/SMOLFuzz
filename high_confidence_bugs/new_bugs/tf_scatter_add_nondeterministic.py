"""
Bug: tf.tensor_scatter_nd_add / tf.math.unsorted_segment_sum with duplicate
indices — GPU uses atomicAdd in arbitrary order → float16/bf16 accumulation
errors 20-40x larger than CPU.
Cross-framework port of pt_scatter_add_nondeterministic.py.

PyTorch confirmed: float16 GPU error = 102 vs CPU = 2.45 (41.9x worse).
TF equivalent: tf.math.unsorted_segment_sum and tf.scatter_nd.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow scatter_add / unsorted_segment_sum CPU vs GPU ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

rng = np.random.default_rng(42)

def run_scatter(dtype, N, M, device):
    src = rng.standard_normal(N).astype(np.float32) * (100 if dtype == tf.float16 else 1.0)
    if dtype == tf.float16:
        src = src.astype(np.float16)
        dtype_np = np.float16
    elif dtype == tf.bfloat16:
        # bfloat16 via float32
        dtype_np = np.float32
    else:
        dtype_np = np.float32

    indices = rng.integers(0, M, N, dtype=np.int32)
    src_tf = tf.constant(src, dtype=dtype)
    indices_tf = tf.constant(indices, dtype=tf.int32)

    with tf.device(f'/{device}:0'):
        result = tf.math.unsorted_segment_sum(src_tf, indices_tf, num_segments=M)
    return result.numpy()

# Float32 reference using float64 numpy
def reference_scatter(src_f32, indices, M):
    out = np.zeros(M, dtype=np.float64)
    src_f64 = src_f32.astype(np.float64)
    for i, idx in enumerate(indices):
        out[idx] += src_f64[i]
    return out

# Test 1: float16
print("\n--- Test 1: float16 unsorted_segment_sum ---")
for N, M in [(100_000, 50), (500_000, 100)]:
    src_np = (rng.standard_normal(N) * 100).astype(np.float16)
    indices_np = rng.integers(0, M, N, dtype=np.int32)
    src_tf = tf.constant(src_np, dtype=tf.float16)
    idx_tf = tf.constant(indices_np, dtype=tf.int32)

    ref = reference_scatter(src_np.astype(np.float32), indices_np, M)
    cpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
    cpu_err = float(np.max(np.abs(cpu_out.astype(np.float64) - ref)))

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
        gpu_err = float(np.max(np.abs(gpu_out.astype(np.float64) - ref)))
        cpu_gpu_diff = float(np.max(np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))))
        ratio = gpu_err / (cpu_err + 1e-30)
        print(f"N={N}, M={M}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")
        if cpu_gpu_diff > 0.1:
            print(f"  *** SIGNIFICANT CPU/GPU DIVERGENCE ***")

# Test 2: bfloat16
print("\n--- Test 2: bfloat16 unsorted_segment_sum ---")
N, M = 200_000, 50
src_np = (rng.standard_normal(N) * 1000).astype(np.float32)
indices_np = rng.integers(0, M, N, dtype=np.int32)
src_bf16 = tf.constant(src_np, dtype=tf.bfloat16)
idx_tf = tf.constant(indices_np, dtype=tf.int32)

ref = reference_scatter(src_np, indices_np, M)
cpu_out_bf16 = tf.math.unsorted_segment_sum(src_bf16, idx_tf, num_segments=M)
cpu_out = tf.cast(cpu_out_bf16, tf.float32).numpy()
cpu_err = float(np.max(np.abs(cpu_out.astype(np.float64) - ref)))

if gpus:
    with tf.device('/GPU:0'):
        gpu_out_bf16 = tf.math.unsorted_segment_sum(src_bf16, idx_tf, num_segments=M)
    gpu_out = tf.cast(gpu_out_bf16, tf.float32).numpy()
    gpu_err = float(np.max(np.abs(gpu_out.astype(np.float64) - ref)))
    cpu_gpu_diff = float(np.max(np.abs(cpu_out - gpu_out)))
    ratio = gpu_err / (cpu_err + 1e-30)
    print(f"N={N}, M={M}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")
    if cpu_gpu_diff > 1.0:
        print(f"  *** SIGNIFICANT CPU/GPU DIVERGENCE ***")

# Test 3: float32 (reference)
print("\n--- Test 3: float32 (should be small divergence) ---")
for N, M in [(1_000_000, 100), (500_000, 10)]:
    src_np = rng.standard_normal(N).astype(np.float32)
    indices_np = rng.integers(0, M, N, dtype=np.int32)
    src_tf = tf.constant(src_np, dtype=tf.float32)
    idx_tf = tf.constant(indices_np, dtype=tf.int32)

    ref = reference_scatter(src_np, indices_np, M)
    cpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
    cpu_err = float(np.max(np.abs(cpu_out.astype(np.float64) - ref)))

    if gpus:
        with tf.device('/GPU:0'):
            gpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
        gpu_err = float(np.max(np.abs(gpu_out.astype(np.float64) - ref)))
        cpu_gpu_diff = float(np.max(np.abs(cpu_out - gpu_out)))
        ratio = gpu_err / (cpu_err + 1e-30)
        print(f"N={N}, M={M}: CPU_err={cpu_err:.3e}, GPU_err={gpu_err:.3e}, CPU_vs_GPU={cpu_gpu_diff:.3e} (ratio={ratio:.1f}x)")

# Summary
print(f"\n=== BUG SUMMARY ===")
N, M = 100_000, 50
src_np = (rng.standard_normal(N) * 100).astype(np.float16)
indices_np = rng.integers(0, M, N, dtype=np.int32)
src_tf = tf.constant(src_np, dtype=tf.float16)
idx_tf = tf.constant(indices_np, dtype=tf.int32)
ref = reference_scatter(src_np.astype(np.float32), indices_np, M)
cpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
cpu_err = float(np.max(np.abs(cpu_out.astype(np.float64) - ref)))
print(f"float16 N={N}, M={M} ({N//M:.0f} dups/bucket):")
print(f"  CPU max error vs float64: {cpu_err:.3e}")
if gpus:
    with tf.device('/GPU:0'):
        gpu_out = tf.math.unsorted_segment_sum(src_tf, idx_tf, num_segments=M).numpy()
    gpu_err = float(np.max(np.abs(gpu_out.astype(np.float64) - ref)))
    cpu_gpu_diff = float(np.max(np.abs(cpu_out.astype(np.float32) - gpu_out.astype(np.float32))))
    ratio = gpu_err / (cpu_err + 1e-30)
    print(f"  GPU max error vs float64: {gpu_err:.3e}")
    print(f"  CPU vs GPU max diff: {cpu_gpu_diff:.3e}")
    print(f"  GPU/CPU ratio: {ratio:.1f}x")
else:
    print("  No GPU available")
