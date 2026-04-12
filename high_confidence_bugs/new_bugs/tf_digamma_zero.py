"""
Bug: tf.math.digamma(0) returns NaN instead of -inf.
Source: TensorFlow GitHub issue #111917
Root cause: The digamma function ψ(x) has a pole at x=0, so ψ(0) = -inf.
PyTorch, SciPy, and Mathematica all return -inf.
TensorFlow returns NaN — mathematically wrong.

Also tests: CPU vs GPU behavior and cross-framework comparison.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

print("=== TensorFlow tf.math.digamma(0) returns NaN instead of -inf ===")
print(f"TF version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

# Test 1: Digamma at poles and near-poles
print("\n--- digamma at poles and near-zero ---")
test_values = [0.0, -1.0, -2.0, -3.0,  # poles: ψ(0), ψ(-1), ψ(-2), ψ(-3) = -inf
               1e-38, 1e-7, 0.5, 1.0, 2.0]  # non-poles

for x_val in test_values:
    x_tf = tf.constant([x_val], dtype=tf.float32)
    cpu_result = tf.math.digamma(x_tf).numpy()[0]

    # PyTorch reference
    try:
        import torch
        pt_result = torch.digamma(torch.tensor([x_val], dtype=torch.float32)).item()
    except:
        pt_result = None

    # SciPy reference
    try:
        from scipy.special import digamma as scipy_digamma
        scipy_result = scipy_digamma(x_val)
    except:
        scipy_result = None

    line = f"digamma({x_val}): TF_CPU={cpu_result:.4g}"
    if pt_result is not None:
        line += f"  PT={pt_result:.4g}"
    if scipy_result is not None:
        line += f"  SciPy={scipy_result:.4g}"

    if gpus:
        with tf.device('/GPU:0'):
            gpu_result = tf.math.digamma(x_tf).numpy()[0]
        line += f"  TF_GPU={gpu_result:.4g}"

    # Flag bugs
    if np.isnan(cpu_result) and (np.isinf(scipy_result if scipy_result is not None else 0)):
        line += "  *** BUG: TF_CPU=NaN, expected -inf ***"
    elif np.isnan(cpu_result) and pt_result is not None and np.isinf(pt_result):
        line += "  *** BUG: TF_CPU=NaN, PT=-inf ***"

    print(line)

# Test 2: Batch of poles
print("\n--- Batch: all integer poles ---")
x_poles = tf.constant([0.0, -1.0, -2.0, -3.0, -4.0, -5.0], dtype=tf.float32)
cpu_poles = tf.math.digamma(x_poles).numpy()
print(f"TF CPU:  {cpu_poles.tolist()}")
print(f"Expected: all -inf")
cpu_should_be_neginf = np.all(np.isneginf(cpu_poles))
cpu_has_nan = np.any(np.isnan(cpu_poles))
print(f"CPU all -inf: {cpu_should_be_neginf}, CPU has NaN: {cpu_has_nan}")
if cpu_has_nan:
    print(f"*** BUG: TF returns NaN at poles ***")

if gpus:
    with tf.device('/GPU:0'):
        gpu_poles = tf.math.digamma(x_poles).numpy()
    print(f"TF GPU:  {gpu_poles.tolist()}")
    if not np.array_equal(np.isnan(cpu_poles), np.isnan(gpu_poles)):
        print(f"*** CPU/GPU DIVERGENCE in NaN pattern ***")

# Test 3: Float64 behavior
print("\n--- float64 digamma(0) ---")
x64 = tf.constant([0.0], dtype=tf.float64)
try:
    result64 = tf.math.digamma(x64).numpy()[0]
    print(f"float64 digamma(0) = {result64}")
    if np.isnan(result64):
        print(f"*** BUG: float64 also returns NaN ***")
except Exception as e:
    print(f"float64 error: {e}")

# Test 4: Compare with PyTorch at all poles
print("\n--- Cross-framework comparison at poles ---")
try:
    import torch
    poles = [0.0, -1.0, -2.0, -3.0]
    print(f"{'x':>6}  {'TF_CPU':>12}  {'PyTorch_CPU':>12}  {'Bug':}")
    for x_val in poles:
        tf_val = float(tf.math.digamma(tf.constant([x_val], dtype=tf.float32)).numpy()[0])
        pt_val = torch.digamma(torch.tensor([x_val], dtype=torch.float32)).item()
        bug = "*** TF=NaN, PT=-inf ***" if np.isnan(tf_val) and np.isinf(pt_val) else ""
        print(f"{x_val:>6.1f}  {tf_val:>12.4g}  {pt_val:>12.4g}  {bug}")
except ImportError:
    print("PyTorch not available for comparison")

print(f"\n=== BUG SUMMARY ===")
result = float(tf.math.digamma(tf.constant([0.0], dtype=tf.float32)).numpy()[0])
print(f"tf.math.digamma(0.0) = {result}")
print(f"Expected:             = -inf")
print(f"Is NaN: {np.isnan(result)}")
if np.isnan(result):
    print("*** BUG CONFIRMED: TF digamma(0) = NaN, should be -inf ***")
if gpus:
    with tf.device('/GPU:0'):
        gpu_result = float(tf.math.digamma(tf.constant([0.0], dtype=tf.float32)).numpy()[0])
    print(f"TF GPU digamma(0.0) = {gpu_result}")
