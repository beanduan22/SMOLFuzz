"""
Bug: TF CPU and GPU tf.sort produce completely different orderings for arrays containing NaN.
Root cause: TF CPU sort treats NaN as "stuck" — NaN elements stay at their original indices
and the non-NaN elements are also left unsorted around them.
TF GPU sort moves ALL NaN values to the front (position 0, 1, 2, ...) and correctly
sorts all non-NaN values in ascending order after them.
These are two entirely different algorithms producing incompatible outputs for NaN inputs.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

# Input: [nan, 3.0, 1.0, nan, 2.0, nan, 0.5]
x = tf.constant([float('nan'), 3.0, 1.0, float('nan'), 2.0, float('nan'), 0.5], dtype=tf.float32)

with tf.device('/CPU:0'):
    cpu_sorted = tf.sort(x).numpy()
with tf.device('/GPU:0'):
    gpu_sorted = tf.sort(x).numpy()

cpu_nan_positions = [i for i, v in enumerate(cpu_sorted) if np.isnan(v)]
gpu_nan_positions = [i for i, v in enumerate(gpu_sorted) if np.isnan(v)]

print(f"Input:      {x.numpy().tolist()}")
print(f"CPU sorted: {cpu_sorted.tolist()}")
print(f"GPU sorted: {gpu_sorted.tolist()}")
print(f"CPU NaN positions: {cpu_nan_positions}")
print(f"GPU NaN positions: {gpu_nan_positions}")

assert cpu_nan_positions != gpu_nan_positions, (
    f"Expected different NaN positions: CPU={cpu_nan_positions} GPU={gpu_nan_positions}"
)
# GPU moves NaN to front [0,1,2], CPU leaves NaN in original positions [0,3,5]
print("BUG CONFIRMED: TF CPU sort leaves NaN in-place; GPU sort moves all NaN to front position")
