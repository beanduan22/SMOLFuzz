"""
Bug   : tf.image.adjust_hue produces inconsistent results between CPU and GPU
Issue : https://github.com/tensorflow/tensorflow/issues/86256
Class : cpu_vs_gpu  (silent wrong result, max|diff| ~ 0.34)
Repro : single hue delta applied to the same RGB tensor produces visibly
        different output values on the second pixel of the second row.

Variants:
  - dtype          : float32, float16, bfloat16
  - shape          : (H,W,3), (B,H,W,3), tall vs wide tensors
  - delta          : negative / positive / zero / out-of-range
"""
from __future__ import annotations

import tensorflow as tf


def _diff(name: str, cpu, gpu, atol=1e-5, rtol=1e-6):
    cpu_f = tf.cast(cpu, tf.float32).numpy()
    gpu_f = tf.cast(gpu, tf.float32).numpy()
    diff = float(tf.reduce_max(tf.abs(cpu_f - gpu_f)))
    consistent = bool(tf.experimental.numpy.allclose(cpu_f, gpu_f, atol=atol, rtol=rtol))
    print(f"  [{name}] shape={tuple(cpu.shape)} dtype={cpu.dtype.name} max|diff|={diff:.3e} consistent={consistent}")


def _adjust(images, delta, dev: str):
    with tf.device(dev):
        return tf.image.adjust_hue(images, delta)


def variant_1_original() -> None:
    print("variant_1_original")
    images = tf.constant([
        [[ 1.9720840,  2.1302242, -0.1902120],
         [ 0.6557856, -1.3016001,  1.1452782]],
        [[-2.2193234,  0.3198028,  0.9568117],
         [-0.3937407, -0.0503466, -0.3693791]],
    ], dtype=tf.float32)
    delta = tf.constant(-0.7441734, dtype=tf.float32)
    _diff("original", _adjust(images, delta, "CPU:0"), _adjust(images, delta, "GPU:0"))


def variant_2_dtype_sweep() -> None:
    print("variant_2_dtype_sweep")
    rng = tf.random.Generator.from_seed(13)
    base = rng.uniform((4, 4, 3), -2.0, 2.0)
    for dtype in (tf.float32, tf.float16, tf.bfloat16):
        x = tf.cast(base, dtype)
        d = tf.cast(0.31415, dtype)
        _diff(f"dtype={dtype.name}", _adjust(x, d, "CPU:0"), _adjust(x, d, "GPU:0"))


def variant_3_batch_and_aspect() -> None:
    print("variant_3_batch_and_aspect")
    rng = tf.random.Generator.from_seed(17)
    for shp in [(1, 8, 8, 3), (4, 16, 16, 3), (2, 32, 4, 3), (1, 4, 64, 3)]:
        x = rng.uniform(shp, 0.0, 1.0)
        d = tf.constant(-0.5, dtype=tf.float32)
        _diff(f"shape={shp}", _adjust(x, d, "CPU:0"), _adjust(x, d, "GPU:0"))


def variant_4_delta_sweep() -> None:
    print("variant_4_delta_sweep")
    rng = tf.random.Generator.from_seed(23)
    x = rng.uniform((6, 6, 3), 0.0, 1.0)
    for delta_val in (-1.5, -0.5, 0.0, 0.25, 0.99, 1.5):
        d = tf.constant(delta_val, dtype=tf.float32)
        _diff(f"delta={delta_val:+.2f}", _adjust(x, d, "CPU:0"), _adjust(x, d, "GPU:0"))


def main() -> None:
    print(f"tf={tf.__version__}")
    if not tf.config.list_physical_devices("GPU"):
        print("[skip] no GPU available")
        return
    variant_1_original()
    variant_2_dtype_sweep()
    variant_3_batch_and_aspect()
    variant_4_delta_sweep()


if __name__ == "__main__":
    main()
