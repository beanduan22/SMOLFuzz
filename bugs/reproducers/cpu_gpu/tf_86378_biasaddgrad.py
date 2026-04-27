"""
Bug   : tf.raw_ops.BiasAddGrad produces inconsistent results between CPU and GPU
Issue : https://github.com/tensorflow/tensorflow/issues/86378
Class : cpu_vs_gpu
Repro : reduce-sum reduction over channels disagrees on CPU vs GPU for
        bfloat16 NCHW out_backprop.

Variants:
  - dtype sweep    : float32, float16, bfloat16
  - data_format    : NCHW, NHWC
  - shape sweep    : 4-D (NHWC/NCHW) and 5-D (NCDHW)
"""
from __future__ import annotations

import tensorflow as tf


def _diff(name: str, cpu, gpu):
    cpu_f = tf.cast(cpu, tf.float32).numpy()
    gpu_f = tf.cast(gpu, tf.float32).numpy()
    diff = float(tf.reduce_max(tf.abs(cpu_f - gpu_f)))
    print(f"  [{name}] shape={cpu.shape} cpu_dtype={cpu.dtype.name} max|diff|={diff:.3e}")


def _bias_add_grad(x, fmt: str, dev: str):
    with tf.device(dev):
        return tf.raw_ops.BiasAddGrad(out_backprop=x, data_format=fmt)


def variant_1_original_bf16_nchw() -> None:
    print("variant_1_original_bf16_nchw")
    out_backprop = tf.constant([
        [[[[ 0.2207,  2.1094], [-0.3730, -1.0625], [ 1.7031,  0.7148]],
          [[ 1.5078, -0.6719], [-0.6367,  0.5039], [-2.3281,  0.5078]]],
         [[[-0.3574,  0.0461], [ 2.3750, -2.9688], [-0.5703, -2.0156]],
          [[ 0.8125,  1.7656], [-0.9570,  0.6250], [-0.6914, -0.4746]]],
         [[[-0.3750, -0.7383], [ 0.3691,  0.4570], [ 1.1641,  0.2715]],
          [[-1.2969, -0.9844], [-0.4863,  1.0938], [-1.4297,  0.8086]]]],
        [[[[ 0.3730,  0.8477], [-0.3887,  1.2266], [ 0.0859, -0.5742]],
          [[-0.7383, -0.2432], [-0.7578, -0.8281], [-0.1660, -0.9336]]],
         [[[ 1.4297,  0.6797], [-1.6172,  0.4941], [-0.3047, -0.3711]],
          [[-0.6250, -0.7617], [ 0.9453,  0.1064], [ 1.4062, -2.9531]]],
         [[[-1.4297, -0.1387], [ 0.0625,  1.0469], [-0.1953,  1.6406]],
          [[-0.3047,  0.5117], [ 1.8125,  1.1797], [-0.8789, -0.4688]]]],
    ], dtype=tf.bfloat16)
    cpu = _bias_add_grad(out_backprop, "NCHW", "CPU:0")
    gpu = _bias_add_grad(out_backprop, "NCHW", "GPU:0")
    _diff("original/bf16/NCHW", cpu, gpu)


def variant_2_dtype_sweep_nhwc() -> None:
    print("variant_2_dtype_sweep_nhwc")
    rng = tf.random.Generator.from_seed(7)
    for dtype in (tf.float32, tf.float16, tf.bfloat16):
        x = tf.cast(rng.normal((2, 4, 4, 8), dtype=tf.float32), dtype)
        cpu = _bias_add_grad(x, "NHWC", "CPU:0")
        gpu = _bias_add_grad(x, "NHWC", "GPU:0")
        _diff(f"dtype={dtype.name}/NHWC", cpu, gpu)


def variant_3_shape_sweep_nchw() -> None:
    print("variant_3_shape_sweep_nchw")
    rng = tf.random.Generator.from_seed(11)
    shapes = [(1, 1, 8, 8), (4, 16, 7, 11), (2, 32, 3, 3), (1, 3, 1, 1)]
    for shp in shapes:
        x = tf.cast(rng.normal(shp, dtype=tf.float32), tf.bfloat16)
        cpu = _bias_add_grad(x, "NCHW", "CPU:0")
        gpu = _bias_add_grad(x, "NCHW", "GPU:0")
        _diff(f"shape={shp}/bf16/NCHW", cpu, gpu)


def main() -> None:
    print(f"tf={tf.__version__}")
    if not tf.config.list_physical_devices("GPU"):
        print("[skip] no GPU available")
        return
    variant_1_original_bf16_nchw()
    variant_2_dtype_sweep_nhwc()
    variant_3_shape_sweep_nchw()


if __name__ == "__main__":
    main()
