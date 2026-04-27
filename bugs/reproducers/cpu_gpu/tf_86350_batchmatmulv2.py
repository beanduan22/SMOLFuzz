"""
Bug   : tf.raw_ops.BatchMatMulV2 inconsistent CPU vs GPU
Issue : https://github.com/tensorflow/tensorflow/issues/86350
Class : cpu_vs_gpu
Repro : matmul of bfloat16 5-D and 4-D batched tensors disagrees beyond
        rtol=1e-3 / atol=1e-2 between CPU and GPU.

Variants:
  - dtype       : float32, float16, bfloat16
  - rank        : 4-D, 5-D, 6-D batched
  - transpose   : adj_x / adj_y combos
"""
from __future__ import annotations

import tensorflow as tf


def _diff(name: str, cpu, gpu, atol=1e-2, rtol=1e-3):
    cpu_f = tf.cast(cpu, tf.float32).numpy()
    gpu_f = tf.cast(gpu, tf.float32).numpy()
    diff = float(tf.reduce_max(tf.abs(cpu_f - gpu_f)))
    consistent = bool(tf.experimental.numpy.allclose(cpu_f, gpu_f, atol=atol, rtol=rtol))
    print(f"  [{name}] shape={tuple(cpu.shape)} dtype={cpu.dtype.name} max|diff|={diff:.3e} ok={consistent}")


def _bmm(x, y, adj_x: bool, adj_y: bool, dev: str):
    with tf.device(dev):
        return tf.raw_ops.BatchMatMulV2(x=x, y=y, adj_x=adj_x, adj_y=adj_y)


def variant_1_original_bf16() -> None:
    print("variant_1_original_bf16")
    x = tf.constant([
        [[[[ -1.3594, -0.3027], [-1.4141,  0.2969]],
          [[ -0.9141,  1.7812], [ 1.2266,  0.8594]]],
         [[[  0.8359, -0.9414], [-1.7969, -0.7461]],
          [[  0.3164,  0.3691], [ 0.7656,  0.2354]]]],
        [[[[ -0.5898,  1.3516], [ 0.4902, -0.1045]],
          [[ -0.1099,  1.5078], [ 0.2852, -0.0957]]],
         [[[-0.9883,  1.3203], [-0.2715, -1.7578]],
          [[ -0.1602, -0.4336], [-0.6875, -0.4492]]]],
    ], dtype=tf.bfloat16)
    y = tf.constant([
        [[[  0.6836, -0.6562], [-0.5508, -0.8438]],
         [[  1.6094, -0.9883], [-0.1318,  1.1094]]],
        [[[  0.4062, -1.1094], [-0.7188, -1.7578]],
         [[ -1.0391, -0.6602], [ 0.8359, -0.6562]]],
    ], dtype=tf.bfloat16)
    _diff("original", _bmm(x, y, False, False, "CPU:0"), _bmm(x, y, False, False, "GPU:0"))


def variant_2_dtype_sweep() -> None:
    print("variant_2_dtype_sweep")
    rng = tf.random.Generator.from_seed(31)
    for dtype in (tf.float32, tf.float16, tf.bfloat16):
        x = tf.cast(rng.normal((2, 3, 4, 5), dtype=tf.float32), dtype)
        y = tf.cast(rng.normal((2, 3, 5, 6), dtype=tf.float32), dtype)
        _diff(f"dtype={dtype.name}", _bmm(x, y, False, False, "CPU:0"), _bmm(x, y, False, False, "GPU:0"))


def variant_3_rank_sweep() -> None:
    print("variant_3_rank_sweep")
    rng = tf.random.Generator.from_seed(37)
    cases = [
        ((2, 4, 5), (2, 5, 6)),
        ((2, 3, 4, 5), (2, 3, 5, 6)),
        ((2, 3, 4, 4, 5), (2, 3, 4, 5, 6)),
        ((1, 2, 3, 4, 4, 5), (1, 2, 3, 4, 5, 6)),
    ]
    for sx, sy in cases:
        x = tf.cast(rng.normal(sx, dtype=tf.float32), tf.bfloat16)
        y = tf.cast(rng.normal(sy, dtype=tf.float32), tf.bfloat16)
        _diff(f"rank={len(sx)}", _bmm(x, y, False, False, "CPU:0"), _bmm(x, y, False, False, "GPU:0"))


def variant_4_transpose_combos() -> None:
    print("variant_4_transpose_combos")
    rng = tf.random.Generator.from_seed(41)
    for adj_x, adj_y in [(False, False), (True, False), (False, True), (True, True)]:
        sx = (2, 5, 4) if adj_x else (2, 4, 5)
        sy = (2, 6, 5) if adj_y else (2, 5, 6)
        x = tf.cast(rng.normal(sx, dtype=tf.float32), tf.bfloat16)
        y = tf.cast(rng.normal(sy, dtype=tf.float32), tf.bfloat16)
        _diff(f"adj_x={adj_x},adj_y={adj_y}", _bmm(x, y, adj_x, adj_y, "CPU:0"), _bmm(x, y, adj_x, adj_y, "GPU:0"))


def main() -> None:
    print(f"tf={tf.__version__}")
    if not tf.config.list_physical_devices("GPU"):
        print("[skip] no GPU available")
        return
    variant_1_original_bf16()
    variant_2_dtype_sweep()
    variant_3_rank_sweep()
    variant_4_transpose_combos()


if __name__ == "__main__":
    main()
