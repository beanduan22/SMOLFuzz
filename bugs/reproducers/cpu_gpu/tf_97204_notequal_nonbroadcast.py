"""
Bug   : tf.raw_ops.NotEqual inconsistent on non-broadcastable shapes
Issue : https://github.com/tensorflow/tensorflow/issues/97204
Class : cpu_vs_gpu  (CPU silently returns scalar True; GPU raises)
Repro : x.shape=(4,1) and y.shape=(1,28,2,3,2) cannot broadcast.
        GPU correctly raises InvalidArgumentError; CPU returns
        tf.Tensor(True, shape=(), dtype=bool).

Variants:
  - dtype             : float32, float64, int32, int64, bool
  - shape combos      : non-broadcastable rank/dim combos
  - flag              : incompatible_shape_error True / False
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _run(name: str, x, y, flag: bool, dev: str):
    try:
        with tf.device(dev):
            out = tf.raw_ops.NotEqual(x=x, y=y, incompatible_shape_error=flag)
        return ("ok", tuple(out.shape), out.numpy().tolist() if out.shape == () else "(tensor)")
    except tf.errors.InvalidArgumentError as exc:
        return ("InvalidArgumentError", None, str(exc).splitlines()[0][:120])
    except Exception as exc:
        return (type(exc).__name__, None, str(exc)[:120])


def _compare(name: str, x, y, flag: bool):
    cpu = _run(name, x, y, flag, "CPU:0")
    gpu = _run(name, x, y, flag, "GPU:0")
    print(f"  [{name}] flag={flag}  cpu={cpu}  gpu={gpu}")


def variant_1_original() -> None:
    print("variant_1_original")
    np.random.seed(202)
    x = np.random.uniform(-32767.0, 127.0, size=(4, 1)).astype(np.float32)
    y = np.random.uniform(0.0, 89.0, size=(1, 28, 2, 3, 2)).astype(np.float32)
    _compare("original/false", tf.constant(x), tf.constant(y), False)
    _compare("original/true", tf.constant(x), tf.constant(y), True)


def variant_2_dtype_sweep() -> None:
    print("variant_2_dtype_sweep")
    for tf_dtype in (tf.float32, tf.float64, tf.int32, tf.int64, tf.bool):
        x = tf.zeros((4, 1), dtype=tf_dtype)
        y = tf.zeros((1, 28, 2, 3, 2), dtype=tf_dtype)
        _compare(f"dtype={tf_dtype.name}/false", x, y, False)


def variant_3_shape_combos() -> None:
    print("variant_3_shape_combos")
    pairs = [
        ((3, 5), (4, 5)),
        ((2, 3, 4), (5, 3, 4)),
        ((1, 2, 3), (1, 4, 3)),
        ((2, 1, 5), (3, 1, 5)),
        ((4,), (5, 6)),
    ]
    for sx, sy in pairs:
        x = tf.zeros(sx, dtype=tf.float32)
        y = tf.zeros(sy, dtype=tf.float32)
        _compare(f"shapes={sx} vs {sy}/false", x, y, False)


def main() -> None:
    print(f"tf={tf.__version__}")
    if not tf.config.list_physical_devices("GPU"):
        print("[skip] no GPU available")
        return
    variant_1_original()
    variant_2_dtype_sweep()
    variant_3_shape_combos()


if __name__ == "__main__":
    main()
