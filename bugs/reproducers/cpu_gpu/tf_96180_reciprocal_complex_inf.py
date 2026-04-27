"""
Bug   : tf.math.reciprocal inconsistent on complex128 inf between CPU and GPU
Issue : https://github.com/tensorflow/tensorflow/issues/96180
Class : cpu_vs_gpu  (silent wrong result; partial NaN-poisoning on CPU)
Repro : reciprocal of a complex128 tensor containing inf returns NaN+NaNj on
        CPU but 0+0j on GPU. Once a NaN appears in the same vector, every CPU
        result becomes NaN (poisoning) while GPU still returns 0.

Variants:
  - dtype          : complex64, complex128
  - input pattern  : pure inf, mixed inf+nan, signed inf
  - shape          : column vector / row vector / 2-D / batched
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def _show(name: str, cpu, gpu):
    cpu_arr = cpu.numpy()
    gpu_arr = gpu.numpy()
    cpu_nan = int(np.isnan(cpu_arr.real).sum() + np.isnan(cpu_arr.imag).sum())
    gpu_nan = int(np.isnan(gpu_arr.real).sum() + np.isnan(gpu_arr.imag).sum())
    print(f"  [{name}] shape={cpu.shape} dtype={cpu.dtype.name}  cpu_NaNs={cpu_nan} gpu_NaNs={gpu_nan}")
    print(f"    cpu={cpu_arr.flatten().tolist()[:6]}")
    print(f"    gpu={gpu_arr.flatten().tolist()[:6]}")


def _recip(arr, dtype, dev: str):
    with tf.device(dev):
        return tf.math.reciprocal(tf.constant(arr, dtype=dtype))


def variant_1_original_inf() -> None:
    print("variant_1_original_inf")
    data = np.array([[0.0 + np.inf], [np.inf + 0.0], [np.inf + np.inf]], dtype="complex128")
    _show("original/complex128", _recip(data, tf.complex128, "CPU:0"), _recip(data, tf.complex128, "GPU:0"))


def variant_2_inf_plus_nan() -> None:
    print("variant_2_inf_plus_nan")
    data = np.array([[0.0 + np.inf], [np.inf + 0.0], [np.inf + np.inf], [np.nan + 0.0]], dtype="complex128")
    _show("inf+nan/complex128", _recip(data, tf.complex128, "CPU:0"), _recip(data, tf.complex128, "GPU:0"))


def variant_3_dtype_sweep() -> None:
    print("variant_3_dtype_sweep")
    for tf_dtype, np_dtype in [(tf.complex64, "complex64"), (tf.complex128, "complex128")]:
        data = np.array([[np.inf + 1j], [1j * np.inf], [np.inf + np.inf * 1j]], dtype=np_dtype)
        _show(f"dtype={tf_dtype.name}", _recip(data, tf_dtype, "CPU:0"), _recip(data, tf_dtype, "GPU:0"))


def variant_4_signed_inf_2d() -> None:
    print("variant_4_signed_inf_2d")
    data = np.array([
        [np.inf + 0j, -np.inf + 0j, 1.0 + 0j],
        [0 + np.inf * 1j, 0 - np.inf * 1j, 0 + 1j],
        [np.inf - np.inf * 1j, -np.inf + np.inf * 1j, 1 + 1j],
    ], dtype="complex128")
    _show("signed_inf_2d", _recip(data, tf.complex128, "CPU:0"), _recip(data, tf.complex128, "GPU:0"))


def variant_5_batched() -> None:
    print("variant_5_batched")
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 4, 2)) + 1j * rng.standard_normal((3, 4, 2))
    base.flat[::5] = np.inf
    base.flat[7::11] = np.nan
    data = base.astype("complex128")
    _show("batched/complex128", _recip(data, tf.complex128, "CPU:0"), _recip(data, tf.complex128, "GPU:0"))


def main() -> None:
    print(f"tf={tf.__version__}")
    if not tf.config.list_physical_devices("GPU"):
        print("[skip] no GPU available")
        return
    variant_1_original_inf()
    variant_2_inf_plus_nan()
    variant_3_dtype_sweep()
    variant_4_signed_inf_2d()
    variant_5_batched()


if __name__ == "__main__":
    main()
