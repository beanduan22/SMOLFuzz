"""
Bug   : tf.io.encode_png / tf.compat.v1.image.encode_png aborts on illegal image
Issue : https://github.com/tensorflow/tensorflow/issues/76726
Class : crash  (core dump / SIGABRT)
Repro : tf.tile(..., [0, 0, 1]) makes a zero-sized image, and encode_png
        receives an empty buffer that triggers a hard abort inside libpng.

Each variant runs in its own subprocess.
Variants exercise:
  - tile multiplier      : zero-axis variants
  - dtype / channels     : uint8 1-ch / 3-ch / 4-ch
  - explicit empty       : tf.zeros((0, W, C)) and tf.zeros((H, 0, C))
  - compression flag     : -1, 0, 9
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(name: str, body: str) -> None:
    body = textwrap.dedent(body).strip("\n")
    code = "import tensorflow as tf\n" + body + '\nprint("OK", flush=True)\n'
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode == 0 and proc.stdout.strip().endswith("OK"):
        status = "ok"
    elif proc.returncode == -6:
        status = "SIGABRT"
    elif proc.returncode == -11:
        status = "SIGSEGV"
    elif proc.returncode == -8:
        status = "SIGFPE"
    elif proc.returncode != 0:
        status = f"exit={proc.returncode}"
    else:
        status = "unexpected"
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    print(f"  [{name}] {status}  | {tail[0][:160]}")


def variant_1_minimal() -> None:
    print("variant_1_minimal")
    _run("tile_zero/v1.encode_png", """
    image = tf.cast(tf.tile([[[0, 0, 0, 1]], [[0, 0, 1, 0]]], [0, 0, 1]), tf.uint8)
    tf.compat.v1.image.encode_png(image)
    """)


def variant_2_compression_flag() -> None:
    print("variant_2_compression_flag")
    for c in ("-1", "0", "9"):
        _run(f"tile_zero/encode_png/comp={c}", f"""
        image = tf.cast(tf.tile([[[0, 0, 0, 1]], [[0, 0, 1, 0]]], [0, 0, 1]), tf.uint8)
        tf.io.encode_png(image, compression={c}, name=None)
        """)


def variant_3_explicit_empty_shapes() -> None:
    print("variant_3_explicit_empty_shapes")
    cases = [
        ("0xWxC", "tf.zeros((0, 4, 3), dtype=tf.uint8)"),
        ("HxWx0", "tf.zeros((4, 4, 0), dtype=tf.uint8)"),
        ("Hx0xC", "tf.zeros((4, 0, 3), dtype=tf.uint8)"),
        ("0x0x0", "tf.zeros((0, 0, 0), dtype=tf.uint8)"),
    ]
    for name, expr in cases:
        _run(f"empty/{name}", f"""
        image = {expr}
        tf.io.encode_png(image)
        """)


def variant_4_channel_sweep() -> None:
    print("variant_4_channel_sweep")
    for ch in (1, 3, 4):
        _run(f"empty/Hx0x{ch}", f"""
        image = tf.zeros((4, 0, {ch}), dtype=tf.uint8)
        tf.io.encode_png(image)
        """)


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    variant_1_minimal()
    variant_2_compression_flag()
    variant_3_explicit_empty_shapes()
    variant_4_channel_sweep()


if __name__ == "__main__":
    main()
