"""
Bug   : torch.arange triggers SIGFPE when writing float results to int64 out tensor
Issue : https://github.com/pytorch/pytorch/issues/173574
Class : crash  (Floating Point Exception / SIGFPE, instead of RuntimeError)
Repro : float (start, end, step) with an int64 `out=` tensor crashes the
        interpreter on PyTorch 2.10.

Each variant runs in its own subprocess.
Variants exercise:
  - out dtype           : int64, int32, int16, int8, uint8
  - step combo           : float step / int step / fractional step
  - start sign           : negative / positive ranges
  - layout                : default (None) and explicit torch.strided
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(name: str, body: str) -> None:
    body = textwrap.dedent(body).strip("\n")
    code = "import torch\n" + body + '\nprint("OK", flush=True)\n'
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode == 0 and proc.stdout.strip().endswith("OK"):
        status = "ok"
    elif proc.returncode == -8:
        status = "SIGFPE"
    elif proc.returncode == -11:
        status = "SIGSEGV"
    elif proc.returncode == -6:
        status = "SIGABRT"
    elif proc.returncode != 0:
        status = f"exit={proc.returncode}"
    else:
        status = "unexpected"
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    print(f"  [{name}] {status}  | {tail[0][:160]}")


def variant_1_minimal() -> None:
    print("variant_1_minimal")
    _run("int64/float-step", """
    out = torch.full((10,), 2, dtype=torch.int64)
    torch.arange(2.0, 7.0, 0.5, out=out)
    """)


def variant_2_out_dtype_sweep() -> None:
    print("variant_2_out_dtype_sweep")
    for dt in ("int8", "int16", "int32", "int64", "uint8"):
        _run(f"out={dt}/float-step", f"""
        out = torch.zeros((10,), dtype=torch.{dt})
        torch.arange(2.0, 7.0, 0.5, out=out)
        """)


def variant_3_step_combos() -> None:
    print("variant_3_step_combos")
    cases = [
        ("step=1.0", "torch.arange(0.0, 5.0, 1.0, out=out)"),
        ("step=0.25", "torch.arange(0.0, 1.0, 0.25, out=out)"),
        ("step=int", "torch.arange(0.0, 5.0, 1,   out=out)"),
        ("step=fractional", "torch.arange(0.0, 1.0, 0.1, out=out)"),
        ("neg-range", "torch.arange(-2.0, 2.0, 0.5, out=out)"),
    ]
    for name, expr in cases:
        _run(name, f"""
        out = torch.zeros((20,), dtype=torch.int64)
        {expr}
        """)


def variant_4_layout_sweep() -> None:
    print("variant_4_layout_sweep")
    for layout in ("None", "torch.strided"):
        _run(f"layout={layout}", f"""
        out = torch.zeros((10,), dtype=torch.int64)
        torch.arange(2.0, 7.0, 0.5, layout={layout}, out=out)
        """)


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    variant_1_minimal()
    variant_2_out_dtype_sweep()
    variant_3_step_combos()
    variant_4_layout_sweep()


if __name__ == "__main__":
    main()
