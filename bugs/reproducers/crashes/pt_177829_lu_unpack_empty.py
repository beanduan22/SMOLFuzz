"""
Bug   : torch.lu_unpack crashes when LU_pivots is an empty tensor
Issue : https://github.com/pytorch/pytorch/issues/177829
Class : crash  (segmentation fault)
Repro : an empty int32 LU_pivots tensor combined with a 3x3 LU_data tensor
        triggers a hard SIGSEGV on PyTorch 2.10.

Each variant runs in its own subprocess so the harness survives the crash.
Variants exercise:
  - LU_pivots dtype       : int32, int64
  - LU_pivots shape       : (0,), (0, 0), (1, 0)
  - LU_data shape          : 3x3, 4x4, batched 2x3x3
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import textwrap


def _run_isolated(name: str, body: str) -> None:
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
    elif proc.returncode == -11 or "segmentation fault" in (proc.stderr or "").lower():
        status = "SIGSEGV"
    elif proc.returncode == -6:
        status = "SIGABRT"
    elif proc.returncode == -8:
        status = "SIGFPE"
    elif proc.returncode != 0:
        status = f"exit={proc.returncode}"
    else:
        status = "unexpected"
    last = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    print(f"  [{name}] {status}  | {last[0][:160]}")


def variant_1_minimal() -> None:
    print("variant_1_minimal")
    _run_isolated("3x3/int32/empty(0,)", """
    LU_data = torch.tensor([[2., 3., 1.],
                            [0.5, 1., 2.],
                            [0.25, 0.5, 1.]])
    LU_pivots = torch.tensor([], dtype=torch.int32)
    P, L, U = torch.lu_unpack(LU_data, LU_pivots)
    """)


def variant_2_pivot_dtype_sweep() -> None:
    print("variant_2_pivot_dtype_sweep")
    for dt in ("int32", "int64"):
        _run_isolated(f"3x3/{dt}/empty(0,)", f"""
        LU_data = torch.tensor([[2., 3., 1.],
                                [0.5, 1., 2.],
                                [0.25, 0.5, 1.]])
        LU_pivots = torch.tensor([], dtype=torch.{dt})
        torch.lu_unpack(LU_data, LU_pivots)
        """)


def variant_3_pivot_shape_sweep() -> None:
    print("variant_3_pivot_shape_sweep")
    for shape_lit in ("(0,)", "(0, 0)", "(1, 0)"):
        _run_isolated(f"3x3/int32/empty{shape_lit}", f"""
        LU_data = torch.tensor([[2., 3., 1.],
                                [0.5, 1., 2.],
                                [0.25, 0.5, 1.]])
        LU_pivots = torch.empty({shape_lit}, dtype=torch.int32)
        torch.lu_unpack(LU_data, LU_pivots)
        """)


def variant_4_lu_data_sweep() -> None:
    print("variant_4_lu_data_sweep")
    cases = [
        ("4x4", "torch.eye(4)"),
        ("batched_2x3x3", "torch.stack([torch.eye(3)]*2)"),
        ("rect_3x4", "torch.randn(3, 4)"),
    ]
    for name, lu_expr in cases:
        _run_isolated(f"lu_data={name}/int32/empty(0,)", f"""
        LU_data = {lu_expr}
        LU_pivots = torch.tensor([], dtype=torch.int32)
        torch.lu_unpack(LU_data, LU_pivots)
        """)


def main() -> None:
    print(f"python={sys.version.split()[0]}  argv0={shlex.quote(sys.executable)}")
    variant_1_minimal()
    variant_2_pivot_dtype_sweep()
    variant_3_pivot_shape_sweep()
    variant_4_lu_data_sweep()


if __name__ == "__main__":
    main()
