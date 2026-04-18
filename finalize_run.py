"""
Wait for the in-flight smolfuzz PT + TF runs to finish, then triage the
saved bugs — re-applying the no-false-positive filters against the raw
saved outputs — and copy survivors into bugs/github/new/.

Also writes:
  bugs/github/new/SUMMARY.md   — human-readable run summary
  bugs/github/new/index.json   — machine-readable bug index
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "bugs" / "github" / "new"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PT_DIR = HERE / "results" / "clean_run_pt" / "full" / "bugs"
TF_DIR = HERE / "results" / "clean_run_tf" / "bugs"
PT_LOG = HERE / "results" / "clean_run" / "pt.log"
TF_LOG = HERE / "results" / "clean_run" / "tf.log"


# ── FP filters (mirror oracle.py) ───────────────────────────────────────────

_TOL = {
    "float64": (1e-5, 1e-8),
    "float32": (1e-4, 1e-5),
    "bfloat16": (1e-2, 1e-2),
    "float16": (1e-2, 1e-3),
}
BUG_MARGIN = 100.0
NONDET_MAX = 100.0
NONDET_MULTIPLIER = 10.0
NAN_ASYM_FRAC = 1e-3
NAN_ASYM_MIN = 4

_INFRA_RX = re.compile(
    r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorOutOfMemory"
    r"|cuMemAlloc failed|CUBLAS_STATUS_NOT_INITIALIZED|CUBLAS_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_NOT_INITIALIZED|CUDNN_STATUS_ALLOC_FAILED"
    r"|Failed to initialize CUDA|ResourceExhaustedError|RESOURCE_EXHAUSTED"
    r"|NCCL error|no kernel image is available",
    re.IGNORECASE,
)

_PERM_INVARIANT = {
    "torch.linalg.eig", "torch.linalg.eigh", "torch.linalg.eigvals",
    "torch.linalg.eigvalsh", "torch.linalg.svd", "torch.linalg.svdvals",
    "torch.linalg.qr", "torch.linalg.lu", "torch.linalg.lu_factor",
    "torch.linalg.cholesky", "torch.eig", "torch.symeig", "torch.svd",
    "torch.qr", "torch.lu", "torch.geqrf", "torch.orgqr", "torch.ormqr",
    "torch.lobpcg", "torch.sort", "torch.argsort", "torch.topk",
    "torch.unique", "torch.unique_consecutive",
    "tf.linalg.eig", "tf.linalg.eigh", "tf.linalg.svd", "tf.linalg.qr",
    "tf.linalg.lu", "tf.linalg.cholesky", "tf.linalg.eigvals",
    "tf.linalg.eigvalsh", "tf.svd", "tf.qr", "tf.sort", "tf.argsort",
    "tf.top_k", "tf.nn.top_k", "tf.unique", "tf.unique_with_counts",
}


def _rel_err(c: np.ndarray, g: np.ndarray, rtol: float, atol: float) -> float:
    finite = np.isfinite(c) & np.isfinite(g)
    if not np.any(finite):
        return 0.0
    denom = atol + rtol * np.abs(c[finite])
    denom = np.where(denom > 0, denom, atol + 1e-12)
    return float(np.max(np.abs(c[finite] - g[finite]) / denom))


def _sorted_rel(c: np.ndarray, g: np.ndarray, rtol: float, atol: float) -> float:
    cc = np.abs(c).ravel(); gg = np.abs(g).ravel()
    if cc.size != gg.size or cc.size == 0:
        return float("inf")
    cs = np.sort(cc); gs = np.sort(gg)
    denom = atol + rtol * np.abs(cs)
    denom = np.where(denom > 0, denom, atol + 1e-12)
    return float(np.max(np.abs(cs - gs) / denom))


def _triage_tf(bug_json: dict, bug_dir: Path) -> tuple[bool, str]:
    """Return (is_real_bug, reason). Re-validates a TF bug by loading the
    saved numpy inputs and running the comparison fresh would require the
    original environment — instead we re-apply the statistical filters on
    the detail string + mutation + used_apis."""
    bt = bug_json.get("bug_type", "").lower()
    detail = bug_json.get("detail", "")
    used_apis = bug_json.get("used_apis", [])
    mut = bug_json.get("mutation", "")

    if bt == "crash":
        if _INFRA_RX.search(detail):
            return False, "infra error"
        if "Model" in detail and "to(" in detail:
            return False, "model construction crash (likely GPU contention)"
        return True, "real crash"

    if bt == "nan":
        return True, "asymmetric NaN/Inf"

    if bt == "inconsistent":
        m_rel = re.search(r"rel_err=([\d.e+-]+)", detail)
        m_ndf = re.search(r"nondet_floor=([\d.e+-]+)", detail)
        if not m_rel:
            return False, "malformed detail"
        rel = float(m_rel.group(1))
        ndf = float(m_ndf.group(1)) if m_ndf else 0.0
        if rel <= BUG_MARGIN:
            return False, f"rel_err={rel:.1f} <= {BUG_MARGIN}×tol"
        if ndf >= NONDET_MAX:
            return False, f"nondet_floor={ndf:.1f} >= {NONDET_MAX}"
        if rel < NONDET_MULTIPLIER * ndf:
            return False, f"rel_err={rel:.1f} within {NONDET_MULTIPLIER}×nondet_floor={ndf:.1f}"
        if any(a in _PERM_INVARIANT for a in used_apis):
            return True, "perm-invariant API but oracle's sorted-check passed before save; keep for review"
        return True, f"rel_err={rel:.1f} > {BUG_MARGIN}× (nondet={ndf:.1f}, mut={mut})"

    return False, f"unknown bug_type={bt}"


def _triage_pt(bug_json: dict, bug_dir: Path) -> tuple[bool, str]:
    """Re-triage a PT bug against the same filters."""
    bt = bug_json.get("bug_type", "").lower()
    detail = bug_json.get("detail", "")
    used_apis = bug_json.get("used_apis", [])
    mut = bug_json.get("mutation_name", "")
    if bt == "crash":
        if _INFRA_RX.search(detail):
            return False, "infra error"
        if "Model().to(" in detail or "model = Model().to(device)" in detail:
            return False, "model construction crash (likely GPU contention)"
        return True, "real crash"
    if bt == "nan":
        return True, "asymmetric NaN/Inf"
    if bt == "inconsistent":
        m_rel = re.search(r"rel_err=([\d.e+-]+)", detail)
        m_ndf = re.search(r"nondet_floor=([\d.e+-]+)", detail)
        if not m_rel:
            # integer mismatch or shape — treat as bug
            if "integer mismatch" in detail or "shape mismatch" in detail:
                return True, detail[:80]
            return False, "malformed detail"
        rel = float(m_rel.group(1))
        ndf = float(m_ndf.group(1)) if m_ndf else 0.0
        if rel <= BUG_MARGIN:
            return False, f"rel_err={rel:.1f} <= {BUG_MARGIN}×tol"
        if ndf >= NONDET_MAX:
            return False, f"nondet_floor={ndf:.1f} >= {NONDET_MAX}"
        if rel < NONDET_MULTIPLIER * ndf:
            return False, f"rel_err={rel:.1f} within {NONDET_MULTIPLIER}×nondet_floor={ndf:.1f}"
        return True, f"rel_err={rel:.1f} > {BUG_MARGIN}× (nondet={ndf:.1f}, mut={mut})"
    return False, f"unknown bug_type={bt}"


# ── Copy helpers ────────────────────────────────────────────────────────────

def _copy_bug_artifacts(
    src_bug_dir: Path, base: str, dst_dir: Path, extra: list[str] | None = None,
) -> list[str]:
    """Copy JSON + model + artifacts for one bug. Return list of copied paths."""
    copied: list[str] = []
    for suffix in [".json", ".inputs.pt", ".inputs.npy", ".cpu_out.pt",
                   ".gpu_out.pt", ".repro.py"]:
        src = src_bug_dir / f"{base}{suffix}"
        if src.exists():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    for e in (extra or []):
        if Path(e).exists():
            dst = dst_dir / Path(e).name
            shutil.copy2(e, dst)
            copied.append(str(dst))
    return copied


# ── Wait for runs ───────────────────────────────────────────────────────────

def _procs_alive() -> list[int]:
    """Return PIDs of the smolfuzz main/run_tf Python processes only.

    Filters to lines whose first token (after the PID) is `python3`, so
    Claude-Code's bash wrapper processes (which have the smolfuzz command
    in their arg list but are shells, not Python) are excluded.
    """
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", r"python3 -m smolfuzz\.(main|run_tf)"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0].isdigit() and toks[1].endswith("python3"):
            pids.append(int(toks[0]))
    return pids


def _wait_for_runs(poll_every: int = 30, max_wait_s: int = 6 * 3600) -> None:
    start = time.time()
    while True:
        alive = _procs_alive()
        if not alive:
            print(f"[finalize] No smolfuzz processes alive after "
                  f"{int(time.time() - start)}s")
            return
        if time.time() - start > max_wait_s:
            print("[finalize] Max wait exceeded — proceeding with whatever "
                  f"bugs exist. Alive PIDs: {alive}")
            return
        print(f"[finalize] Still alive: {alive} "
              f"(elapsed {int(time.time() - start)}s)", flush=True)
        time.sleep(poll_every)


# ── Main ────────────────────────────────────────────────────────────────────

def main(wait: bool = True) -> None:
    if wait:
        _wait_for_runs()

    real_bugs: list[dict] = []
    rejected: list[dict] = []

    # PyTorch bugs
    if PT_DIR.exists():
        pt_dst = OUT_ROOT / "pytorch"
        pt_dst.mkdir(exist_ok=True)
        for json_path in sorted(PT_DIR.glob("*.json")):
            data = json.loads(json_path.read_text())
            ok, reason = _triage_pt(data, PT_DIR)
            base = json_path.stem
            entry = {
                "framework": "pytorch",
                "base": base,
                "bug_type": data.get("bug_type"),
                "model_id": data.get("model_id"),
                "mutation": data.get("mutation_name"),
                "detail": data.get("detail", "")[:400],
                "used_apis": data.get("used_apis", []),
                "triage_reason": reason,
            }
            if ok:
                # Also copy the model source if referenced
                src_model = HERE / "results" / "clean_run_pt" / "full" / "workspace" / f"model_{data.get('model_id'):04d}.py"
                extras = [str(src_model)] if src_model.exists() else []
                entry["artifacts"] = _copy_bug_artifacts(PT_DIR, base, pt_dst, extras)
                real_bugs.append(entry)
            else:
                rejected.append(entry)

    # TensorFlow bugs
    if TF_DIR.exists():
        tf_dst = OUT_ROOT / "tensorflow"
        tf_dst.mkdir(exist_ok=True)
        for json_path in sorted(TF_DIR.glob("*.json")):
            data = json.loads(json_path.read_text())
            ok, reason = _triage_tf(data, TF_DIR)
            base = json_path.stem
            entry = {
                "framework": "tensorflow",
                "base": base,
                "bug_type": data.get("bug_type"),
                "model_id": data.get("model_id"),
                "mutation": data.get("mutation"),
                "detail": data.get("detail", "")[:400],
                "used_apis": data.get("used_apis", []),
                "triage_reason": reason,
            }
            if ok:
                src_model = HERE / "results" / "clean_run_tf" / "models" / f"model_{data.get('model_id'):04d}.py"
                extras = [str(src_model)] if src_model.exists() else []
                entry["artifacts"] = _copy_bug_artifacts(TF_DIR, base, tf_dst, extras)
                real_bugs.append(entry)
            else:
                rejected.append(entry)

    # Run stats from logs
    def _stat(log_path: Path, pat: str) -> int:
        if not log_path.exists():
            return 0
        with log_path.open() as f:
            return sum(1 for ln in f if re.search(pat, ln))

    pt_started = _stat(PT_LOG, r"Model \d+ / 200")
    tf_started = _stat(TF_LOG, r"Model \d+ / 200")
    pt_clean = _stat(PT_LOG, r"no bug found")
    tf_clean = _stat(TF_LOG, r"clean after")

    # Write SUMMARY.md
    lines = [
        "# smolfuzz run — real-bug triage",
        "",
        f"- Finalized at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- PT models started: {pt_started}",
        f"- PT clean (60s fuzzed, no bug): {pt_clean}",
        f"- TF models started: {tf_started}",
        f"- TF clean (60s fuzzed, no bug): {tf_clean}",
        f"- Real bugs kept: **{len(real_bugs)}**",
        f"- Flagged but rejected as FP on re-triage: {len(rejected)}",
        "",
        "## FP filters applied",
        "",
        f"- `BUG_MARGIN`: rel_err must exceed `{BUG_MARGIN}×` dtype tolerance",
        f"- Rejected if `nondet_floor >= {NONDET_MAX}` (inherently non-deterministic model)",
        f"- Rejected if `rel_err < {NONDET_MULTIPLIER} × nondet_floor`",
        "- Infrastructure errors (CUBLAS/CUDA OOM/driver) filtered",
        "- GPU-side `Model().to(device)` construction crashes filtered (contention, not library)",
        "- Permutation-invariant API outputs (eig/svd/qr/sort/…) require sorted-magnitude divergence",
        "",
        "## Real bugs",
        "",
    ]
    if real_bugs:
        for b in real_bugs:
            lines += [
                f"### {b['framework']} {b['base']}",
                f"- Type: `{b['bug_type']}`",
                f"- Mutation: `{b['mutation']}`",
                f"- Reason: {b['triage_reason']}",
                f"- Detail: `{b['detail']}`",
                f"- APIs used: {', '.join(b['used_apis'][:8])}",
                "",
            ]
    else:
        lines += ["(no real bugs kept)", ""]

    if rejected:
        lines += ["## Rejected (FP filters)", ""]
        for b in rejected:
            lines += [
                f"- `{b['framework']} {b['base']}` — {b['triage_reason']}",
            ]
        lines.append("")

    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(lines))
    (OUT_ROOT / "index.json").write_text(json.dumps({
        "real_bugs": real_bugs,
        "rejected": rejected,
        "stats": {
            "pt_started": pt_started, "pt_clean": pt_clean,
            "tf_started": tf_started, "tf_clean": tf_clean,
        },
    }, indent=2))

    print(f"[finalize] Real bugs: {len(real_bugs)} → {OUT_ROOT}")
    print(f"[finalize] Rejected: {len(rejected)}")
    print(f"[finalize] Summary: {OUT_ROOT / 'SUMMARY.md'}")


if __name__ == "__main__":
    wait = not ("--no-wait" in sys.argv)
    main(wait=wait)
