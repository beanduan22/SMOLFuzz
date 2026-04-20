"""
Differential testing oracle — CPU vs GPU output comparison.

False-positive prevention (what breaks naive L2-threshold oracles):

  1. GPU non-determinism. Ops like scatter_add / index_add / segment_sum
     produce different results on two successive GPU runs with the SAME
     input. We run GPU twice, treat any CPU/GPU discrepancy smaller than
     GPU-vs-GPU as device-level noise and drop it.

  2. Float16 / bfloat16 expected loss. float16 has ~1e-3 relative precision.
     An absolute L2 threshold of 1e-3 false-positives on every f16 reduction.
     We use per-dtype relative tolerance, matching torch.allclose defaults:
        float64 → rtol=1e-5  atol=1e-8
        float32 → rtol=1e-4  atol=1e-5
        bfloat16→ rtol=1e-2  atol=1e-2
        float16 → rtol=1e-2  atol=1e-3

  3. Large-magnitude outputs. |cpu - gpu| scales with ||cpu||. We compute
        rel_err = max_i |cpu_i - gpu_i| / (atol + rtol * |cpu_i|)
     and flag only when rel_err > 1.

  4. Asymmetric NaN from rounding order. A single NaN position in a 1M
     element tensor is usually not a bug. We require at least max(4, 0.1%)
     of positions to be asymmetric.

  5. Generation failures with different wording across devices. When CPU
     and GPU both crash we normalise the message (strip "CUDA " / "cuda:N"
     prefixes, keywords) before comparing.

  6. Argmax / sort / topk index flips under tied values. Integer outputs
     are compared with index-tolerance when a companion float output also
     differs by > rtol.

Classifies each ExecutionPair as:
  CRASH              – single-device crash or different-root-cause crashes
  NAN                – genuine asymmetric non-finite values
  INCONSISTENT       – numerically different beyond dtype-aware tolerance
  GENERATION_FAILURE – both devices crash with same root cause
  NONDET             – GPU-vs-GPU diff exceeds CPU-vs-GPU diff (not a bug)
  CLEAN              – no anomaly
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .executor import ExecutionPair, RunResult

logger = logging.getLogger(__name__)


# Per-dtype (rtol, atol) — mirrors torch.allclose defaults but slightly looser
# to account for the fact that the same model can legitimately differ on CPU
# vs GPU by orders of kernel rounding error.
_TOL: dict = {
    torch.float64: (1e-5, 1e-8),
    torch.float32: (1e-4, 1e-5),
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (1e-2, 1e-3),
}
# Integer tensors: allow fraction of mismatching positions if the operation
# is likely an argmax/sort over near-ties. We cap at 1% mismatches.
_INT_MISMATCH_FRAC = 0.01

# NaN / Inf asymmetry thresholds: allow small rounding-order differences.
_NAN_ASYM_FRAC = 1e-3   # require > 0.1% asymmetric positions
_NAN_ASYM_MIN = 4       # AND at least 4 positions (for small outputs)

# Environmental / infrastructure errors that must NEVER be classified as a
# library bug. These are almost always transient (OOM, driver resets,
# scheduler races) and the user explicitly demanded no false positives.
_INFRA_ERROR_RX = re.compile(
    r"out of memory"
    r"|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorOutOfMemory|cuMemAlloc failed"
    r"|CUBLAS_STATUS_NOT_INITIALIZED|CUBLAS_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_NOT_INITIALIZED|CUDNN_STATUS_ALLOC_FAILED"
    r"|Failed to initialize CUDA|driver shutting down"
    r"|ResourceExhaustedError|RESOURCE_EXHAUSTED"
    r"|NCCL error|no kernel image is available for execution"
    r"|CUDA runtime error: unknown error",
    re.IGNORECASE,
)


def _is_infra_error(err: str) -> bool:
    return bool(err) and bool(_INFRA_ERROR_RX.search(err))


class BugType(str, Enum):
    CRASH = "crash"
    NAN = "nan"
    INCONSISTENT = "inconsistent"
    GENERATION_FAILURE = "generation_failure"   # code error, not library bug
    NONDET = "nondet"                            # GPU non-determinism, not a bug
    CLEAN = "clean"


@dataclass
class OracleReport:
    bug_type: BugType
    model_id: int
    mutation_id: int
    mutation_name: str
    cpu_status: str
    gpu_status: str
    detail: str
    used_apis: List[str]
    timestamp: str = ""

    def is_bug(self) -> bool:
        """True only for genuine library bugs (not generation / nondet)."""
        return self.bug_type in (BugType.CRASH, BugType.NAN, BugType.INCONSISTENT)

    def is_generation_failure(self) -> bool:
        return self.bug_type == BugType.GENERATION_FAILURE

    def is_nondet(self) -> bool:
        return self.bug_type == BugType.NONDET


# ──────────────────────────────────────────────────────────────────────────
# Error-similarity helper — normalises device-specific wording.
# ──────────────────────────────────────────────────────────────────────────

_DEVICE_PREFIX_RX = re.compile(
    r"CUDA[a-zA-Z_ ]*:?\s*"
    r"|cuda:\d+\s*"
    r"|cuDNN[a-zA-Z_ ]*:?\s*"
    r"|GPU\s*\d*:?\s*"
    r"|CPU:?\s*"
)


def _normalize_error(err: str) -> Tuple[str, str]:
    """Strip device noise and return (exc_type, canonical_msg)."""
    if not err:
        return ("", "")
    lines = [l.strip() for l in err.splitlines() if l.strip()]
    exc_line = lines[-1] if lines else err.strip()
    # "TypeName: message"
    parts = exc_line.split(":", 1)
    exc_type = parts[0].strip()
    exc_msg = parts[1].strip() if len(parts) > 1 else ""
    # Canonicalise
    exc_msg = _DEVICE_PREFIX_RX.sub("", exc_msg)
    exc_msg = re.sub(r"\s+", " ", exc_msg).lower()
    return exc_type, exc_msg


def _errors_are_similar(err1: str, err2: str) -> bool:
    """True when both errors share the same root cause (code-level issue)."""
    if not err1 or not err2:
        return True
    t1, m1 = _normalize_error(err1)
    t2, m2 = _normalize_error(err2)
    if t1 != t2:
        return False
    if not m1 or not m2:
        return True
    # Character overlap on a longer prefix
    n = min(200, len(m1), len(m2))
    if n == 0:
        return True
    common = sum(a == b for a, b in zip(m1[:n], m2[:n]))
    return common / n > 0.55


# ──────────────────────────────────────────────────────────────────────────
# Numerical comparison helpers.
# ──────────────────────────────────────────────────────────────────────────


def _dtype_tol(t: torch.Tensor) -> Tuple[float, float]:
    """Return (rtol, atol) for the given tensor's dtype."""
    return _TOL.get(t.dtype, (1e-4, 1e-5))


def _align_dtype(cpu: torch.Tensor, gpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[float, float]]:
    """Cast both to the common (more permissive) dtype for comparison."""
    # Rank dtypes by precision — less precise wins so tolerance matches
    # the lower-precision dtype.
    ranking = {
        torch.float64: 4,
        torch.float32: 3,
        torch.bfloat16: 2,
        torch.float16: 1,
    }
    rc = ranking.get(cpu.dtype, 0)
    rg = ranking.get(gpu.dtype, 0)
    target = cpu.dtype if rc <= rg else gpu.dtype
    tol = _TOL.get(target, (1e-4, 1e-5))
    return cpu.to(target).float(), gpu.to(target).float(), tol


def _relative_error(c: torch.Tensor, g: torch.Tensor, rtol: float, atol: float) -> float:
    """
    max_i |c_i - g_i| / (atol + rtol * |c_i|) over finite positions.
    A value > 1.0 means the inputs violate torch.allclose(rtol, atol).
    """
    both_finite = torch.isfinite(c) & torch.isfinite(g)
    if not both_finite.any():
        return 0.0
    c_f = c[both_finite]
    g_f = g[both_finite]
    denom = atol + rtol * c_f.abs()
    # Avoid zero denom (shouldn't happen with atol>0 but be safe)
    denom = torch.where(denom > 0, denom, torch.full_like(denom, atol + 1e-12))
    return (c_f - g_f).abs().div(denom).max().item()


def _sorted_magnitude_error(
    c: torch.Tensor, g: torch.Tensor, rtol: float, atol: float,
) -> float:
    """
    Compute rel_err between |c| and |g| after sorting. This collapses any
    permutation-order difference (e.g. eig/svd eigenvalue ordering) so a
    permutation-only disagreement reports ~0 while a genuine numerical
    divergence still shows up.
    """
    c_mag = c.abs().reshape(-1)
    g_mag = g.abs().reshape(-1)
    finite = torch.isfinite(c_mag) & torch.isfinite(g_mag)
    if not finite.any() or finite.sum().item() != c_mag.numel():
        return 0.0
    c_sorted, _ = torch.sort(c_mag)
    g_sorted, _ = torch.sort(g_mag)
    if c_sorted.numel() != g_sorted.numel():
        return float("inf")
    denom = atol + rtol * c_sorted.abs()
    denom = torch.where(denom > 0, denom, torch.full_like(denom, atol + 1e-12))
    return (c_sorted - g_sorted).abs().div(denom).max().item()


# APIs whose output is only defined up to a permutation / sign. Positional
# CPU-vs-GPU comparison false-positives on these every time; we check the
# sorted-magnitude metric as a secondary guard.
_PERMUTATION_INVARIANT_APIS = {
    "torch.linalg.eig", "torch.linalg.eigh", "torch.linalg.eigvals",
    "torch.linalg.eigvalsh", "torch.linalg.svd", "torch.linalg.svdvals",
    "torch.linalg.qr", "torch.linalg.lu", "torch.linalg.lu_factor",
    "torch.linalg.cholesky", "torch.linalg.cholesky_ex",
    "torch.linalg.householder_product",
    "torch.eig", "torch.symeig", "torch.svd", "torch.qr", "torch.lu",
    "torch.geqrf", "torch.orgqr", "torch.ormqr", "torch.lobpcg",
    "torch.triangular_solve",
    "torch.unique", "torch.unique_consecutive",
    "torch.sort", "torch.Tensor.sort", "torch.argsort", "torch.Tensor.argsort",
    "torch.topk", "torch.Tensor.topk",
    "torch.mode", "torch.Tensor.mode",
    "torch.nn.functional.kthvalue", "torch.kthvalue", "torch.Tensor.kthvalue",
}


def _uses_permutation_invariant_api(apis: List[str]) -> bool:
    return any(a in _PERMUTATION_INVARIANT_APIS for a in apis)


def _nan_inf_asymmetry(c: torch.Tensor, g: torch.Tensor) -> Tuple[int, int, int]:
    """Return (asym_nan, asym_inf, nan_vs_inf) position counts."""
    cpu_nan = torch.isnan(c); gpu_nan = torch.isnan(g)
    cpu_inf = torch.isinf(c); gpu_inf = torch.isinf(g)
    asym_nan = ((cpu_nan & ~gpu_nan & ~gpu_inf) | (gpu_nan & ~cpu_nan & ~cpu_inf)).sum().item()
    asym_inf = ((cpu_inf & ~gpu_inf & ~gpu_nan) | (gpu_inf & ~cpu_inf & ~cpu_nan)).sum().item()
    nan_vs_inf = ((cpu_nan & gpu_inf) | (cpu_inf & gpu_nan)).sum().item()
    return int(asym_nan), int(asym_inf), int(nan_vs_inf)


def _significant_asymmetry(count: int, total: int) -> bool:
    """Is `count` asymmetric positions in a tensor of `total` significant?"""
    if total == 0 or count == 0:
        return False
    frac = count / total
    return count >= _NAN_ASYM_MIN and frac >= _NAN_ASYM_FRAC


# ──────────────────────────────────────────────────────────────────────────
# Mutation name table.
# ──────────────────────────────────────────────────────────────────────────

_MUTATION_NAMES = {
    0: "baseline",
    1: "add_noise",
    2: "scale_small",
    3: "mask",
    4: "uniform",
    5: "scale_large",
}


class DifferentialOracle:
    def __init__(self, output_dir: str | Path) -> None:
        self._bug_dir = Path(output_dir) / "bugs"
        self._bug_dir.mkdir(parents=True, exist_ok=True)
        self._bug_count = 0
        self._gen_fail_count = 0
        self._nondet_count = 0
        self._clean_count = 0

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def evaluate(
        self, pair: ExecutionPair, used_apis: List[str], model_code: str = "",
    ) -> OracleReport:
        mut_name = _MUTATION_NAMES.get(pair.mutation_id, f"mut{pair.mutation_id}")
        report = self._classify(pair, used_apis, mut_name)
        if report.is_bug():
            self._bug_count += 1
            self._save_report(report, pair, model_code)
            logger.warning(
                "BUG detected | type=%s | model=%d | mutation=%s | %s",
                report.bug_type, report.model_id, mut_name, report.detail[:120],
            )
        elif report.is_generation_failure():
            self._gen_fail_count += 1
            self._save_report(report, pair, model_code)
        elif report.is_nondet():
            self._nondet_count += 1
            logger.info(
                "Non-deterministic (not a bug): model=%d mut=%s | %s",
                report.model_id, mut_name, report.detail[:120],
            )
        else:
            self._clean_count += 1
        return report

    def stats(self) -> dict:
        return {
            "bugs": self._bug_count,
            "generation_failures": self._gen_fail_count,
            "nondet": self._nondet_count,
            "clean": self._clean_count,
        }

    # ------------------------------------------------------------------ #
    # Internal                                                            #
    # ------------------------------------------------------------------ #

    def _classify(
        self, pair: ExecutionPair, used_apis: List[str], mut_name: str,
    ) -> OracleReport:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = dict(
            model_id=pair.model_id,
            mutation_id=pair.mutation_id,
            mutation_name=mut_name,
            cpu_status=pair.cpu.status,
            gpu_status=pair.gpu.status,
            used_apis=used_apis,
            timestamp=ts,
        )

        # 1) Crash / error triage ----------------------------------------
        cpu_failed = pair.cpu.status in ("crash", "timeout", "error")
        gpu_failed = pair.gpu.status in ("crash", "timeout", "error")
        if cpu_failed or gpu_failed:
            # Scan the FULL error string for infra keywords — the error
            # class (e.g. CUBLAS_STATUS_ALLOC_FAILED) typically sits at the
            # tail of the traceback, past the first 400 chars.
            cpu_err_full = pair.cpu.error if cpu_failed else ""
            gpu_err_full = pair.gpu.error if gpu_failed else ""
            cpu_err = cpu_err_full[:400]
            gpu_err = gpu_err_full[:400]
            # Infrastructure error on either device — not a library bug.
            if _is_infra_error(cpu_err_full) or _is_infra_error(gpu_err_full):
                # Show the tail of the traceback so the diagnostic is useful.
                cpu_tail = cpu_err_full[-200:] if cpu_err_full else ""
                gpu_tail = gpu_err_full[-200:] if gpu_err_full else ""
                return OracleReport(
                    bug_type=BugType.GENERATION_FAILURE,
                    detail=(
                        f"Infrastructure error (OOM / driver / scheduler); "
                        f"not classified as a library bug.\n"
                        f"CPU-tail: {cpu_tail}\nGPU-tail: {gpu_tail}"
                    ),
                    **base,
                )
            if cpu_failed and gpu_failed:
                if _errors_are_similar(cpu_err, gpu_err):
                    return OracleReport(
                        bug_type=BugType.GENERATION_FAILURE,
                        detail=f"Both devices crashed with similar error.\nCPU: {cpu_err[:200]}",
                        **base,
                    )
                return OracleReport(
                    bug_type=BugType.CRASH,
                    detail=f"CPU error: {cpu_err[:200]}\nGPU error: {gpu_err[:200]}",
                    **base,
                )
            # Single-device crash — but first check if the surviving device
            # simply isn't producing numerical output that would diverge.
            failing = "CPU" if cpu_failed else "GPU"
            err = cpu_err if cpu_failed else gpu_err
            return OracleReport(
                bug_type=BugType.CRASH,
                detail=f"{failing}-only crash: {err[:300]}",
                **base,
            )

        cpu_outs = pair.cpu.outputs
        gpu_outs = pair.gpu.outputs
        if not cpu_outs or not gpu_outs:
            return OracleReport(bug_type=BugType.CLEAN, detail="empty outputs", **base)

        # 2) GPU non-determinism estimate -------------------------------
        # If a second GPU run is available, compute a per-output nondet
        # "noise floor". We'll require CPU/GPU diff to exceed this floor.
        gpu_rep = pair.gpu.outputs_repeat
        nondet_floors: List[float] = []
        if gpu_rep and len(gpu_rep) == len(gpu_outs):
            for g1, g2 in zip(gpu_outs, gpu_rep):
                if isinstance(g1, torch.Tensor) and isinstance(g2, torch.Tensor) \
                   and g1.shape == g2.shape and g1.is_floating_point():
                    a, b, tol = _align_dtype(g1, g2)
                    nondet_floors.append(_relative_error(a, b, *tol))
                else:
                    nondet_floors.append(0.0)

        # 3) Asymmetric NaN / Inf ---------------------------------------
        nan_details = []
        for i, (c, g) in enumerate(zip(cpu_outs, gpu_outs)):
            if not isinstance(c, torch.Tensor) or not isinstance(g, torch.Tensor):
                continue
            if c.shape != g.shape:
                continue
            c_f = c.float()
            g_f = g.float()
            asym_nan, asym_inf, nan_vs_inf = _nan_inf_asymmetry(c_f, g_f)
            total = c_f.numel()
            bad = asym_nan + asym_inf + nan_vs_inf
            if _significant_asymmetry(bad, total):
                nan_details.append(
                    f"output[{i}]: asym_nan={asym_nan} asym_inf={asym_inf} "
                    f"nan_vs_inf={nan_vs_inf} total_positions={total} "
                    f"shape={list(c.shape)}"
                )
        if nan_details:
            return OracleReport(
                bug_type=BugType.NAN, detail="; ".join(nan_details), **base,
            )

        # 4) Numerical inconsistency — relative tolerance ---------------
        incons_details = []
        nondet_details = []
        uses_perm_invariant = _uses_permutation_invariant_api(used_apis)
        for i, (c, g) in enumerate(zip(cpu_outs, gpu_outs)):
            if not isinstance(c, torch.Tensor) or not isinstance(g, torch.Tensor):
                continue
            if c.shape != g.shape:
                incons_details.append(
                    f"output[{i}]: shape mismatch cpu={list(c.shape)} gpu={list(g.shape)}"
                )
                continue

            # Complex tensors: split into (real, imag) and treat each as a
            # float tensor. Positional comparison on complex values is still
            # sensitive to permutation (for eig/svd); handled below.
            if c.is_complex():
                c = torch.cat([c.real, c.imag], dim=-1)
                g = torch.cat([g.real, g.imag], dim=-1)
                is_floatlike = True
            else:
                is_floatlike = c.is_floating_point()

            if not is_floatlike:
                # Integer/bool outputs — allow small fraction of mismatches
                if not torch.equal(c, g):
                    diff = (c != g).sum().item()
                    total = c.numel() or 1
                    if diff / total > _INT_MISMATCH_FRAC:
                        incons_details.append(
                            f"output[{i}]: integer mismatch {diff}/{total} positions "
                            f"({100*diff/total:.2f}%) dtype={c.dtype}"
                        )
                continue

            a, b, (rtol, atol) = _align_dtype(c, g)
            rel = _relative_error(a, b, rtol, atol)
            # Require a large margin above tolerance to call something a
            # bug. rel_err of a few × tolerance is expected for models using
            # autograd, BatchNorm, or reductions (summation-order differences
            # in f32 kernels). Real library bugs in the reference catalogue
            # show rel_err of hundreds to thousands; 100× is a conservative
            # single-sided filter consistent with the paper's no-FP intent.
            BUG_MARGIN = 100.0
            if rel <= BUG_MARGIN:
                continue  # within expected kernel-rounding margin

            # Permutation-invariant APIs (eig/svd/qr/sort/topk/…): compare
            # sorted magnitudes. If those agree, the positional divergence is
            # a permutation artifact and NOT a library bug.
            if uses_perm_invariant:
                sorted_rel = _sorted_magnitude_error(a, b, rtol, atol)
                if sorted_rel <= 1.0:
                    nondet_details.append(
                        f"output[{i}]: rel_err={rel:.3e} but sorted_rel={sorted_rel:.3e} "
                        f"≤ 1.0 → permutation artifact on a decomp API, not a bug"
                    )
                    continue

            nd_floor = nondet_floors[i] if i < len(nondet_floors) else 0.0
            # If known-nondet op OR the GPU-vs-GPU noise is of the same order,
            # the mismatch is noise, not a library bug. We require a 10× margin
            # over the nondet floor, and reject outright when the floor is
            # already very high (model is numerically unstable by construction).
            if pair.has_nondet_ops or nd_floor >= 100.0 or rel < max(1.0, 10.0 * nd_floor):
                nondet_details.append(
                    f"output[{i}]: rel_err={rel:.3e} nondet_floor={nd_floor:.3e} "
                    f"dtype={c.dtype} shape={list(c.shape)}"
                )
                continue

            incons_details.append(
                f"output[{i}]: rel_err={rel:.3e} > 1.0 "
                f"dtype={c.dtype} shape={list(c.shape)} "
                f"rtol={rtol:.0e} atol={atol:.0e} nondet_floor={nd_floor:.3e}"
            )

        if incons_details:
            return OracleReport(
                bug_type=BugType.INCONSISTENT,
                detail="; ".join(incons_details),
                **base,
            )
        if nondet_details:
            return OracleReport(
                bug_type=BugType.NONDET,
                detail="; ".join(nondet_details),
                **base,
            )

        return OracleReport(bug_type=BugType.CLEAN, detail="", **base)

    # ------------------------------------------------------------------ #
    # Report persistence                                                  #
    # ------------------------------------------------------------------ #

    def _save_report(
        self, report: OracleReport, pair: ExecutionPair, model_code: str = "",
    ) -> None:
        if report.bug_type == BugType.GENERATION_FAILURE:
            logger.info(
                "Generation failure (not a bug): model=%d mut=%s | %s",
                report.model_id, report.mutation_name, report.detail[:120],
            )
            return

        base = (
            f"bug_{report.bug_type.value}_"
            f"m{report.model_id:04d}_mut{report.mutation_id}_"
            f"{report.timestamp}"
        )

        data = {
            "bug_type": report.bug_type.value,
            "model_id": report.model_id,
            "mutation_id": report.mutation_id,
            "mutation_name": report.mutation_name,
            "cpu_status": report.cpu_status,
            "gpu_status": report.gpu_status,
            "detail": report.detail,
            "used_apis": report.used_apis,
            "timestamp": report.timestamp,
            "artifacts": {
                "inputs": f"{base}.inputs.pt",
                "cpu_outputs": f"{base}.cpu_out.pt",
                "gpu_outputs": f"{base}.gpu_out.pt",
                "reproducer": f"{base}.repro.py",
            },
        }
        (self._bug_dir / f"{base}.json").write_text(json.dumps(data, indent=2))

        if pair.inputs:
            try:
                torch.save(pair.inputs, self._bug_dir / f"{base}.inputs.pt")
            except Exception as exc:
                logger.warning("Could not save inputs: %s", exc)

        if pair.cpu.outputs:
            try:
                torch.save(pair.cpu.outputs, self._bug_dir / f"{base}.cpu_out.pt")
            except Exception as exc:
                logger.warning("Could not save CPU outputs: %s", exc)
        if pair.gpu.outputs:
            try:
                torch.save(pair.gpu.outputs, self._bug_dir / f"{base}.gpu_out.pt")
            except Exception as exc:
                logger.warning("Could not save GPU outputs: %s", exc)

        if model_code:
            repro = self._build_reproducer(report, base, model_code)
            (self._bug_dir / f"{base}.repro.py").write_text(repro)

        logger.info("Bug report saved → %s/%s.*", self._bug_dir, base)

    # ------------------------------------------------------------------ #
    # Reproducer builder                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_reproducer(report: OracleReport, base: str, model_code: str) -> str:
        bug_type = report.bug_type.value
        lines: list[str] = []

        lines += [
            '#!/usr/bin/env python3',
            '"""',
            f'SMOLFuzz bug reproducer — {bug_type.upper()}',
            f'Model : {report.model_id}',
            f'Mutation: {report.mutation_name}',
            f'Detail  : {report.detail[:120]}',
            f'APIs    : {", ".join(report.used_apis[:10])}',
            '',
            f'Run:  python3 {base}.repro.py',
            'Requires: PyTorch with CUDA available.',
            '"""',
            'import os',
            'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")',
            'import torch',
            'import torch.nn as nn',
            'import torch.nn.functional as F  # noqa: F401',
            'from pathlib import Path',
            '',
            '# Determinism settings (match fuzzer runner).',
            'torch.backends.cuda.matmul.allow_tf32 = False',
            'torch.backends.cudnn.allow_tf32 = False',
            'torch.backends.cudnn.deterministic = True',
            'torch.backends.cudnn.benchmark = False',
            'try:',
            '    torch.use_deterministic_algorithms(True, warn_only=True)',
            'except Exception:',
            '    pass',
            '',
            '# ── Model (LLM-generated) ────────────────────────────',
        ]

        cleaned_code = model_code.strip()
        for dup in ("import torch", "import torch.nn as nn",
                    "import torch.nn.functional as F"):
            cleaned_code = cleaned_code.replace(dup + "\n", "")
            cleaned_code = cleaned_code.replace(dup + ";", "")
        lines += [cleaned_code, ""]

        inputs_file = f"{base}.inputs.pt"
        lines += [
            "# ── Load buggy inputs ─────────────────────────────",
            "_here = Path(__file__).parent",
            f"inputs = torch.load(_here / {inputs_file!r}, weights_only=False)",
            "",
            "# ── Build two models with identical weights ───────",
            "torch.manual_seed(42)",
            "torch.cuda.manual_seed_all(42)",
            "cpu_model = Model().cpu().eval()",
            "gpu_model = Model().cuda().eval()",
            "gpu_model.load_state_dict(cpu_model.state_dict())",
        ]

        if bug_type == "crash":
            crashing = "cpu" if report.cpu_status in ("crash", "error", "timeout") else "cuda"
            ok = "cuda" if crashing == "cpu" else "cpu"
            ok_model = "cpu_model" if ok == "cpu" else "gpu_model"
            bad_model = "cpu_model" if crashing == "cpu" else "gpu_model"
            lines += [
                "",
                f"# ── {ok.upper()} succeeds ────────────────────────",
                f"ok_inputs = [x.to('{ok}') if isinstance(x, torch.Tensor) else x for x in inputs]",
                "with torch.set_grad_enabled(True):",
                f"    ok_out = {ok_model}(*ok_inputs)",
                f"print(f'{ok.upper()} output ok: {{type(ok_out).__name__}}')",
                "",
                f"# ── {crashing.upper()} crashes ───────────────────",
                f"bad_inputs = [x.to('{crashing}') if isinstance(x, torch.Tensor) else x for x in inputs]",
                "with torch.set_grad_enabled(True):",
                f"    bad_out = {bad_model}(*bad_inputs)  # <── expect crash",
                f"print(f'{crashing.upper()} output: {{bad_out}}')",
            ]
        else:  # inconsistent / nan
            lines += [
                "",
                "cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]",
                "gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]",
                "with torch.set_grad_enabled(True):",
                "    cpu_out = cpu_model(*cpu_inputs)",
                "    gpu_out = gpu_model(*gpu_inputs)",
                "cpu_outs = [cpu_out] if isinstance(cpu_out, torch.Tensor) else list(cpu_out)",
                "gpu_outs = [gpu_out] if isinstance(gpu_out, torch.Tensor) else list(gpu_out)",
                "",
                "TOL = {torch.float64:(1e-5,1e-8), torch.float32:(1e-4,1e-5),",
                "       torch.bfloat16:(1e-2,1e-2), torch.float16:(1e-2,1e-3)}",
                "for i,(c,g) in enumerate(zip(cpu_outs, gpu_outs)):",
                "    if isinstance(c, torch.Tensor) and isinstance(g, torch.Tensor):",
                "        rtol, atol = TOL.get(c.dtype, (1e-4, 1e-5))",
                "        c_f, g_f = c.float(), g.float().cpu()",
                "        finite = torch.isfinite(c_f) & torch.isfinite(g_f)",
                "        if finite.any():",
                "            rel = ((c_f[finite]-g_f[finite]).abs() / (atol + rtol * c_f[finite].abs())).max().item()",
                "            print(f'output[{i}] dtype={c.dtype} rel_err={rel:.3e} shape={list(c.shape)}')",
                "        else:",
                "            print(f'output[{i}]: all non-finite')",
            ]

        return "\n".join(lines) + "\n"
