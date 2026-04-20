"""
SMOLFuzz — TensorFlow track.

Pipeline (mirrors the paper):
  1. api_loader.load_and_classify(..., framework="tf")  — 11 groups
  2. MultiRouletteSelector                               — Eq. 1/2/3
  3. LLM synthesis with three dependency-type few-shot examples
     + self-repair loop                                  — §3.1.3 / Appendix prompt
  4. 5-strategy tensor mutation fuzzing, 60s budget      — §3.2
  5. CPU/GPU differential oracle                         — §3.3
        • CRASH / NAN / INCONSISTENT
        • dtype-aware relative tolerance (no FPs on f16 reductions)
        • symmetric NaN/Inf is NOT a bug
        • baseline crash ⇒ invalid model (never reported as a library bug)

Usage:
    python3 run_tf.py --models 500 --budget 60 --out results/tf_500
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from smolfuzz.api_loader import group_summary, load_and_classify
from smolfuzz.llm_client import OllamaClient
from smolfuzz.prompts import build_tf_repair_prompt, build_tf_synthesis_prompt
from smolfuzz.selector import MultiRouletteSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tf_fuzz")

# ─────────────────────────────────────────────────────────────────────────────
# Subprocess wrapper that runs a model twice on CPU and twice on GPU and
# returns the outputs as lists. Double-run on GPU exposes GPU non-determinism.
# ─────────────────────────────────────────────────────────────────────────────

_WRAPPER = r"""
import os, sys, json, traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
import numpy as np
import tensorflow as tf

for _g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

# ── LLM-generated class body ──────────────────────────────────────────────
{model_src}
# ─────────────────────────────────────────────────────────────────────────

input_file = sys.argv[1]
x_np = np.load(input_file).astype(np.float32)
result = dict(cpu=None, gpu=None, gpu_repeat=None, crash=None, no_gpu=False)

try:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        result['no_gpu'] = True
        print(json.dumps(result))
        sys.exit(0)

    def _flatten(t):
        return np.array(t, dtype=np.float64).flatten().tolist()

    # Build CPU model and initialise weights with a fixed seed.
    tf.random.set_seed(42)
    with tf.device('/CPU:0'):
        model_cpu = Model()
        x_cpu = tf.constant(x_np)
        _ = model_cpu(x_cpu, training=False)

    tf.random.set_seed(42)
    with tf.device('/GPU:0'):
        model_gpu = Model()
        x_gpu = tf.constant(x_np)
        _ = model_gpu(x_gpu, training=False)

    # Copy CPU → GPU weights so both models hold the exact same parameters.
    if len(model_cpu.variables) == len(model_gpu.variables):
        for vc, vg in zip(model_cpu.variables, model_gpu.variables):
            vg.assign(tf.cast(vc, vg.dtype))

    with tf.device('/CPU:0'):
        out_cpu = model_cpu(x_cpu, training=False)
    with tf.device('/GPU:0'):
        out_gpu_a = model_gpu(x_gpu, training=False)
        out_gpu_b = model_gpu(x_gpu, training=False)

    result['cpu']        = _flatten(out_cpu)
    result['gpu']        = _flatten(out_gpu_a)
    result['gpu_repeat'] = _flatten(out_gpu_b)
    result['cpu_dtype']  = str(out_cpu.dtype.name)

except Exception:
    result['crash'] = traceback.format_exc()[-800:]

print(json.dumps(result))
"""

# Quick CPU-only sanity check so we skip obviously broken LLM output before
# loading the GPU driver (cuts wasted compute dramatically).
_VALIDATOR = r"""
import os, sys, traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
tf.random.set_seed(42)

{model_src}

x_np = np.load(sys.argv[1]).astype(np.float32)
try:
    with tf.device('/CPU:0'):
        m = Model()
        out = m(tf.constant(x_np), training=False)
    if out is None:
        print('FAIL:no output')
    else:
        print('OK')
except Exception:
    print('FAIL:' + traceback.format_exc()[-500:])
"""


# ─────────────────────────────────────────────────────────────────────────────
# Source extraction / randomness gate
# ─────────────────────────────────────────────────────────────────────────────

_RANDOMNESS_RX = re.compile(
    r"\b("
    r"tf\.random\."
    r"|tf\.experimental\.numpy\.random\."
    r"|tf\.keras\.layers\.(Dropout|SpatialDropout[123]D"
    r"|AlphaDropout|GaussianDropout|GaussianNoise)"
    r"|tf\.nn\.dropout"
    r")"
)


def _extract_class(text: str) -> str | None:
    """Extract `class Model(...)` from the raw LLM reply."""
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    m = re.search(r"(class\s+Model\b.*?)(?=\nclass\s|\Z)", text, re.DOTALL)
    if not m:
        return None
    src = m.group(1).strip()
    if "def call(" not in src:
        return None
    return src


def _extract_used_apis(src: str) -> list[str]:
    m = re.search(r"USED_APIS\s*=\s*(\[.*?\])", src, re.DOTALL)
    if not m:
        return []
    try:
        import ast as _ast
        return list(_ast.literal_eval(m.group(1)))
    except Exception:
        return []


def _has_random_in_model(src: str) -> bool:
    """True if the model class uses a stochastic op (FP source)."""
    return bool(_RANDOMNESS_RX.search(src))


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis + repair
# ─────────────────────────────────────────────────────────────────────────────

MAX_REPAIR_ATTEMPTS = 3


def synthesize_tf_model(
    llm: OllamaClient,
    apis: list[str],
    ws_dir: Path,
    mid: int,
    x_val: np.ndarray,
    validator_timeout: int = 40,
) -> tuple[str | None, list[str], int]:
    """
    Generate a TF model class for the given API list and return
        (source_or_None, used_apis, attempts).
    Uses a self-repair loop of up to MAX_REPAIR_ATTEMPTS rounds.
    """
    prompt = build_tf_synthesis_prompt(apis)
    try:
        raw = llm.generate(prompt, advance=True)
    except Exception as exc:
        log.warning("m%04d: LLM error: %s", mid, exc)
        return None, [], 1

    src = _extract_class(raw)
    attempts = 1

    while attempts <= MAX_REPAIR_ATTEMPTS + 1:
        if src is None:
            err = "No `class Model(tf.keras.Model)` with `def call(...)` found."
        elif _has_random_in_model(src):
            err = ("Determinism violation: stochastic op found inside Model "
                   "(tf.random / Dropout / GaussianNoise / GaussianDropout). "
                   "Remove every stochastic op from the class.")
        else:
            err = _cpu_validate(src, x_val, ws_dir, mid, validator_timeout)
            if err is None:
                used = _extract_used_apis(src)
                return src, used, attempts

        if attempts > MAX_REPAIR_ATTEMPTS:
            return None, [], attempts

        log.info("m%04d: repair round %d — %s", mid, attempts, (err or "")[:120])
        repair_prompt = build_tf_repair_prompt(src or "<no class>", err)
        try:
            raw = llm.generate(repair_prompt, advance=False)
        except Exception as exc:
            log.warning("m%04d: repair LLM error: %s", mid, exc)
            return None, [], attempts
        src = _extract_class(raw)
        attempts += 1

    return None, [], attempts


def _cpu_validate(
    src: str, x_np: np.ndarray, ws_dir: Path, mid: int, timeout: int,
) -> str | None:
    """Return None on success, or a short error string."""
    script_path = ws_dir / f"validate_{mid:04d}_{os.getpid()}.py"
    inp_path = ws_dir / f"validate_inp_{mid:04d}_{os.getpid()}.npy"
    try:
        script_path.write_text(_VALIDATOR.format(model_src=src))
        np.save(str(inp_path), x_np)
        r = subprocess.run(
            [sys.executable, str(script_path), str(inp_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (r.stdout or "").strip()
        if out.startswith("OK"):
            return None
        if out.startswith("FAIL:"):
            return out[5:][:400]
        return (r.stderr or out or f"exit {r.returncode}")[:400]
    except subprocess.TimeoutExpired:
        return f"CPU validation timed out after {timeout}s"
    except Exception as exc:
        return f"CPU validator error: {exc}"
    finally:
        for p in (script_path, inp_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Comparison run
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    src: str, x_np: np.ndarray, ws_dir: Path, mid: int, timeout: int = 90,
) -> dict:
    """Run the LLM-generated model on CPU + GPU (GPU twice)."""
    script_path = ws_dir / f"run_{mid:04d}_{os.getpid()}.py"
    inp_path = ws_dir / f"inp_{mid:04d}_{os.getpid()}.npy"
    result: dict = {
        "cpu": None, "gpu": None, "gpu_repeat": None,
        "crash": None, "no_gpu": False, "cpu_dtype": None,
    }
    try:
        script_path.write_text(_WRAPPER.format(model_src=src))
        np.save(str(inp_path), x_np)
        proc = subprocess.run(
            [sys.executable, str(script_path), str(inp_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    result.update(json.loads(line))
                    break
                except json.JSONDecodeError:
                    pass
        if (proc.returncode != 0
                and result["cpu"] is None
                and result["crash"] is None):
            result["crash"] = (proc.stderr or proc.stdout or "")[-400:] \
                              or f"exit {proc.returncode}"
    except subprocess.TimeoutExpired:
        result["crash"] = "TIMEOUT"
    except Exception as exc:
        result["crash"] = str(exc)
    finally:
        for p in (script_path, inp_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────

# Dtype-aware (rtol, atol). Mirrors the PyTorch oracle; keeps the paper's
# "standard floating-point comparison" semantics but uses relative error so
# large-magnitude outputs (e.g. under scale_large mutation) do not FP the
# L2 threshold.
_TOL = {
    "float64": (1e-5, 1e-8),
    "float32": (1e-4, 1e-5),
    "bfloat16": (1e-2, 1e-2),
    "float16": (1e-2, 1e-3),
}

_NAN_ASYM_FRAC = 1e-3
_NAN_ASYM_MIN = 4

# Infrastructure errors that must never be classified as a library bug.
_INFRA_ERROR_RX = re.compile(
    r"out of memory"
    r"|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorOutOfMemory"
    r"|ResourceExhaustedError|RESOURCE_EXHAUSTED"
    r"|CUDNN_STATUS_NOT_INITIALIZED|CUBLAS_STATUS_NOT_INITIALIZED"
    r"|Failed to initialize CUDA|driver shutting down"
    r"|NCCL error|no kernel image is available for execution",
    re.IGNORECASE,
)


def _is_infra_error(err: str | None) -> bool:
    return bool(err) and bool(_INFRA_ERROR_RX.search(err))


# TF APIs whose output is only defined up to a permutation / sign.
# Positional CPU-vs-GPU comparison false-positives on these.
_PERMUTATION_INVARIANT_TF_APIS = {
    "tf.linalg.eig", "tf.linalg.eigh", "tf.linalg.svd",
    "tf.linalg.qr", "tf.linalg.lu", "tf.linalg.cholesky",
    "tf.linalg.eigvals", "tf.linalg.eigvalsh",
    "tf.self_adjoint_eig", "tf.self_adjoint_eigvals",
    "tf.svd", "tf.qr",
    "tf.unique", "tf.unique_with_counts",
    "tf.sort", "tf.argsort", "tf.top_k", "tf.nn.top_k",
    "tf.math.top_k",
}


def _uses_permutation_invariant_tf(apis: list[str]) -> bool:
    return any(a in _PERMUTATION_INVARIANT_TF_APIS for a in apis)


def _relative_error(c: np.ndarray, g: np.ndarray, rtol: float, atol: float) -> float:
    finite = np.isfinite(c) & np.isfinite(g)
    if not np.any(finite):
        return 0.0
    cf = c[finite]; gf = g[finite]
    denom = atol + rtol * np.abs(cf)
    denom = np.where(denom > 0, denom, atol + 1e-12)
    return float(np.max(np.abs(cf - gf) / denom))


def _significant_asymmetry(count: int, total: int) -> bool:
    if total == 0 or count == 0:
        return False
    return count >= _NAN_ASYM_MIN and (count / total) >= _NAN_ASYM_FRAC


def compare_outputs(
    cpu_out, gpu_out, gpu_repeat=None, cpu_dtype: str | None = None,
    used_apis: list[str] | None = None,
) -> tuple[str | None, str]:
    """
    Classify (cpu, gpu, gpu_repeat) as CRASH / NAN / INCONSISTENT / None.

    • Symmetric NaN/Inf ⇒ no bug (expected math).
    • Asymmetric NaN/Inf positions ≥ max(4, 0.1%) ⇒ NAN bug.
    • Numeric divergence beyond dtype-aware rel_err, and beyond GPU-vs-GPU
      non-determinism, AND beyond sorted-magnitude tolerance for decomp
      APIs (eig/svd/qr/sort/…) ⇒ INCONSISTENT bug.
    """
    if cpu_out is None or gpu_out is None:
        return None, ""

    a = np.asarray(cpu_out, dtype=np.float64)
    b = np.asarray(gpu_out, dtype=np.float64)
    if a.size != b.size:
        return "INCONSISTENT", f"size mismatch cpu={a.size} gpu={b.size}"

    cpu_nan = np.isnan(a); gpu_nan = np.isnan(b)
    cpu_inf = np.isinf(a); gpu_inf = np.isinf(b)
    asym_nan = (cpu_nan & ~gpu_nan & ~gpu_inf) | (gpu_nan & ~cpu_nan & ~cpu_inf)
    asym_inf = (cpu_inf & ~gpu_inf & ~gpu_nan) | (gpu_inf & ~cpu_inf & ~cpu_nan)
    nan_vs_inf = (cpu_nan & gpu_inf) | (cpu_inf & gpu_nan)
    n_asym = int((asym_nan | asym_inf | nan_vs_inf).sum())
    total = int(a.size)
    if _significant_asymmetry(n_asym, total):
        return "NAN", (
            f"asym={n_asym}/{total} (asym_nan={int(asym_nan.sum())} "
            f"asym_inf={int(asym_inf.sum())} nan_vs_inf={int(nan_vs_inf.sum())})"
        )

    both_finite = ~(cpu_nan | gpu_nan | cpu_inf | gpu_inf)
    if not np.any(both_finite):
        return None, ""  # all non-finite but symmetric

    rtol, atol = _TOL.get(cpu_dtype or "float32", (1e-4, 1e-5))
    rel = _relative_error(a, b, rtol, atol)
    # Require a large margin above tolerance to call something a bug.
    # rel_err of a few × tolerance is expected for models using GradientTape,
    # BatchNorm, or reductions (summation-order differences in f32 kernels).
    # Real library bugs in the reference catalogue show rel_err of hundreds
    # to thousands; 100× tolerance is a conservative single-sided filter.
    BUG_MARGIN = 100.0
    if rel <= BUG_MARGIN:
        return None, ""

    # Permutation-invariant APIs: compare sorted magnitudes as a secondary
    # check. If sorted agrees, positional divergence is a permutation
    # artifact, NOT a library bug.
    if used_apis and _uses_permutation_invariant_tf(used_apis):
        ca = np.abs(a[both_finite]); gb_m = np.abs(b[both_finite])
        if ca.size == gb_m.size and ca.size > 0:
            ca_s = np.sort(ca); gb_s = np.sort(gb_m)
            denom = atol + rtol * np.abs(ca_s)
            denom = np.where(denom > 0, denom, atol + 1e-12)
            sorted_rel = float(np.max(np.abs(ca_s - gb_s) / denom))
            if sorted_rel <= 1.0:
                return None, (
                    f"rel_err={rel:.3e} but sorted_rel={sorted_rel:.3e} ≤ 1.0 "
                    f"→ permutation artifact on a decomp API, not a bug"
                )

    # GPU non-determinism floor.
    nd_floor = 0.0
    if gpu_repeat is not None:
        g2 = np.asarray(gpu_repeat, dtype=np.float64)
        if g2.size == b.size:
            nd_floor = _relative_error(b, g2, rtol, atol)
    # If the model is highly non-deterministic on a single device
    # (nd_floor already far above tolerance), any CPU-vs-GPU diff is noise,
    # not a library bug. Require a strong 10× margin over the nondet floor
    # AND reject outright when the floor is already very high.
    if nd_floor >= 100.0:
        return None, (
            f"rel_err={rel:.3e} but GPU is highly non-deterministic "
            f"(floor={nd_floor:.3e}); not a library bug"
        )
    if rel < max(1.0, 10.0 * nd_floor):
        return None, f"rel_err={rel:.3e} within 10×nondet_floor={nd_floor:.3e}"

    return "INCONSISTENT", (
        f"rel_err={rel:.3e} > {BUG_MARGIN:.1f}×tol  rtol={rtol:.0e} atol={atol:.0e} "
        f"nondet_floor={nd_floor:.3e} dtype={cpu_dtype} n={total}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mutations (paper §3.2 — same order as PT)
# ─────────────────────────────────────────────────────────────────────────────

def _mut_add_noise(x: np.ndarray) -> np.ndarray:
    return x + np.random.randn(*x.shape).astype(np.float32) * 0.1


def _mut_scale_small(x: np.ndarray) -> np.ndarray:
    return x * np.float32(np.random.uniform(0.01, 0.1))


def _mut_mask(x: np.ndarray) -> np.ndarray:
    m = (np.random.rand(*x.shape) < 0.3).astype(np.float32)
    return (x * (1.0 - m)).astype(np.float32)


def _mut_uniform(x: np.ndarray) -> np.ndarray:
    val = np.float32(np.random.randn())
    return np.full_like(x, val, dtype=np.float32)


def _mut_scale_large(x: np.ndarray) -> np.ndarray:
    return x * np.float32(np.random.uniform(1e3, 1e6))


_MUTATIONS = {
    "add_noise":   _mut_add_noise,
    "scale_small": _mut_scale_small,
    "mask":        _mut_mask,
    "uniform":     _mut_uniform,
    "scale_large": _mut_scale_large,
}
_MUT_NAMES = list(_MUTATIONS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="SMOLFuzz TF (CPU vs GPU)")
    ap.add_argument("--models", type=int, default=300,
                    help="Number of models to synthesise")
    ap.add_argument("--budget", type=int, default=60,
                    help="Mutation fuzzing budget in seconds per model")
    ap.add_argument("--api-set-size", type=int, default=30,
                    help="APIs per synthesised model")
    ap.add_argument("--out", default=str(HERE / "results" / "tf_run"),
                    help="Output directory")
    ap.add_argument("--apis", default=str(HERE / "tf_valid_apis.txt"),
                    help="Path to the TF API list")
    ap.add_argument("--seed", type=int, default=2026,
                    help="RNG seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir   = Path(args.out)
    bug_dir   = out_dir / "bugs"
    model_dir = out_dir / "models"
    ws_dir    = out_dir / "workspace"
    for d in (bug_dir, model_dir, ws_dir):
        d.mkdir(parents=True, exist_ok=True)

    api_path = Path(args.apis)
    log.info("Loading TF APIs from %s", api_path)
    groups = load_and_classify(api_path, framework="tf")
    log.info("API classification:\n%s", group_summary(groups))

    selectable = {k: v for k, v in groups.items() if k != "_excluded"}
    selector = MultiRouletteSelector(selectable)
    llm = OllamaClient()

    rng = np.random.RandomState(args.seed)

    stats = dict(
        models=0, failed_synth=0, invalid_models=0, rejected_random=0,
        no_gpu=0, bugs=0, clean=0, mutations_run=0,
        bug_types=dict(CRASH=0, NAN=0, INCONSISTENT=0),
    )
    bug_log: list[dict] = []

    # Coverage tracking
    all_apis = set()
    for apis in selectable.values():
        all_apis.update(apis)
    apis_attempted: set[str] = set()
    apis_executed: set[str]  = set()
    coverage_history: list[tuple[int, int]] = []

    log.info("SMOLFuzz TF (CPU vs GPU) | models=%d budget=%ds → %s",
             args.models, args.budget, out_dir)

    for mid in range(1, args.models + 1):
        stats["models"] += 1
        log.info("=" * 60)
        log.info("Model %d / %d", mid, args.models)

        # 1. Select APIs (multi-roulette)
        apis = selector.select(n=args.api_set_size)
        apis_attempted.update(apis)
        log.info("Selected %d APIs", len(apis))

        # 2. Baseline input (identical across devices; fixed seed ⇒ reproducible)
        x_base = rng.randn(4, 8).astype(np.float32)

        # 3. Synthesize + repair
        src, used_apis, attempts = synthesize_tf_model(
            llm, apis, ws_dir, mid, x_base,
        )
        if src is None:
            log.warning("m%04d: synthesis failed after %d attempts", mid, attempts)
            stats["failed_synth"] += 1
            selector.record_usage(apis, triggered_bug=False)
            continue

        model_file = model_dir / f"model_{mid:04d}.py"
        model_file.write_text(
            f"# SMOLFuzz TF model {mid} | attempts={attempts}\n"
            f"# APIS_SELECTED = {apis}\n"
            f"# USED_APIS = {used_apis}\n\n{src}\n"
        )

        # 4. Baseline CPU vs GPU — if baseline fails, the LLM code is bad;
        #    count as invalid_model, NEVER as a library bug.
        r_base = run_comparison(src, x_base, ws_dir, mid, timeout=90)
        if r_base.get("no_gpu"):
            log.error("No GPU available — aborting TF run")
            stats["no_gpu"] += 1
            break
        if r_base.get("crash"):
            tag = "infra" if _is_infra_error(r_base["crash"]) else "invalid"
            log.warning("m%04d: baseline crash (%s model): %s",
                        mid, tag, (r_base["crash"] or "")[:100])
            stats["invalid_models"] += 1
            selector.record_usage(used_apis, triggered_bug=False)
            continue
        if r_base["cpu"] is None or r_base["gpu"] is None:
            log.warning("m%04d: baseline missing output — invalid", mid)
            stats["invalid_models"] += 1
            selector.record_usage(used_apis, triggered_bug=False)
            continue

        # Baseline must itself be consistent on CPU vs GPU; otherwise the model
        # is numerically unstable by construction and mutation-time divergence
        # is not a library bug.
        base_type, base_detail = compare_outputs(
            r_base["cpu"], r_base["gpu"],
            gpu_repeat=r_base.get("gpu_repeat"),
            cpu_dtype=r_base.get("cpu_dtype"),
            used_apis=used_apis,
        )
        if base_type is not None:
            log.warning(
                "m%04d: baseline already divergent (%s: %s) — invalid model",
                mid, base_type, base_detail[:100],
            )
            stats["invalid_models"] += 1
            selector.record_usage(used_apis, triggered_bug=False)
            continue

        apis_executed.update(used_apis)
        coverage_history.append((mid, len(apis_executed)))
        log.info("m%04d: baseline OK — fuzzing for %ds", mid, args.budget)

        # 5. Mutation fuzzing — 60s budget
        found_bug = False
        mut_count = 0
        bstart = time.time()
        while time.time() - bstart < args.budget:
            mut_name = random.choice(_MUT_NAMES)
            x_mut = _MUTATIONS[mut_name](x_base.copy())
            mut_count += 1
            stats["mutations_run"] += 1

            remaining = args.budget - (time.time() - bstart)
            if remaining <= 5:
                break
            run_timeout = min(int(remaining) + 10, 120)

            r = run_comparison(src, x_mut, ws_dir, mid, timeout=run_timeout)

            bug_type, detail = None, ""

            if r.get("crash"):
                # Infrastructure-level crashes (OOM, driver resets) are not
                # library bugs — the user explicitly rejects false positives.
                # Scan the FULL error string, not the truncated head.
                if _is_infra_error(r["crash"]):
                    log.info("  m%04d mut %s: infra error (%s) — skipping",
                             mid, mut_name, (r["crash"] or "")[-100:])
                    continue
                # Mutation-time crash is a bug (baseline was clean).
                bug_type = "CRASH"
                detail = (r["crash"] or "")[-400:]
            else:
                bug_type, detail = compare_outputs(
                    r["cpu"], r["gpu"],
                    gpu_repeat=r.get("gpu_repeat"),
                    cpu_dtype=r.get("cpu_dtype"),
                    used_apis=used_apis,
                )

            if bug_type:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                bname = f"bug_{bug_type.lower()}_m{mid:04d}_{mut_name}_{ts}"
                entry = dict(
                    bug_type=bug_type.lower(), model_id=mid,
                    mutation=mut_name, detail=detail,
                    apis_selected=apis, used_apis=used_apis,
                    timestamp=ts,
                    model_file=str(model_file),
                )
                (bug_dir / f"{bname}.json").write_text(json.dumps(entry, indent=2))
                np.save(str(bug_dir / f"{bname}.inputs.npy"), x_mut)

                stats["bugs"] += 1
                stats["bug_types"][bug_type] = stats["bug_types"].get(bug_type, 0) + 1
                bug_log.append(entry)
                log.warning("  BUG %s [%s] %s", bug_type, mut_name, detail[:90])
                found_bug = True
                break

        if found_bug:
            log.info("m%04d: bug found after %d mutations (%.1fs)",
                     mid, mut_count, time.time() - bstart)
        else:
            stats["clean"] += 1
            log.info("m%04d: clean after %d mutations (%.1fs budget)",
                     mid, mut_count, time.time() - bstart)

        # 6. Feedback → selector. APIs associated with bugs are exempt from
        #    penalty (score unchanged), so they keep being sampled.
        selector.record_usage(used_apis, triggered_bug=found_bug)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(
        "DONE | models=%d failed_synth=%d invalid=%d no_gpu=%d "
        "bugs=%d clean=%d mutations=%d",
        stats["models"], stats["failed_synth"], stats["invalid_models"],
        stats["no_gpu"], stats["bugs"], stats["clean"], stats["mutations_run"],
    )
    log.info("Bug types: %s", stats["bug_types"])

    # Coverage
    n_total = len(all_apis)
    n_att   = len(apis_attempted)
    n_exec  = len(apis_executed)
    log.info("API COVERAGE  total=%d attempted=%d executed=%d",
             n_total, n_att, n_exec)
    coverage_data = {
        "n_models": args.models,
        "n_total_apis": n_total,
        "n_attempted": n_att,
        "n_executed": n_exec,
        "attempted": sorted(apis_attempted),
        "executed": sorted(apis_executed),
        "coverage_history": coverage_history,
        "per_group": {
            g: {"total": len(apis),
                "executed": sum(1 for a in apis if a in apis_executed)}
            for g, apis in selectable.items()
        },
        "totals": stats,
    }
    (out_dir / "coverage.json").write_text(json.dumps(coverage_data, indent=2))

    summary_lines = [
        "SMOLFuzz TF (CPU vs GPU) — COMPLETED",
        f"Finished: {datetime.now()}",
        "",
        "=== STATS ===",
        f"models={stats['models']}  failed_synth={stats['failed_synth']}  "
        f"invalid={stats['invalid_models']}  no_gpu={stats['no_gpu']}",
        f"bugs={stats['bugs']}  clean={stats['clean']}  "
        f"mutations_run={stats['mutations_run']}",
        "",
        "=== BUG TYPES ===",
        *[f"  {k}: {v}" for k, v in stats["bug_types"].items()],
        f"  TOTAL: {stats['bugs']}",
        "",
        "=== BUG DETAILS (one per model) ===",
        f"{'Model':<6}  {'Type':<14} {'Mutation':<14} {'Detail'}",
        f"{'-----':<6}  {'----':<14} {'--------':<14} {'------'}",
        *[
            f"{b['model_id']:<6}  {b['bug_type'].upper():<14} "
            f"{b['mutation']:<14} {b['detail'][:70]}"
            for b in bug_log
        ],
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines))
    log.info("Summary → %s", out_dir / "summary.txt")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
