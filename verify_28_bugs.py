#!/usr/bin/env python3
"""Verify all 28 bugs from the clean run are real CPU/GPU divergences."""
import json, os, subprocess, sys, tempfile, textwrap
from pathlib import Path
import numpy as np

BUGS_DIR = Path(__file__).parent / "bugs/github/new"
PT_DIR   = BUGS_DIR / "pytorch"
TF_DIR   = BUGS_DIR / "tensorflow"
MODELS_DIR = Path(__file__).parent / "results/clean_run_tf/models"

RTOL, ATOL = 1e-4, 1e-5
MIN_MARGIN = 10.0  # rel_err must still be > 10× tolerance to confirm

# ─── PyTorch ──────────────────────────────────────────────────────────────────

def run_pt_repro(repro_py: Path) -> tuple[bool, str]:
    """Run a PT repro.py and check if it still prints a high rel_err."""
    r = subprocess.run(
        [sys.executable, str(repro_py)],
        capture_output=True, text=True, timeout=60,
        cwd=str(repro_py.parent),
    )
    out = r.stdout + r.stderr
    # Look for "rel_err=X.XXXeYY" in output
    import re
    m = re.search(r"rel_err=([\d.e+\-]+)", out)
    if m:
        rel_err = float(m.group(1))
        confirmed = rel_err > MIN_MARGIN
        return confirmed, f"rel_err={rel_err:.3e} ({'CONFIRMED' if confirmed else 'WEAK'})"
    if "all non-finite" in out:
        return True, "all non-finite (confirmed asymmetric)"
    if r.returncode != 0:
        return False, f"CRASH: {out[-300:]}"
    return False, f"no divergence detected. stdout={out[:300]}"


# ─── TensorFlow ───────────────────────────────────────────────────────────────

_TF_WRAPPER = r"""
import os, sys, json, traceback, re
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

{model_src}

x_np = np.load(sys.argv[1]).astype(np.float32)
result = dict(cpu=None, gpu=None, gpu_repeat=None, crash=None)

try:
    def _flatten(t):
        return np.array(t, dtype=np.float64).flatten().tolist()

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

except Exception:
    result['crash'] = traceback.format_exc()[-800:]

print(json.dumps(result))
"""


def run_tf_bug(model_py: Path, inputs_npy: Path) -> tuple[bool, str]:
    src_lines = model_py.read_text().splitlines()
    # Strip comment header lines, keep class definition onward
    class_start = next((i for i, l in enumerate(src_lines) if l.startswith("class Model")), None)
    if class_start is None:
        return False, "no class Model found in model file"
    model_src = "\n".join(src_lines[class_start:])

    script = _TF_WRAPPER.format(model_src=model_src)

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        r = subprocess.run(
            [sys.executable, script_path, str(inputs_npy)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False, f"CRASH/empty: {(r.stderr or r.stdout)[-300:]}"

        res = json.loads(r.stdout.strip())
        if res.get("crash"):
            return False, f"exception: {res['crash'][-200:]}"

        cpu = np.array(res["cpu"])
        gpu = np.array(res["gpu"])
        gpu2 = np.array(res["gpu_repeat"])

        finite = np.isfinite(cpu) & np.isfinite(gpu)
        if not finite.any():
            return True, "all non-finite (confirmed asymmetric inf/nan)"

        cpu_f, gpu_f = cpu[finite], gpu[finite]
        rel_err = np.max(np.abs(cpu_f - gpu_f) / (ATOL + RTOL * np.abs(cpu_f)))

        # nondet check
        gpu_f2 = gpu2[finite]
        nondet = np.max(np.abs(gpu_f - gpu_f2) / (ATOL + RTOL * np.abs(gpu_f))) if gpu_f2.shape == gpu_f.shape else 0.0

        if nondet >= rel_err / MIN_MARGIN:
            return False, f"NONDET: rel_err={rel_err:.3e} but nondet_floor={nondet:.3e}"

        confirmed = rel_err > MIN_MARGIN
        return confirmed, f"rel_err={rel_err:.3e} nondet={nondet:.3e} ({'CONFIRMED' if confirmed else 'WEAK'})"

    finally:
        os.unlink(script_path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = []

    # --- PyTorch bugs ---
    pt_repros = sorted(PT_DIR.glob("*.repro.py"))
    print(f"\n{'='*60}")
    print(f"PyTorch bugs: {len(pt_repros)}")
    print(f"{'='*60}")
    for repro in pt_repros:
        bug_id = repro.stem.replace(".repro", "")
        try:
            ok, msg = run_pt_repro(repro)
        except Exception as e:
            ok, msg = False, f"ERROR: {e}"
        status = "REAL" if ok else "FP?"
        print(f"  [{status}] {bug_id}\n         {msg}")
        results.append((bug_id, "pytorch", ok, msg))

    # --- TensorFlow bugs ---
    tf_jsons = sorted(TF_DIR.glob("*.json"))
    print(f"\n{'='*60}")
    print(f"TensorFlow bugs: {len(tf_jsons)}")
    print(f"{'='*60}")
    for jf in tf_jsons:
        info = json.loads(jf.read_text())
        mid = info["model_id"]
        model_py = MODELS_DIR / f"model_{mid:04d}.py"
        inputs_npy = TF_DIR / jf.name.replace(".json", ".inputs.npy")

        if not model_py.exists():
            print(f"  [FP?] m{mid:04d} — model file missing: {model_py}")
            results.append((f"tf_m{mid:04d}", "tensorflow", False, "model file missing"))
            continue
        if not inputs_npy.exists():
            print(f"  [FP?] m{mid:04d} — inputs.npy missing")
            results.append((f"tf_m{mid:04d}", "tensorflow", False, "inputs.npy missing"))
            continue

        try:
            ok, msg = run_tf_bug(model_py, inputs_npy)
        except Exception as e:
            ok, msg = False, f"ERROR: {e}"
        status = "REAL" if ok else "FP?"
        print(f"  [{status}] tf_m{mid:04d} ({info['mutation']})\n         {msg}")
        results.append((f"tf_m{mid:04d}", "tensorflow", ok, msg))

    # --- Summary ---
    real  = [r for r in results if r[2]]
    fp    = [r for r in results if not r[2]]
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(real)}/{len(results)} confirmed real bugs")
    print(f"  REAL : {len(real)}")
    print(f"  FP?  : {len(fp)}")
    if fp:
        print("\nSuspect bugs:")
        for bid, fw, _, msg in fp:
            print(f"  {fw} {bid}: {msg[:100]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
