#!/usr/bin/env python3
import os, sys, json, subprocess, tempfile
from pathlib import Path
import numpy as np

RTOL, ATOL = 1e-4, 1e-5

_WRAPPER = r"""
import os, sys, json, traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
import numpy as np
import tensorflow as tf
for _g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except: pass

{model_src}

x_np = np.load(sys.argv[1]).astype(np.float32)
result = dict(cpu=None, gpu=None, gpu2=None, crash=None)
try:
    def flat(t): return np.array(t, dtype=np.float64).flatten().tolist()
    tf.random.set_seed(42)
    with tf.device('/CPU:0'):
        mc = Model(); xc = tf.constant(x_np); _ = mc(xc, training=False)
    tf.random.set_seed(42)
    with tf.device('/GPU:0'):
        mg = Model(); xg = tf.constant(x_np); _ = mg(xg, training=False)
    if len(mc.variables) == len(mg.variables):
        for vc, vg in zip(mc.variables, mg.variables):
            vg.assign(tf.cast(vc, vg.dtype))
    with tf.device('/CPU:0'): out_c = mc(xc, training=False)
    with tf.device('/GPU:0'):
        out_g1 = mg(xg, training=False)
        out_g2 = mg(xg, training=False)
    result['cpu'] = flat(out_c)
    result['gpu'] = flat(out_g1)
    result['gpu2'] = flat(out_g2)
except Exception:
    result['crash'] = traceback.format_exc()[-600:]
print(json.dumps(result))
"""

def check(model_py: Path, inputs_npy: Path, label: str):
    src = model_py.read_text().splitlines()
    start = next((i for i, l in enumerate(src) if l.startswith("class Model")), None)
    if start is None:
        print(f"  [{label}] no class Model"); return
    model_src = "\n".join(src[start:])
    script = _WRAPPER.format(model_src=model_src)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); spath = f.name
    try:
        r = subprocess.run([sys.executable, spath, str(inputs_npy)],
                           capture_output=True, text=True, timeout=90)
        raw = r.stdout.strip()
        if not raw:
            print(f"  [{label}] CRASH/empty: {r.stderr[-200:]}"); return
        res = json.loads(raw)
        if res.get("crash"):
            print(f"  [{label}] EXCEPTION: {res['crash'][-150:]}"); return
        cpu = np.array(res["cpu"]); gpu = np.array(res["gpu"]); gpu2 = np.array(res["gpu2"])
        fin = np.isfinite(cpu) & np.isfinite(gpu)
        if not fin.any():
            print(f"  [{label}] all non-finite (asymmetric inf/nan) → CONFIRMED"); return
        cf, gf = cpu[fin], gpu[fin]
        rel = np.max(np.abs(cf - gf) / (ATOL + RTOL * np.abs(cf)))
        g2f = gpu2[fin] if gpu2.shape == gpu.shape else gf
        nd = np.max(np.abs(gf - g2f) / (ATOL + RTOL * np.abs(gf)))
        tag = "CONFIRMED" if rel > 10 else "WEAK"
        if nd >= rel / 10:
            tag = f"NONDET (nondet={nd:.2e})"
        print(f"  [{label}] rel_err={rel:.3e}  nondet={nd:.3e}  → {tag}")
    finally:
        os.unlink(spath)

if __name__ == "__main__":
    TF_DIR = Path(__file__).parent / "bugs/github/new/tensorflow"
    MODELS = Path(__file__).parent / "results/clean_run_tf/models"
    for jf in sorted(TF_DIR.glob("*.json")):
        info = json.loads(jf.read_text())
        mid = info["model_id"]
        check(MODELS / f"model_{mid:04d}.py",
              TF_DIR / jf.name.replace(".json", ".inputs.npy"),
              f"tf_m{mid:04d} {info['mutation']}")
