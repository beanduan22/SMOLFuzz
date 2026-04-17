#!/usr/bin/env python3
"""Print actual CPU and GPU output values for all TF bugs."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

TF_DIR = Path(__file__).parent / "bugs/github/new/tensorflow"
MODELS = Path(__file__).parent / "results/clean_run_tf/models"
RTOL, ATOL = 1e-4, 1e-5

_WRAPPER = '''
import os, sys, json, traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
import numpy as np
import tensorflow as tf
for _g in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(_g, True)
    except: pass

MODELSRC

x_np = np.load(sys.argv[1]).astype(np.float32)
try:
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
    with tf.device('/GPU:0'): out_g = mg(xg, training=False)
    c = np.array(out_c, dtype=np.float64).flatten()[:8].tolist()
    g = np.array(out_g, dtype=np.float64).flatten()[:8].tolist()
    result = dict(cpu=c, gpu=g, crash=None)
except Exception:
    result = dict(cpu=None, gpu=None, crash=traceback.format_exc()[-400:])
print(json.dumps(result))
'''

def run(model_py, inputs_npy):
    src = model_py.read_text().splitlines()
    start = next((i for i, l in enumerate(src) if l.startswith("class Model")), None)
    model_src = "\n".join(src[start:])
    script = _WRAPPER.replace("MODELSRC", model_src)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); spath = f.name
    try:
        r = subprocess.run([sys.executable, spath, str(inputs_npy)],
                           capture_output=True, text=True, timeout=90)
        if not r.stdout.strip():
            return None, None, r.stderr[-200:]
        res = json.loads(r.stdout.strip())
        return res.get("cpu"), res.get("gpu"), res.get("crash")
    finally:
        os.unlink(spath)

if __name__ == "__main__":
    for jf in sorted(TF_DIR.glob("*.json")):
        info = json.loads(jf.read_text())
        mid = info["model_id"]
        npy = TF_DIR / jf.name.replace(".json", ".inputs.npy")
        mpy = MODELS / f"model_{mid:04d}.py"
        cpu, gpu, err = run(mpy, npy)
        print(f"--- TF m{mid:04d} ({info['mutation']}) ---")
        if err:
            print(f"  ERROR: {err}")
        else:
            print(f"  CPU: {cpu}")
            print(f"  GPU: {gpu}")
        print()
