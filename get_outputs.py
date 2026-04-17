#!/usr/bin/env python3
"""Print actual CPU and GPU output values for all 27 bugs."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent / "bugs/github/new"
PT_DIR = BASE / "pytorch"
TF_DIR = BASE / "tensorflow"
MODELS = Path(__file__).parent / "results/clean_run_tf/models"

# ── PyTorch ───────────────────────────────────────────────────────────────────
PT_SCRIPT = """
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try: torch.use_deterministic_algorithms(True, warn_only=True)
except: pass

{model_src}

_here = Path("{model_dir}")
inputs = torch.load(_here / "{inputs_file}", weights_only=False)

torch.manual_seed(42); torch.cuda.manual_seed_all(42)
cpu_model = Model().cpu().eval()
gpu_model = Model().cuda().eval()
gpu_model.load_state_dict(cpu_model.state_dict())

cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
gpu_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
with torch.set_grad_enabled(True):
    cpu_out = cpu_model(*cpu_inputs)
    gpu_out = gpu_model(*gpu_inputs)

cpu_outs = [cpu_out] if isinstance(cpu_out, torch.Tensor) else list(cpu_out)
gpu_outs = [gpu_out] if isinstance(gpu_out, torch.Tensor) else list(gpu_out)
for i,(c,g) in enumerate(zip(cpu_outs, gpu_outs)):
    if isinstance(c, torch.Tensor):
        print(f"CPU[{{i}}]: {{c.detach().cpu().numpy().flatten()[:8].tolist()}}")
        print(f"GPU[{{i}}]: {{g.detach().cpu().numpy().flatten()[:8].tolist()}}")
"""

def run_pt_bug(repro_py: Path):
    json_f = Path(str(repro_py).replace(".repro.py", ".json"))
    info = json.loads(json_f.read_text())
    model_f = PT_DIR / f"model_{info['model_id']:04d}.py"
    src = model_f.read_text().splitlines()
    start = next((i for i,l in enumerate(src) if l.startswith("class Model")), None)
    model_src = "\n".join(src[start:])
    inputs_file = info["artifacts"]["inputs"]
    script = PT_SCRIPT.format(model_src=model_src, model_dir=str(PT_DIR), inputs_file=inputs_file)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); spath = f.name
    try:
        r = subprocess.run([sys.executable, spath], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or r.stderr[-200:]
    finally:
        os.unlink(spath)

# ── TensorFlow ────────────────────────────────────────────────────────────────
TF_SCRIPT = """
import os, sys, traceback
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
    print(f"CPU: {c}")
    print(f"GPU: {g}")
except Exception:
    print("ERROR:", traceback.format_exc()[-200:])
"""

def run_tf_bug(model_py: Path, inputs_npy: Path):
    src = model_py.read_text().splitlines()
    start = next((i for i,l in enumerate(src) if l.startswith("class Model")), None)
    model_src = "\n".join(src[start:])
    script = TF_SCRIPT.format(model_src=model_src)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(script); spath = f.name
    try:
        r = subprocess.run([sys.executable, spath, str(inputs_npy)],
                           capture_output=True, text=True, timeout=90)
        return r.stdout.strip() or r.stderr[-200:]
    finally:
        os.unlink(spath)

if __name__ == "__main__":
    print("=== PYTORCH ===")
    for rp in sorted(PT_DIR.glob("*.repro.py")):
        mid = json.loads(Path(str(rp).replace(".repro.py",".json")).read_text())["model_id"]
        print(f"\n--- PT m{mid:04d} ---")
        print(run_pt_bug(rp))

    print("\n=== TENSORFLOW ===")
    for jf in sorted(TF_DIR.glob("*.json")):
        info = json.loads(jf.read_text())
        mid = info["model_id"]
        npy = TF_DIR / jf.name.replace(".json", ".inputs.npy")
        mpy = MODELS / f"model_{mid:04d}.py"
        print(f"\n--- TF m{mid:04d} ({info['mutation']}) ---")
        print(run_tf_bug(mpy, npy))
