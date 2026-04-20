"""
Subprocess-based model executor + 5 mutation strategies.

Each model run forks a child process that:
  1. Executes the LLM-generated code
  2. Runs Model(*inputs) on the requested device
  3. Saves the result to a temp file (torch.save)

This isolates crashes (SIGSEGV, SIGABRT) from the main process.

Mutation strategies (paper §3.2):
  M1  add_noise      – add Gaussian noise (σ=0.1)
  M2  scale_small    – multiply by U(0.01, 0.1)
  M3  mask           – zero out random 30% of elements
  M4  uniform        – set all elements to a constant
  M5  scale_large    – multiply by U(1e3, 1e6)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEVICE_TIMEOUT = 30    # seconds per device run (paper §3.1.3)
MUTATION_TIMEOUT = 60  # seconds per mutation run


# ------------------------------------------------------------------ #
# Runner template injected around the LLM code                       #
# ------------------------------------------------------------------ #

_RUNNER_SUFFIX = textwrap.dedent("""
# ===================== SMOLFuzz runner =====================
# False-positive prevention:
#   • .eval() disables Dropout/BatchNorm stochastic behavior
#   • TF32 off → GPU matmul matches CPU float32 (not ~1e-3 looser)
#   • cuDNN deterministic=True, benchmark=False → stable kernel choice
#   • CUBLAS_WORKSPACE_CONFIG :4096:8 → required for deterministic matmul
#   • use_deterministic_algorithms(True, warn_only=True) → catches non-det ops
#   • The same run is executed TWICE on GPU; a non-matching pair is marked
#     "nondet" so the oracle can drop it instead of reporting a false bug.
if __name__ == "__main__":
    import argparse, os, sys, traceback
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--double-run", action="store_true",
                        help="Run forward twice and include both outputs")
    args = parser.parse_args()

    # ----- Determinism / precision knobs -----
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        if args.input_file:
            inputs = torch.load(args.input_file, weights_only=False)
        else:
            torch.manual_seed(42)
            inputs = make_inputs()
            save_path = args.output_file + ".inputs.pt"
            torch.save(inputs, save_path)

        device = args.device
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = Model().to(device)
        model.eval()                       # disable Dropout / BatchNorm updates

        device_inputs = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                device_inputs.append(x.to(device))
            else:
                device_inputs.append(x)

        def to_cpu_list(out):
            if isinstance(out, torch.Tensor):
                return [out.detach().cpu()]
            elif isinstance(out, (list, tuple)):
                flat = []
                for o in out:
                    if isinstance(o, torch.Tensor):
                        flat.append(o.detach().cpu())
                return flat
            else:
                try:
                    return [torch.tensor(out)]
                except Exception:
                    return []

        # NB: not `torch.inference_mode()` — the paper explicitly requires
        # computational-graph / autograd dependencies to be exercised, which
        # inference_mode() forbids (in-place ops, grad, etc.). We leave grad
        # tracking enabled and rely on the model's own context managers
        # (no_grad, enable_grad, GradientTape) to drive the semantics.
        with torch.set_grad_enabled(True):
            output_a = model(*device_inputs)
        outputs_a = to_cpu_list(output_a)

        outputs_b = None
        if args.double_run:
            # Second run on the same device with identical inputs.
            # Any difference between outputs_a and outputs_b is
            # DEVICE-LEVEL non-determinism, not a CPU/GPU library bug.
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            model2 = Model().to(device)
            model2.eval()
            device_inputs2 = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    device_inputs2.append(x.to(device))
                else:
                    device_inputs2.append(x)
            with torch.set_grad_enabled(True):
                output_b = model2(*device_inputs2)
            outputs_b = to_cpu_list(output_b)

        result = {"status": "ok", "outputs": outputs_a, "outputs_repeat": outputs_b}
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    torch.save(result, args.output_file)
""")


def _strip_main_block(code: str) -> str:
    """Remove any `if __name__ == '__main__':` block from LLM-generated code."""
    import re
    # Match from `if __name__` to end of file, handling both quote styles
    cleaned = re.sub(
        r'\nif\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:.*',
        '',
        code,
        flags=re.DOTALL,
    )
    return cleaned.rstrip()


# Regex that catches LLM-inserted randomness / non-determinism.
# If the MODEL CODE (inside forward, Model, or make_inputs) contains any of
# these, we mark the model as generation-invalid: CPU and GPU RNG streams
# differ so comparing them is guaranteed to false-positive.
#
# We deliberately match inside the whole generated script (not only forward)
# because many LLM-generated models put the random call inside Model.__init__
# or helper functions.
import re as _re

_RANDOMNESS_RX = _re.compile(
    r"\b("
    r"torch\.rand(?:n|int|perm)?(?:_like)?"
    r"|torch\.normal|torch\.bernoulli|torch\.multinomial|torch\.poisson"
    r"|torch\.manual_seed|torch\.seed"
    r"|nn\.Dropout\d?d?|F\.dropout\d?d?"
    r"|nn\.FeatureAlphaDropout|nn\.AlphaDropout"
    r"|torch\.randperm"
    r"|\.bernoulli_?\(|\.normal_\(|\.uniform_\(|\.cauchy_\("
    r"|\.exponential_\(|\.geometric_\(|\.log_normal_\("
    r"|\.random_\("
    r")"
)

# Ops known to be non-deterministic on CUDA (per the PyTorch docs —
# https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).
# Models using any of these will get a diff flagged as "nondet" rather than
# a genuine CPU/GPU library bug.
_NONDET_GPU_RX = _re.compile(
    r"\b("
    # Scatter / indexed writes with floating-point values
    r"torch\.(scatter_add|index_add|index_put|index_copy"
    r"|index_reduce|scatter_reduce|scatter)"
    r"|\.scatter_add_?\(|\.index_add_?\(|\.index_put_?\("
    r"|\.scatter_reduce_?\("
    # Reductions with non-deterministic CUDA kernels
    r"|torch\.(kthvalue|median|cumsum|cumprod|nanmedian|nansum)"
    r"|\.kthvalue\(|\.median\("
    # Histogram / bincount / unique
    r"|torch\.(histc|bincount|nonzero|repeat_interleave)"
    r"|\.histc\(|\.bincount\(|\.nonzero\("
    # nn.functional paths
    r"|F\.(embedding_bag|ctc_loss|interpolate"
    r"|upsample(?:_nearest|_bilinear)?|grid_sample|affine_grid"
    r"|max_pool[23]d_with_indices|adaptive_avg_pool[23]d"
    r"|adaptive_max_pool[23]d|avg_pool3d"
    r"|fractional_max_pool[23]d|nll_loss2d"
    r"|replication_pad[23]d|reflection_pad2d)"
    # nn.* classes
    r"|nn\.EmbeddingBag|nn\.CTCLoss|nn\.Upsample"
    r"|nn\.FractionalMaxPool"
    r"|nn\.AdaptiveMaxPool[23]d|nn\.AdaptiveAvgPool[23]d"
    r"|nn\.MaxPool[23]d|nn\.AvgPool3d"
    r"|nn\.ReflectionPad2d|nn\.ReplicationPad[23]d"
    # AMP / sparse ops
    r"|torch\.put"
    r")"
)


def _extract_model_body(code: str) -> str:
    """
    Return the body of `class Model(...)` and any method it defines.

    `make_inputs()` is allowed to call random ops (the runner persists its
    output and reuses it identically on CPU and GPU), so we only check
    randomness inside the Model class.
    """
    m = _re.search(r"class\s+Model\b[^:]*:", code)
    if not m:
        return code  # No Model class found → check whole file defensively
    start = m.end()
    # Take lines until we hit a line that is left-aligned (end of class).
    tail = code[start:].splitlines()
    body_lines = [tail[0]] if tail else []
    for line in tail[1:]:
        if line and not line.startswith((" ", "\t")) and not line.startswith("#"):
            break
        body_lines.append(line)
    return "\n".join(body_lines)


def has_randomness(code: str) -> bool:
    """True if the Model class body contains randomness that makes CPU/GPU diverge.

    Random calls in make_inputs() are fine — the runner saves make_inputs()
    output once and reuses the exact same tensor for each device.
    """
    return bool(_RANDOMNESS_RX.search(_extract_model_body(code)))


def has_nondet_gpu_op(code: str) -> bool:
    """True if the code uses an op that is non-deterministic on GPU."""
    return bool(_NONDET_GPU_RX.search(code))


# ------------------------------------------------------------------ #
# Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class RunResult:
    device: str
    status: str                        # "ok" | "error" | "crash" | "timeout"
    outputs: List[torch.Tensor] = field(default_factory=list)
    outputs_repeat: Optional[List[torch.Tensor]] = None  # 2nd same-device run
    error: str = ""


@dataclass
class ExecutionPair:
    cpu: RunResult
    gpu: RunResult
    model_id: int
    mutation_id: int = 0              # 0 = baseline; 1-5 = mutation type
    inputs: Optional[List[torch.Tensor]] = None
    has_nondet_ops: bool = False       # Model uses ops known to be nondet on GPU


# ------------------------------------------------------------------ #
# Mutation helpers                                                    #
# ------------------------------------------------------------------ #

_MUTATION_NAMES = {
    1: "add_noise",
    2: "scale_small",
    3: "mask",
    4: "uniform",
    5: "scale_large",
}


def apply_mutation(inputs: List[torch.Tensor], strategy: int) -> List[torch.Tensor]:
    """Return mutated copies of input tensors (strategy 1–5)."""
    mutated = []
    for t in inputs:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            mutated.append(t)
            continue
        t = t.clone().float()
        if strategy == 1:    # add Gaussian noise
            t = t + torch.randn_like(t) * 0.1
        elif strategy == 2:  # multiply by small factor
            f = torch.empty(1).uniform_(0.01, 0.1).item()
            t = t * f
        elif strategy == 3:  # random masking (30%)
            mask = torch.bernoulli(torch.full_like(t, 0.3)).bool()
            t = t.masked_fill(mask, 0.0)
        elif strategy == 4:  # uniform constant
            val = torch.randn(1).item()
            t = torch.full_like(t, val)
        elif strategy == 5:  # large scaling
            f = torch.empty(1).uniform_(1e3, 1e6).item()
            t = t * f
        mutated.append(t)
    return mutated


# ------------------------------------------------------------------ #
# Executor                                                            #
# ------------------------------------------------------------------ #

class ModelExecutor:
    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ---- public ---------------------------------------------------- #

    def run_baseline(
        self, code: str, model_id: int
    ) -> Tuple[ExecutionPair, Optional[List[torch.Tensor]]]:
        """Run model with make_inputs() on CPU and GPU. Return pair + inputs.

        GPU is executed twice so the oracle can detect GPU non-determinism
        and not report it as a CPU/GPU library bug.
        """
        script = self._write_script(code, model_id)
        cpu_out_file = self._tmp_path(f"m{model_id}_cpu_base.pt")
        gpu_out_file = self._tmp_path(f"m{model_id}_gpu_base.pt")

        cpu_result = self._run_subprocess(script, "cpu", None, cpu_out_file)
        inputs = self._load_inputs(cpu_out_file)
        # Always double-run GPU to expose non-determinism.
        gpu_result = self._run_subprocess(
            script, "cuda", inputs, gpu_out_file, double_run=True,
        )

        pair = ExecutionPair(
            cpu=cpu_result, gpu=gpu_result,
            model_id=model_id, mutation_id=0,
            inputs=inputs,
            has_nondet_ops=has_nondet_gpu_op(code),
        )
        return pair, inputs

    def run_mutation(
        self,
        code: str,
        model_id: int,
        inputs: List[torch.Tensor],
        strategy: int,
    ) -> ExecutionPair:
        """Run model with mutated inputs on CPU and GPU (GPU twice)."""
        mutated = apply_mutation(inputs, strategy)
        script = self._write_script(code, model_id)
        cpu_out_file = self._tmp_path(f"m{model_id}_cpu_mut{strategy}.pt")
        gpu_out_file = self._tmp_path(f"m{model_id}_gpu_mut{strategy}.pt")
        inp_file = self._tmp_path(f"m{model_id}_inputs_mut{strategy}.pt")
        torch.save(mutated, inp_file)

        cpu_result = self._run_subprocess(
            script, "cpu", mutated, cpu_out_file, inp_file,
        )
        gpu_result = self._run_subprocess(
            script, "cuda", mutated, gpu_out_file, inp_file, double_run=True,
        )

        return ExecutionPair(
            cpu=cpu_result, gpu=gpu_result,
            model_id=model_id, mutation_id=strategy,
            inputs=mutated,
            has_nondet_ops=has_nondet_gpu_op(code),
        )

    # ---- internal -------------------------------------------------- #

    def _write_script(self, code: str, model_id: int) -> Path:
        path = self._output_dir / f"model_{model_id:04d}.py"
        path.write_text(_strip_main_block(code) + "\n" + _RUNNER_SUFFIX)
        return path

    def _tmp_path(self, name: str) -> Path:
        return self._output_dir / name

    def _run_subprocess(
        self,
        script: Path,
        device: str,
        inputs: Optional[List[torch.Tensor]],
        out_file: Path,
        inp_file: Optional[Path] = None,
        double_run: bool = False,
    ) -> RunResult:
        cmd = [sys.executable, str(script),
               "--device", device,
               "--output-file", str(out_file)]
        if inp_file and inp_file.exists():
            cmd += ["--input-file", str(inp_file)]
        if double_run:
            cmd += ["--double-run"]

        timeout = DEVICE_TIMEOUT

        try:
            proc = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Timeout on device=%s", device)
            return RunResult(device=device, status="timeout",
                             error=f"Timed out after {timeout}s")
        except Exception as exc:
            return RunResult(device=device, status="crash",
                             error=str(exc))

        # Non-zero exit without output file → crash (SIGSEGV, etc.)
        if proc.returncode != 0 and not out_file.exists():
            err = (proc.stderr or proc.stdout or "")[:500]
            return RunResult(device=device, status="crash", error=err)

        # Load saved result
        if not out_file.exists():
            return RunResult(device=device, status="crash",
                             error="Output file not created")

        try:
            result = torch.load(out_file, weights_only=False)
        except Exception as exc:
            return RunResult(device=device, status="crash",
                             error=f"Could not load output: {exc}")

        if result["status"] == "error":
            return RunResult(device=device, status="error",
                             outputs=[], error=result.get("traceback", result.get("error", "")))

        return RunResult(
            device=device, status="ok",
            outputs=result.get("outputs", []),
            outputs_repeat=result.get("outputs_repeat"),
        )

    def _load_inputs(self, cpu_out_file: Path) -> Optional[List[torch.Tensor]]:
        inp_path = Path(str(cpu_out_file) + ".inputs.pt")
        if inp_path.exists():
            try:
                return torch.load(inp_path, weights_only=False)
            except Exception:
                pass
        return None
