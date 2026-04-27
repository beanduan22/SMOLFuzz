from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..backends.llm_client import LLMBackend
from .skeletons import Skeleton
from .prompts import build_repair_prompt, build_synthesis_prompt

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 4
EXEC_TIMEOUT_SECONDS = 20


@dataclass
class SynthesisResult:
    code: str
    used_apis: List[str]
    model_id: int
    llm_model: str
    skeleton_id: str
    attempts: int = 1
    repaired: bool = False
    error: Optional[str] = None


def _strip_markdown(text: str) -> str:
    text = text.strip()
    block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if block:
        return block.group(1).strip()
    text = re.sub(r"^```(?:python)?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_attr_paths(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    used: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node: ast.Attribute) -> None:
            parts: list[str] = []
            cur: ast.AST | None = node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                used.add(".".join(reversed(parts)))
            self.generic_visit(node)

    Visitor().visit(tree)
    return used


def _extract_used_apis(code: str, candidate_pool: List[str]) -> List[str]:
    attrs = _extract_attr_paths(code)
    candidate_set = set(candidate_pool)
    matched: set[str] = {api for api in candidate_set if api in attrs}
    attr_suffixes = {
        path.rsplit(".", 1)[-1]
        for path in attrs
        if "." in path
    }
    for api in candidate_pool:
        if api in matched:
            continue
        if api.startswith("torch.Tensor.") or api.startswith("tf.Tensor."):
            method_name = api.rsplit(".", 1)[-1]
            if method_name in attr_suffixes:
                matched.add(api)
    return sorted(matched)


def _quick_exec_check(code: str) -> Optional[str]:
    try:
        compile(code, "<generated>", "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"


def _runtime_check(code: str) -> Optional[str]:
    runner = """
import sys
import traceback
import torch

ns = {}
code = open(sys.argv[1]).read()
try:
    exec(compile(code, "<generated>", "exec"), ns)
    if "Model" not in ns:
        raise RuntimeError("Generated code does not define a 'Model' class.")
    if "make_inputs" not in ns:
        raise RuntimeError("Generated code does not define 'make_inputs()'.")
    inputs = ns["make_inputs"]()
    if not isinstance(inputs, (list, tuple)):
        raise RuntimeError(f"make_inputs() must return a list or tuple, got {type(inputs).__name__}")
    model = ns["Model"]()
    cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
    _ = model(*cpu_inputs)
    print("OK")
except Exception:
    print(traceback.format_exc(limit=4))
"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "generated_model.py"
            runner_path = Path(tmpdir) / "validate_runner.py"
            code_path.write_text(code)
            runner_path.write_text(runner)
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(code_path)],
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        if proc.returncode == 0 and proc.stdout.strip() == "OK":
            return None
        return _summarize_error((proc.stdout or proc.stderr or "unknown runtime validation failure").strip())
    except subprocess.TimeoutExpired:
        return f"Runtime validation timed out after {EXEC_TIMEOUT_SECONDS} seconds"
    except Exception:
        return _summarize_error(traceback.format_exc(limit=3))


_FRAMEWORK_FRAME_RE = re.compile(
    r"^\s*File \".*(site-packages|torch|tensorflow|tf2|keras)[\\/].*\".*$",
    re.IGNORECASE,
)


def _summarize_error(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return raw[:600]
    last = lines[-1]
    head = ":".join(last.split(":", 2)[:2])
    msg = last
    user_frames: list[str] = []
    for ln in lines:
        if ln.lstrip().startswith("File ") and not _FRAMEWORK_FRAME_RE.match(ln):
            user_frames.append(ln.strip())
    user_frames = user_frames[-3:]
    framework_kept = lines[-3:] if not user_frames else []
    pieces = [head, msg] + user_frames + framework_kept
    seen: set[str] = set()
    unique = []
    for p in pieces:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return "\n".join(unique)[:1500]


class ModelSynthesizer:
    def __init__(self, client: LLMBackend, target_lib: str = "PyTorch") -> None:
        self._client = client
        self._model_counter = 0
        self._target_lib = target_lib

    def synthesize(self, skeleton: Skeleton, api_list: List[str]) -> SynthesisResult:
        self._model_counter += 1
        model_id = self._model_counter
        llm_model = self._client.current_model

        prompt = build_synthesis_prompt(api_list, skeleton.template, target_lib=self._target_lib)
        try:
            raw = self._client.generate(prompt, advance=True)
        except RuntimeError as exc:
            return SynthesisResult(
                code="", used_apis=[], model_id=model_id,
                llm_model=llm_model, skeleton_id=skeleton.skeleton_id, error=str(exc),
            )

        code = _strip_markdown(raw)
        logger.debug("Synthesized code (%d chars)", len(code))

        attempts = 1
        repaired = False
        last_error: Optional[str] = None

        for attempt in range(MAX_REPAIR_ATTEMPTS):
            err = _quick_exec_check(code) or _runtime_check(code)
            if err is None:
                break
            last_error = err
            logger.warning("Model %d attempt %d failed: %s", model_id, attempt + 1, err[:200])
            try:
                raw = self._client.generate(build_repair_prompt(code, err), advance=False)
            except RuntimeError as exc:
                last_error = str(exc)
                break
            code = _strip_markdown(raw)
            attempts += 1
            repaired = True

        used_apis = _extract_used_apis(code, api_list)
        final_err = _quick_exec_check(code) or _runtime_check(code)
        logger.info(
            "Model %d synthesised | skeleton=%s | attempts=%d | used_apis=%d | repaired=%s",
            model_id, skeleton.skeleton_id, attempts, len(used_apis), repaired,
        )

        return SynthesisResult(
            code=code,
            used_apis=used_apis,
            model_id=model_id,
            llm_model=llm_model,
            skeleton_id=skeleton.skeleton_id,
            attempts=attempts,
            repaired=repaired,
            error=last_error if final_err else None,
        )
