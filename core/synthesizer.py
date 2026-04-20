"""
Model synthesis: call the LLM, extract USED_APIS, and run a self-repair
loop when the generated code is not executable.
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

from ..backends.llm_client import LLMBackend
from .prompts import build_repair_prompt, build_synthesis_prompt

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 4


@dataclass
class SynthesisResult:
    code: str
    used_apis: List[str]
    model_id: int
    llm_model: str
    attempts: int = 1           # synthesis + repair rounds
    repaired: bool = False
    error: Optional[str] = None  # set if all repair attempts failed


def _strip_markdown(text: str) -> str:
    """Remove ```python … ``` fences that some models include."""
    text = text.strip()
    # If wrapped in a single code block, extract its content
    block = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if block:
        return block.group(1).strip()
    # Otherwise strip any leading/trailing fence lines
    text = re.sub(r"^```(?:python)?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_used_apis(code: str) -> List[str]:
    """Parse USED_APIS = [...] from generated code using ast."""
    pattern = re.search(r"USED_APIS\s*=\s*(\[.*?\])", code, re.DOTALL)
    if not pattern:
        return []
    try:
        return ast.literal_eval(pattern.group(1))
    except Exception:
        return []


# Random operations that must never appear inside the Model class.
# make_inputs() is exempt — its outputs are saved once and reloaded identically
# on every device, so any RNG there does not affect the differential comparison.
_RANDOM_OPS_RE = re.compile(
    r'\b('
    r'torch\.rand(?:n|int|perm|_like|n_like|int_like)?'   # rand family
    r'|torch\.bernoulli'
    r'|torch\.multinomial'
    r'|torch\.poisson'
    r'|torch\.normal'
    r'|torch\.dropout'
    r'|nn\.Dropout'
    r'|F\.dropout'
    r'|functional\.dropout'
    r')\b'
)


def _check_determinism(code: str) -> Optional[str]:
    """
    Verify that the Model class contains no stochastic (random) operations.
    Returns None if the model is deterministic, or an error string otherwise.

    Only the Model class body is scanned — make_inputs() is intentionally
    excluded because its outputs are saved and reloaded identically on every
    device, so any RNG there never affects the differential comparison.
    """
    # Extract the Model class definition (everything from 'class Model' onward)
    match = re.search(r"^class Model\b", code, re.MULTILINE)
    if not match:
        return None  # no Model class yet — other checks will catch this

    model_body = code[match.start():]
    # Trim to just the class block: stop at the next top-level definition
    # (another class or def at column 0) so we don't accidentally scan
    # make_inputs() even if it appears right after Model.
    next_top = re.search(r"^(?:class|def)\s+(?!Model\b)", model_body, re.MULTILINE)
    if next_top:
        model_body = model_body[: next_top.start()]

    hit = _RANDOM_OPS_RE.search(model_body)
    if hit:
        return (
            f"Determinism violation: '{hit.group()}' found inside the Model class. "
            "Model.__init__ and Model.forward() must be fully deterministic — "
            "remove all random/stochastic operations (torch.rand*, torch.bernoulli, "
            "nn.Dropout, F.dropout, etc.) from the Model class. "
            "Random tensors belong only in make_inputs()."
        )
    return None


def _quick_exec_check(code: str) -> Optional[str]:
    """
    Try to compile the code (syntax check only).
    Returns None on success, error string on failure.
    """
    try:
        compile(code, "<generated>", "exec")
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"


def _runtime_check(code: str) -> Optional[str]:
    """
    Try executing the code and running a forward pass on CPU.
    Returns None on success, error string on failure.
    """
    import torch
    ns: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102
        if "Model" not in ns:
            return "Generated code does not define a 'Model' class."
        if "make_inputs" not in ns:
            return "Generated code does not define 'make_inputs()'."

        inputs = ns["make_inputs"]()
        if not isinstance(inputs, (list, tuple)):
            return f"make_inputs() must return a list, got {type(inputs).__name__}"

        # Actually run a forward pass on CPU to catch shape/dim mismatches
        model = ns["Model"]()
        model.cpu()
        cpu_inputs = [x.cpu() if isinstance(x, torch.Tensor) else x for x in inputs]
        _ = model(*cpu_inputs)
        return None
    except Exception:
        return traceback.format_exc(limit=3)


MAX_RUNTIME_REPAIR_ATTEMPTS = 2   # extra repair rounds after execution failure


class ModelSynthesizer:
    def __init__(self, client: LLMBackend) -> None:
        self._client = client
        self._model_counter = 0

    def synthesize(self, api_list: List[str]) -> SynthesisResult:
        """
        Synthesize one executable model from the given API list.
        Runs up to MAX_REPAIR_ATTEMPTS repair rounds on failure.
        """
        self._model_counter += 1
        model_id = self._model_counter
        llm_model = self._client.current_model

        # --- Initial synthesis ---
        prompt = build_synthesis_prompt(api_list)
        try:
            raw = self._client.generate(prompt, advance=True)
        except RuntimeError as exc:
            return SynthesisResult(
                code="", used_apis=[], model_id=model_id,
                llm_model=llm_model, error=str(exc),
            )

        code = _strip_markdown(raw)
        logger.debug("Synthesized code (%d chars)", len(code))

        # --- Validate; repair if needed ---
        attempts = 1
        repaired = False
        last_error: Optional[str] = None

        for attempt in range(MAX_REPAIR_ATTEMPTS):
            err = _check_determinism(code) or _quick_exec_check(code) or _runtime_check(code)
            if err is None:
                break                      # success

            last_error = err
            logger.warning("Model %d attempt %d failed: %s", model_id, attempt + 1,
                           err[:200])

            repair_prompt = build_repair_prompt(code, err)
            try:
                raw = self._client.generate(repair_prompt, advance=False)
            except RuntimeError as exc:
                last_error = str(exc)
                break

            code = _strip_markdown(raw)
            attempts += 1
            repaired = True
        else:
            # All repair attempts exhausted — final check
            err = _check_determinism(code) or _quick_exec_check(code) or _runtime_check(code)
            if err:
                last_error = err

        used_apis = _extract_used_apis(code)
        logger.info(
            "Model %d synthesised | attempts=%d | used_apis=%d | repaired=%s",
            model_id, attempts, len(used_apis), repaired,
        )

        return SynthesisResult(
            code=code,
            used_apis=used_apis,
            model_id=model_id,
            llm_model=llm_model,
            attempts=attempts,
            repaired=repaired,
            error=last_error if (
            _check_determinism(code) or _quick_exec_check(code) or _runtime_check(code)
        ) else None,
        )

    def repair_from_execution_error(
        self, result: SynthesisResult, execution_error: str
    ) -> SynthesisResult:
        """
        Attempt to repair a model that passed synthesis checks but failed
        during subprocess execution (e.g. a CUDA-specific error, or an
        error that only appears with the runner harness).

        Returns a new SynthesisResult.  If repair still fails the in-process
        check, the returned result will have .error set.
        """
        logger.info(
            "Runtime repair for model %d | error: %s",
            result.model_id, execution_error[:120],
        )
        code = result.code
        for attempt in range(MAX_RUNTIME_REPAIR_ATTEMPTS):
            repair_prompt = build_repair_prompt(code, execution_error)
            try:
                raw = self._client.generate(repair_prompt, advance=False)
            except RuntimeError as exc:
                logger.warning("Runtime repair LLM call failed: %s", exc)
                break

            code = _strip_markdown(raw)
            err = _check_determinism(code) or _quick_exec_check(code) or _runtime_check(code)
            if err is None:
                logger.info(
                    "Runtime repair succeeded on attempt %d for model %d",
                    attempt + 1, result.model_id,
                )
                used_apis = _extract_used_apis(code)
                return SynthesisResult(
                    code=code,
                    used_apis=used_apis,
                    model_id=result.model_id,
                    llm_model=result.llm_model,
                    attempts=result.attempts + attempt + 1,
                    repaired=True,
                    error=None,
                )
            logger.warning(
                "Runtime repair attempt %d still fails: %s",
                attempt + 1, err[:150],
            )

        # All repair attempts exhausted — return original with error flagged
        return SynthesisResult(
            code=result.code,
            used_apis=result.used_apis,
            model_id=result.model_id,
            llm_model=result.llm_model,
            attempts=result.attempts + MAX_RUNTIME_REPAIR_ATTEMPTS,
            repaired=True,
            error=f"Runtime repair exhausted: {execution_error[:200]}",
        )
