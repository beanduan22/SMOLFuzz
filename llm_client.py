"""
LLM backend abstraction for SMOLFuzz.

To swap backends, implement the LLMBackend protocol and pass your instance
to ModelSynthesizer:

    from smolfuzz.llm_client import LLMBackend, OllamaClient, OpenAIClient

    client = OllamaClient()                       # local Ollama (default)
    client = OpenAIClient(model="gpt-4o")         # OpenAI
    client = AnthropicClient(model="claude-opus-4-6")  # Anthropic
    synthesizer = ModelSynthesizer(client)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Protocol, runtime_checkable

import requests

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Backend Protocol — implement this to add a new LLM backend         #
# ------------------------------------------------------------------ #

@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    @property
    def current_model(self) -> str:
        """Human-readable name of the model currently in use."""
        ...

    def generate(self, prompt: str, advance: bool = True) -> str:
        """
        Send *prompt* to the LLM and return the response text.

        ``advance`` hints that the caller is starting a new independent
        generation; backends that round-robin over multiple models may
        advance their pointer when this is True and hold it during
        self-repair loops when it is False.

        Raises RuntimeError on unrecoverable communication errors.
        """
        ...

    def stats(self) -> dict:
        """Return a JSON-serialisable dict with usage statistics."""
        ...


# ------------------------------------------------------------------ #
# Ollama backend (local, open-source models)                         #
# ------------------------------------------------------------------ #

OLLAMA_URL = "http://localhost:11434"

# Default model list — override via OllamaClient(models=[...]).
# We default to a single model to avoid Ollama reloading weights on every
# round-robin call when the GPU has no spare VRAM. Multi-model runs can
# be enabled explicitly via the --llm-models CLI flag once memory allows.
_DEFAULT_OLLAMA_MODELS = [
    "qwen2.5-coder:32b",  # primary code-specialist (~20 GB VRAM)
]


class OllamaClient:
    """
    HTTP client for a local Ollama server.  Round-robins across *models*.

    Compatible with any model served by ``ollama run <model>``.
    """

    def __init__(
        self,
        models: list[str] = _DEFAULT_OLLAMA_MODELS,
        base_url: str = OLLAMA_URL,
        timeout: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
    ) -> None:
        self._models = models
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self._idx = 0
        self._call_counts = {m: 0 for m in models}

    @property
    def current_model(self) -> str:
        return self._models[self._idx % len(self._models)]

    def generate(self, prompt: str, advance: bool = True) -> str:
        model = self.current_model
        if advance:
            self._idx += 1

        logger.info("LLM call → %s", model)
        self._call_counts[model] += 1

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "top_p": self._top_p,
                "num_predict": self._max_tokens,
            },
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama timeout after {self._timeout}s for model {model}"
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        elapsed = time.time() - start
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned non-JSON: {resp.text[:200]}"
            ) from exc

        response_text = data.get("response", "")
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(response_text))
        return response_text

    def stats(self) -> dict:
        return {"call_counts": dict(self._call_counts), "next_model": self.current_model}


# ------------------------------------------------------------------ #
# OpenAI backend                                                      #
# ------------------------------------------------------------------ #

class OpenAIClient:
    """
    OpenAI-compatible backend (OpenAI, Together AI, Fireworks, etc.).

    Install: pip install openai
    Set:     OPENAI_API_KEY=sk-...
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai") from None
        self._client = openai.OpenAI()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_count = 0

    @property
    def current_model(self) -> str:
        return self._model

    def generate(self, prompt: str, advance: bool = True) -> str:
        self._call_count += 1
        logger.info("LLM call → %s", self._model)
        start = time.time()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        elapsed = time.time() - start
        text = resp.choices[0].message.content or ""
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(text))
        return text

    def stats(self) -> dict:
        return {"call_counts": {self._model: self._call_count}, "next_model": self._model}


# ------------------------------------------------------------------ #
# Anthropic backend                                                   #
# ------------------------------------------------------------------ #

class AnthropicClient:
    """
    Anthropic Claude backend.

    Install: pip install anthropic
    Set:     ANTHROPIC_API_KEY=sk-ant-...
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic") from None
        self._client = anthropic.Anthropic()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._call_count = 0

    @property
    def current_model(self) -> str:
        return self._model

    def generate(self, prompt: str, advance: bool = True) -> str:
        self._call_count += 1
        logger.info("LLM call → %s", self._model)
        start = time.time()
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.time() - start
        text = msg.content[0].text if msg.content else ""
        logger.info("LLM response in %.1fs (%d chars)", elapsed, len(text))
        return text

    def stats(self) -> dict:
        return {"call_counts": {self._model: self._call_count}, "next_model": self._model}
