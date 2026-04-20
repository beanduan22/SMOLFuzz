"""
SMOLFuzz — Custom backend example.

Implement the LLMBackend protocol to plug in any LLM provider:
HuggingFace Transformers, vLLM, LM Studio, LiteLLM, etc.

Three members are required:
    current_model  (property) → str
    generate(prompt, advance) → str
    stats()                   → dict

Run from the parent directory of this repo:
    python3 -m smolfuzz.examples.custom_backend_example
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smolfuzz.backends.llm_client import LLMBackend
from smolfuzz.core.api_loader import load_and_classify
from smolfuzz.core.executor import ModelExecutor
from smolfuzz.core.oracle import DifferentialOracle
from smolfuzz.core.selector import MultiRouletteSelector
from smolfuzz.core.synthesizer import ModelSynthesizer

# ── Implement the LLMBackend protocol ──────────────────────────────────────────

class MyCustomClient:
    """
    Minimal example using a HuggingFace text-generation pipeline.
    Replace the body of generate() with any inference call.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct") -> None:
        # Example: load a local HuggingFace model.
        # Uncomment and install: pip install transformers accelerate
        #
        # from transformers import pipeline
        # self._pipe = pipeline(
        #     "text-generation", model=model_name,
        #     device_map="auto", torch_dtype="auto",
        # )
        self._model_name = model_name
        self._call_count = 0

    @property
    def current_model(self) -> str:
        return self._model_name

    def generate(self, prompt: str, advance: bool = True) -> str:
        self._call_count += 1
        # Replace the stub below with your actual inference call.
        # Example for a HuggingFace pipeline:
        #
        # outputs = self._pipe(prompt, max_new_tokens=4096, temperature=0.7)
        # return outputs[0]["generated_text"][len(prompt):]
        #
        raise NotImplementedError(
            "Replace this stub with your inference call in generate()."
        )

    def stats(self) -> dict:
        return {"call_counts": {self._model_name: self._call_count},
                "next_model": self._model_name}


# ── Configuration ──────────────────────────────────────────────────────────────

API_FILE    = Path(__file__).resolve().parents[1] / "torch_valid_apis.txt"
OUTPUT_DIR  = Path("results/custom_run")
N_MODELS    = 10
API_SET     = 30
BUDGET      = 60

# ── Backend ────────────────────────────────────────────────────────────────────

client = MyCustomClient(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

# Verify that MyCustomClient satisfies the LLMBackend protocol.
assert isinstance(client, LLMBackend), (
    "MyCustomClient does not implement the LLMBackend protocol. "
    "Check that current_model, generate(), and stats() are all defined."
)

# ── Pipeline ───────────────────────────────────────────────────────────────────

groups   = load_and_classify(str(API_FILE), framework="torch")
selector = MultiRouletteSelector(groups)
synth    = ModelSynthesizer(client)
executor = ModelExecutor(output_dir=OUTPUT_DIR)
oracle   = DifferentialOracle(output_dir=OUTPUT_DIR / "bugs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "bugs").mkdir(exist_ok=True)

bugs_found = 0

for i in range(N_MODELS):
    api_set = selector.select(n=API_SET)
    result  = synth.synthesize(api_set, model_id=i)

    if result.code is None:
        print(f"[{i+1}/{N_MODELS}] synthesis failed — skipping")
        continue

    pair   = executor.run_baseline(result.code, model_id=i)
    report = oracle.judge(pair)

    if report.is_bug:
        bugs_found += 1
        print(f"[{i+1}/{N_MODELS}] BUG: {report.bug_type}  (total={bugs_found})")
        selector.record_usage(result.used_apis, found_bug=True)
        executor.run_mutations(result.code, model_id=i, budget=BUDGET,
                               oracle=oracle, selector=selector,
                               used_apis=result.used_apis)
    else:
        print(f"[{i+1}/{N_MODELS}] clean — {report.bug_type}")
        selector.record_usage(result.used_apis, found_bug=False)

print(f"\nDone. {bugs_found} bug(s) found. Reports in {OUTPUT_DIR / 'bugs'}")
print(f"LLM stats: {client.stats()}")
