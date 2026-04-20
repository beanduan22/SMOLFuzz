"""
SMOLFuzz — Anthropic Claude backend example.

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Run from the parent directory of this repo:
    python3 -m smolfuzz.examples.anthropic_example
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smolfuzz.backends.llm_client import AnthropicClient
from smolfuzz.core.api_loader import load_and_classify
from smolfuzz.core.executor import ModelExecutor
from smolfuzz.core.oracle import DifferentialOracle
from smolfuzz.core.selector import MultiRouletteSelector
from smolfuzz.core.synthesizer import ModelSynthesizer

# ── Configuration ──────────────────────────────────────────────────────────────

API_FILE    = Path(__file__).resolve().parents[1] / "torch_valid_apis.txt"
OUTPUT_DIR  = Path("results/anthropic_run")
N_MODELS    = 10
API_SET     = 30
BUDGET      = 60

# ── Backend ────────────────────────────────────────────────────────────────────

client = AnthropicClient(
    model="claude-sonnet-4-6",
    temperature=0.7,
    max_tokens=4096,
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
