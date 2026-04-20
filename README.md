# SMOLFuzz

**S**ynthesizing **M**odels with **O**pen **L**LMs for **Fuzz**ing Deep Learning Libraries.

SMOLFuzz is a differential fuzzer that uses an LLM to synthesize PyTorch and TensorFlow models, then compares CPU vs GPU numerical outputs to detect divergence bugs in deep learning library kernels.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/setup.md](docs/setup.md) | Environment setup and dependencies |
| [docs/usage.md](docs/usage.md) | How to reproduce experiments |
| [docs/llm_backends.md](docs/llm_backends.md) | How to swap the LLM backend |
| [docs/api_classification.md](docs/api_classification.md) | 11-group API taxonomy |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Pull a local model (Ollama default)
ollama pull qwen2.5-coder:32b

# 3. Validate the setup (5 models)
python3 -m smolfuzz.main --mode subset

# 4. Full PyTorch run
python3 -m smolfuzz.main --mode full --models 300 --budget 60
```

See [docs/usage.md](docs/usage.md) for TensorFlow, parallel runs, and all CLI options.  
See [docs/llm_backends.md](docs/llm_backends.md) to use OpenAI, Anthropic, or any custom backend.

---

## Project Structure

```
smolfuzz/
├── main.py            # PyTorch fuzzing entry point
├── run_tf.py          # TensorFlow fuzzing entry point
├── run_both.py        # Run PT + TF in parallel
├── api_loader.py      # API loader + 11-group classifier
├── selector.py        # Multi-roulette API selector
├── synthesizer.py     # LLM model synthesis + self-repair loop
├── executor.py        # Subprocess executor + 5 mutation strategies
├── oracle.py          # Differential oracle (CPU vs GPU)
├── llm_client.py      # LLM backend (Ollama / OpenAI / Anthropic)
├── prompts.py         # LLM prompt templates
├── torch_valid_apis.txt
├── tf_valid_apis.txt
└── docs/
    ├── setup.md
    ├── usage.md
    ├── llm_backends.md
    └── api_classification.md
```
