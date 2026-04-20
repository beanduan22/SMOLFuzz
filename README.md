# SMOLFuzz

**S**ynthesizing **M**odels with **O**pen **L**LMs for **Fuzz**ing Deep Learning Libraries.

SMOLFuzz is a differential fuzzer that uses an LLM to synthesize PyTorch and TensorFlow models, then compares CPU vs GPU numerical outputs to detect divergence bugs in deep learning library kernels.

---

## Setup

See [docs/setup.md](docs/setup.md) for full requirements and GPU verification steps.

```bash
pip install -r requirements.txt

# Pull the default local model (~20 GB VRAM)
ollama pull qwen2.5-coder:32b

# Validate the installation (5 models)
python3 -m smolfuzz.main --mode subset
```

---

## Reproduce Experiments

See [docs/usage.md](docs/usage.md) for all CLI flags and output layout.

```bash
# PyTorch — 300 models, 60-second mutation budget each
python3 -m smolfuzz.main --mode full --models 300 --budget 60

# TensorFlow
python3 -m smolfuzz.run_tf --models 300 --budget 60

# Both frameworks in parallel
python3 run_both.py --models 300 --budget 60
```

---

## Replace the LLM Backend

SMOLFuzz decouples model synthesis from the LLM provider. Switching backends requires only changing which client object is passed to `ModelSynthesizer` in `main.py`.

### Ollama (default — local models)

No API key required. Requires a running Ollama server.

```bash
ollama serve
ollama pull qwen2.5-coder:32b
```

```python
from smolfuzz.backends.llm_client import OllamaClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = OllamaClient(models=["qwen2.5-coder:32b"])
synthesizer = ModelSynthesizer(client)
```

Via CLI (round-robin across multiple models):

```bash
python3 -m smolfuzz.main --mode full --llm-models "qwen2.5-coder:32b,llama3.3:70b"
```

### OpenAI

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

```python
from smolfuzz.backends.llm_client import OpenAIClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = OpenAIClient(model="gpt-4o")
synthesizer = ModelSynthesizer(client)
```

Also works with any OpenAI-compatible endpoint (Together AI, Fireworks, vLLM) by passing `base_url` to `openai.OpenAI(base_url=...)`.

### Anthropic

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from smolfuzz.backends.llm_client import AnthropicClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = AnthropicClient(model="claude-sonnet-4-6")
synthesizer = ModelSynthesizer(client)
```

### Custom Backend

Implement the `LLMBackend` protocol — three members are required:

```python
from smolfuzz.backends.llm_client import LLMBackend

class MyClient:
    @property
    def current_model(self) -> str:
        return "my-model"

    def generate(self, prompt: str, advance: bool = True) -> str:
        # Call your LLM and return the response string.
        ...

    def stats(self) -> dict:
        return {}
```

Wire it in by replacing the client in `main.py`:

```python
# Replace:
client = OllamaClient(models=llm_models) if llm_models else OllamaClient()

# With:
client = MyClient()
```

---

## Project Structure

```
smolfuzz/
├── main.py              # PyTorch fuzzing entry point
├── run_tf.py            # TensorFlow fuzzing entry point
├── run_both.py          # Run PT + TF in parallel
├── torch_valid_apis.txt
├── tf_valid_apis.txt
├── core/
│   ├── api_loader.py    # API loader + 11-group classifier
│   ├── selector.py      # Multi-roulette API selector
│   ├── synthesizer.py   # LLM model synthesis + self-repair loop
│   ├── executor.py      # Subprocess executor + 5 mutation strategies
│   ├── oracle.py        # Differential oracle (CPU vs GPU)
│   └── prompts.py       # LLM prompt templates
├── backends/
│   └── llm_client.py    # LLM backends (Ollama / OpenAI / Anthropic)
└── docs/
    ├── setup.md
    ├── usage.md
    └── api_classification.md
```
