# SMOLFuzz

**S**ynthesizing **M**odels with **O**pen **L**LMs for **Fuzz**ing Deep Learning Libraries.

SMOLFuzz is a differential fuzzer that uses an LLM to synthesize PyTorch and TensorFlow models, then compares CPU vs GPU numerical outputs to detect divergence bugs in deep learning library kernels.

---

## Setup

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.9.1 with CUDA, TensorFlow ≥ 2.21 with GPU, NVIDIA GPU with CUDA ≥ 11.8.

```bash
pip install -r requirements.txt
```

Install Ollama and pull both required models:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5-coder:32b
ollama pull deepseek-v2
ollama serve
```

Verify GPU access:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Validate the installation:

```bash
python3 -m smolfuzz.main --mode subset
```

---

## Reproduce Experiments

The fuzzer runs until the paper's stopping criterion triggers: 10 consecutive models that introduce no previously unseen API — no model count needs to be specified.

**PyTorch:**

```bash
python3 -m smolfuzz.main --mode full --budget 60
```

**TensorFlow:**

```bash
python3 -m smolfuzz.run_tf --budget 60
```

**Both frameworks in parallel:**

```bash
python3 run_both.py --budget 60
```

Results are written to `results/`, with bug reports under `results/bugs/`.

---

## Replace the LLM Backend

SMOLFuzz decouples model synthesis from the LLM provider. Switching backends requires only changing which client object is passed to `ModelSynthesizer` in `main.py`.

### Ollama (default — local models)

Qwen and DeepSeek are both required; the fuzzer round-robins between them.

```bash
ollama serve
ollama pull qwen2.5-coder:32b
ollama pull deepseek-v2
```

```python
from smolfuzz.backends.llm_client import OllamaClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = OllamaClient(models=["qwen2.5-coder:32b", "deepseek-v2"])
synthesizer = ModelSynthesizer(client)
```

Via CLI:

```bash
python3 -m smolfuzz.main --mode full --llm-models "qwen2.5-coder:32b,deepseek-v2"
```

### OpenAI (GPT-4 Turbo)

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

```python
from smolfuzz.backends.llm_client import OpenAIClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = OpenAIClient(model="gpt-4-turbo")
synthesizer = ModelSynthesizer(client)
```

### Anthropic (Claude 3.5)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from smolfuzz.backends.llm_client import AnthropicClient
from smolfuzz.core.synthesizer import ModelSynthesizer

client = AnthropicClient(model="claude-3-5-sonnet-20241022")
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
└── backends/
    └── llm_client.py    # LLM backends (Ollama / OpenAI / Anthropic)
```
