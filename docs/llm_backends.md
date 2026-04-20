# LLM Backends

SMOLFuzz decouples model synthesis from the LLM provider through a small `LLMBackend` protocol defined in `llm_client.py`. Switching backends requires only changing which client object you pass to `ModelSynthesizer`.

## Built-in Backends

### Ollama (default — local models)

No API key required. Requires a running Ollama server.

```bash
ollama serve
ollama pull qwen2.5-coder:32b   # or any code-capable model
```

```python
from smolfuzz.llm_client import OllamaClient
from smolfuzz.synthesizer import ModelSynthesizer

client = OllamaClient(models=["qwen2.5-coder:32b"])
synthesizer = ModelSynthesizer(client)
```

**Via CLI:**

```bash
python3 -m smolfuzz.main --mode full --llm-models "qwen2.5-coder:32b"

# Round-robin across multiple models
python3 -m smolfuzz.main --mode full --llm-models "qwen2.5-coder:32b,llama3.3:70b"
```

### OpenAI

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

```python
from smolfuzz.llm_client import OpenAIClient
from smolfuzz.synthesizer import ModelSynthesizer

client = OpenAIClient(model="gpt-4o")
synthesizer = ModelSynthesizer(client)
```

Works with any OpenAI-compatible endpoint (Together AI, Fireworks, vLLM, etc.) by passing a custom `base_url` to `openai.OpenAI(base_url=...)`.

### Anthropic

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

```python
from smolfuzz.llm_client import AnthropicClient
from smolfuzz.synthesizer import ModelSynthesizer

client = AnthropicClient(model="claude-sonnet-4-6")
synthesizer = ModelSynthesizer(client)
```

## Plug In a Custom Backend

Implement the `LLMBackend` protocol — three members are required:

```python
from smolfuzz.llm_client import LLMBackend

class MyClient:
    @property
    def current_model(self) -> str:
        return "my-model"

    def generate(self, prompt: str, advance: bool = True) -> str:
        # Call your LLM here and return the response string.
        ...

    def stats(self) -> dict:
        return {}
```

Pass it directly to `ModelSynthesizer`:

```python
from smolfuzz.synthesizer import ModelSynthesizer

synthesizer = ModelSynthesizer(MyClient())
```

Or wire it into the full pipeline by editing `main.py`:

```python
# In run() inside main.py, replace:
client = OllamaClient(models=llm_models) if llm_models else OllamaClient()

# With your client:
client = MyClient()
```

## Configuration Reference

All three built-in clients accept these common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Maximum tokens per response |

`OllamaClient` additionally accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `models` | `["qwen2.5-coder:32b"]` | Model list to round-robin |
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `timeout` | `300` | Request timeout in seconds |
| `top_p` | `0.95` | Nucleus sampling probability |
