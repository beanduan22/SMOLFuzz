# Setup

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 with CUDA |
| TensorFlow | ≥ 2.13 with GPU |
| Ollama | latest (for local models) |
| GPU | NVIDIA with CUDA ≥ 11.8 |

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Set Up the LLM Backend

SMOLFuzz defaults to a local [Ollama](https://ollama.ai) server. Install Ollama and pull at least one code-capable model:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the default model (~20 GB VRAM)
ollama pull qwen2.5-coder:32b

# Lighter alternative (~8 GB VRAM)
ollama pull qwen2.5-coder:7b
```

Ollama must be running before you start SMOLFuzz:

```bash
ollama serve   # starts the local API on http://localhost:11434
```

To use OpenAI, Anthropic, or a custom backend instead, see [llm_backends.md](llm_backends.md).

## Verify GPU Access

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Both commands should report a GPU device. If not, check your CUDA driver and framework installation.

## Validate the Installation

Run the 5-model subset check to confirm the full pipeline works end-to-end:

```bash
python3 -m smolfuzz.main --mode subset
```

A successful run prints synthesis, execution, and oracle results for 5 models and exits cleanly.
