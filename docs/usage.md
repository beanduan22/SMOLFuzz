# Usage

All commands must be run from the repository root (the directory containing `smolfuzz/`).

## PyTorch Campaign

```bash
# Quick validation — 5 models, no GPU needed
python3 -m smolfuzz.main --mode subset

# Standard run — 300 models, 60-second mutation budget each
python3 -m smolfuzz.main --mode full --models 300 --budget 60

# Custom API set size (paper default: 30)
python3 -m smolfuzz.main --mode full --models 300 --api-set-size 30

# Custom output directory
python3 -m smolfuzz.main --mode full --output-dir /tmp/smolfuzz_pt
```

## TensorFlow Campaign

```bash
# Standard run
python3 -m smolfuzz.run_tf --models 300 --budget 60

# Custom output directory
python3 -m smolfuzz.run_tf --models 300 --out /tmp/smolfuzz_tf
```

## Run Both Frameworks in Parallel

```bash
python3 run_both.py --models 300 --budget 60
```

This launches PyTorch and TensorFlow campaigns simultaneously and writes a combined summary when both finish.

## CLI Reference

### `smolfuzz.main` (PyTorch)

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `subset` | `subset` = 5 models; `full` = `--models` count |
| `--models` | `300` | Number of models to synthesise |
| `--api-set-size` | `30` | APIs selected per model |
| `--budget` | `60` | Mutation fuzzing budget in seconds per model |
| `--api-file` | `torch_valid_apis.txt` | API pool file |
| `--output-dir` | `results/` | Where to write results |
| `--llm-models` | `qwen2.5-coder:32b` | Comma-separated Ollama model names |

### `smolfuzz.run_tf` (TensorFlow)

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `300` | Number of models |
| `--api-set-size` | `30` | APIs selected per model |
| `--budget` | `60` | Mutation budget per model in seconds |
| `--out` | `results/tf_run` | Output directory |
| `--apis` | `tf_valid_apis.txt` | API pool file |

### `run_both.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--models` | `500` | Models per framework |
| `--budget` | `60` | Mutation budget per model in seconds |
| `--api-set-size` | `30` | APIs per PyTorch model |
| `--tf-api-set-size` | `30` | APIs per TensorFlow model |

## Output

Results are written to `results/` (gitignored) with the following layout:

```
results/
├── models/          # synthesised model scripts
├── workspace/       # temporary per-run files
├── bugs/            # oracle reports (.json) + reproducer scripts (.repro.py)
└── coverage.json    # API coverage summary
```

Each detected anomaly produces a `.json` report and a standalone `.repro.py` that reproduces the divergence without running the fuzzer.
