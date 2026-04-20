"""
Prompt templates for SMOLFuzz model synthesis and self-repair.

Based on Appendix B of the SMOLFuzz paper, adapted for PyTorch.
Three dependency types are exercised:
  1. Computational graph  – autograd / gradient scopes
  2. Mode                 – train() / eval(), no_grad / enable_grad
  3. Concurrency          – torch.no_grad / enable_grad context managers,
                            inference_mode
"""

# ------------------------------------------------------------------ #
# Few-shot examples showing each dependency type                      #
# ------------------------------------------------------------------ #

_EXAMPLE_GRAD = '''
# Example – Computational-graph dependency
import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        # Gradient-tracking scope changes semantics of subsequent ops
        x = x.requires_grad_(True)
        with torch.enable_grad():
            y = self.fc(x)
            z = torch.sin(y) * torch.cos(y)
            # Jacobian computation depends on the autograd graph built above
            jac = torch.autograd.functional.jacobian(
                lambda t: torch.sin(self.fc(t)), x
            )
        return jac.sum()

def make_inputs():
    return [torch.randn(2, 4)]

USED_APIS = ["torch.nn.Linear", "torch.sin", "torch.cos",
             "torch.autograd.functional.jacobian", "torch.enable_grad"]
'''

_EXAMPLE_MODE = '''
# Example – Mode dependency (deterministic — no stochastic ops)
import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(8)
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        # train() vs eval() changes BatchNorm statistics behaviour
        self.train()
        x = self.bn(x)          # uses batch statistics in train mode
        self.eval()
        x = self.bn(x)          # uses running statistics in eval mode
        x = self.ln(x.detach())
        x = self.fc(x)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.BatchNorm1d", "torch.nn.LayerNorm", "torch.nn.Linear"]
'''

_EXAMPLE_CTX = '''
# Example – Concurrency / context dependency
import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x):
        # Operations inside no_grad share no gradient graph with those outside
        with torch.no_grad():
            a = self.fc(x)
            b = torch.relu(a)
        # Re-enable gradients: operations here DO track gradients
        with torch.enable_grad():
            c = b.detach().requires_grad_(True)
            d = torch.tanh(c)
            g = torch.autograd.grad(d.sum(), c)[0]
        return d + g

def make_inputs():
    return [torch.randn(3, 6)]

USED_APIS = ["torch.nn.Linear", "torch.relu", "torch.no_grad",
             "torch.enable_grad", "torch.tanh", "torch.autograd.grad"]
'''

FEW_SHOT_EXAMPLES = "\n".join([_EXAMPLE_GRAD, _EXAMPLE_MODE, _EXAMPLE_CTX])


# ------------------------------------------------------------------ #
# Synthesis prompt                                                     #
# ------------------------------------------------------------------ #

SYNTHESIS_TEMPLATE = """You are an expert PyTorch developer. Write ONE complete, self-contained, executable Python script that tests PyTorch library behavior using the given APIs.

**Target library**: PyTorch

**Candidate APIs** (use as many as you can without breaking correctness):
{api_set}

**STRICT RULES** (violating any rule causes test failure):

### Shape Consistency (most common mistake — read carefully)
- Use **one fixed 2D shape throughout**: `(batch=4, features=8)`.
  `make_inputs()` returns `[torch.randn(4, 8)]`.
- Every `nn.Linear(in, out)` must have `in == 8` (or the output dim of the previous Linear).
- Every `nn.BatchNorm1d(n)` must have `n == 8` (or the features dim at that point).
- **NEVER change the shape with `.view()`, `.reshape()`, or `pixel_shuffle`/`pixel_unshuffle`.**
  These are the #1 cause of failures. Avoid them entirely.
  If you must flatten, use `nn.Flatten()` only as the *last* step before a Linear.
- Avoid any op that needs spatial / 4D tensors (Conv2d, Conv3d, pool layers, etc.).
- All ops must accept `(4, 8)` float tensors and produce `(4, *)` float tensors.

### Code Structure
1. `import torch; import torch.nn as nn` at the top.
2. Define `class Model(nn.Module)` with an `__init__` and `forward(self, x)` (single input tensor is simplest).
3. `forward` MUST form a **sequential dependency chain** where output of each op feeds the next.
4. Include at least ONE of these API dependency contexts:
   - `with torch.enable_grad():` or `torch.no_grad():` scopes around ops
   - `self.train()` / `self.eval()` mode switches inside forward
   - `x.requires_grad_(True)` combined with `torch.autograd.grad(...)` or `.backward()`
5. Define `make_inputs()` returning `[torch.randn(BATCH, FEATURES)]` matching your model's expected input.
6. Define `USED_APIS = [...]` listing actual APIs used from the candidate set.
7. Do NOT include a `if __name__ == "__main__":` block — the runner adds one.

### Determinism (CRITICAL — violations are automatically detected and rejected)
- `Model.__init__` and `Model.forward()` MUST be **fully deterministic**.
- **NEVER** place any of these inside the Model class:
  `torch.rand`, `torch.randn`, `torch.randint`, `torch.randperm`,
  `torch.rand_like`, `torch.randn_like`, `torch.randint_like`,
  `torch.bernoulli`, `torch.multinomial`, `torch.poisson`, `torch.normal`,
  `nn.Dropout`, `F.dropout`, `torch.dropout`, or any other stochastic op.
- Random tensors belong **only** in `make_inputs()` — the runner saves them
  once and reloads identical tensors for every device, so `make_inputs()` is
  always deterministic at comparison time.
- If a candidate API is inherently stochastic (e.g. `torch.distributions.*`,
  `torch.nn.Dropout`), **skip it silently**.

### Skipping Hard-to-Use APIs
If an API from the candidate set is difficult to incorporate without shape errors
(e.g., `pairwise_distance`, `instance_norm` needing 4D input, boolean ops on float tensors),
**skip that API silently** — do not include it in `USED_APIS`.
It is better to skip 3-4 APIs than to produce a broken model.

**Reference examples** (follow these dependency patterns):
{few_shot}

Write the complete script now. Output ONLY valid Python code — no markdown, no explanation.
"""


# ------------------------------------------------------------------ #
# Self-repair prompt                                                   #
# ------------------------------------------------------------------ #

REPAIR_TEMPLATE = """The following PyTorch model script failed during execution.

**Error** (read carefully):
{error}

**Original script**:
{original_code}

**Fix instructions** (apply ALL that are relevant):
1. Correct the error with **minimal edits** — do not rewrite the whole model.
   If the error mentions "Determinism violation": remove every stochastic op
   (torch.rand*, torch.bernoulli, nn.Dropout, F.dropout, torch.dropout, etc.)
   from the Model class. Replace with deterministic alternatives:
   - nn.Dropout → remove it entirely, or replace with nn.LayerNorm / nn.Linear
   - torch.randn(...) inside forward → replace with a deterministic transform of x
2. If the error mentions shape mismatch (e.g. "mat1 and mat2 shapes cannot be multiplied", "running_mean should contain N elements", "shape is invalid for input of size"):
   - **Remove any `.view()`, `.reshape()`, `pixel_shuffle`, or `pixel_unshuffle` calls** — these are the most common cause.
   - Adjust `nn.Linear(in, out)`, `nn.BatchNorm1d(num_features)` sizes so they match the actual tensor dims.
   - Keep all tensors as 2D `(batch, features)` throughout the forward pass.
3. If the error is a `SyntaxError`: ensure the output is pure Python with no markdown fences, no natural-language sentences.
4. Preserve the original API dependency chain and context managers (autograd/mode/no_grad).
5. Ensure `USED_APIS` reflects the APIs actually used after repair.
6. The script must be fully self-contained and executable — no external files or network calls.

Output ONLY the corrected Python code — no markdown, no explanation, no triple backticks.
"""


def build_synthesis_prompt(api_list: list[str]) -> str:
    api_block = "\n".join(f"  - {a}" for a in api_list)
    return SYNTHESIS_TEMPLATE.format(
        api_set=api_block,
        few_shot=FEW_SHOT_EXAMPLES,
    )


def build_repair_prompt(original_code: str, error: str) -> str:
    return REPAIR_TEMPLATE.format(
        error=error[:2000],          # truncate very long tracebacks
        original_code=original_code,
    )


# ------------------------------------------------------------------ #
# TensorFlow prompts                                                  #
# ------------------------------------------------------------------ #

_TF_EXAMPLE_GRAD = '''
# Example – Computational-graph dependency (TF GradientTape)
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(8)
        self.d2 = tf.keras.layers.Dense(8)

    def call(self, x, training=False):
        # Tape-recorded computation changes the semantics of grads/jvp
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.d1(x)
            z = tf.sin(y) * tf.cos(y)
        # Gradient is only defined inside the tape scope
        g = tape.gradient(z, x)
        return self.d2(z + g)

USED_APIS = ["tf.keras.layers.Dense", "tf.sin", "tf.cos", "tf.GradientTape"]
'''

_TF_EXAMPLE_MODE = '''
# Example – Mode dependency (training vs inference)
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.ln = tf.keras.layers.LayerNormalization()
        self.d  = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        # BatchNormalization uses batch stats in train mode and running stats
        # in inference mode — changing `training` flips semantics.
        y = self.bn(x, training=training)
        y = self.ln(y)
        return self.d(y)

USED_APIS = ["tf.keras.layers.BatchNormalization",
             "tf.keras.layers.LayerNormalization", "tf.keras.layers.Dense"]
'''

_TF_EXAMPLE_CTX = '''
# Example – Concurrency / strategy-scope dependency
import tensorflow as tf

# Variables created inside strategy.scope() are distributed; ops outside
# the scope run on the default device.  This alters variable placement and
# collective synchronization semantics (paper §1 dependency type iii).
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        with strategy.scope():
            self.d1 = tf.keras.layers.Dense(8)
            self.d2 = tf.keras.layers.Dense(4)

    def call(self, x, training=False):
        with strategy.scope():
            a = self.d1(x)
            b = tf.nn.relu(a)
        c = self.d2(b)
        return c

USED_APIS = ["tf.distribute.OneDeviceStrategy", "tf.keras.layers.Dense",
             "tf.nn.relu"]
'''

TF_FEW_SHOT_EXAMPLES = "\n".join(
    [_TF_EXAMPLE_GRAD, _TF_EXAMPLE_MODE, _TF_EXAMPLE_CTX]
)


TF_SYNTHESIS_TEMPLATE = """You are an expert TensorFlow 2.x developer. Write ONE complete, self-contained class definition that exercises TensorFlow library behavior using the given APIs.

**Target library**: TensorFlow 2.x

**Candidate APIs** (use as many as you can without breaking correctness):
{api_set}

**STRICT RULES** (violating any rule causes the test to be skipped):

### Shape Consistency
- Input `x` is shape `[4, 8]` float32 throughout.
- Every `tf.keras.layers.Dense(units)` keeps the feature dim consistent with
  the previous op. Output feature dim may change, but batch dim stays at 4.
- NEVER reshape to a 4D NHWC tensor or 3D sequence tensor.
- Avoid any API that needs spatial tensors (Conv*, Pool*, UpSampling*,
  LSTM/GRU, Embedding, Attention).

### Code Structure
1. Define ONLY `class Model(tf.keras.Model)` with `__init__` and
   `call(self, x, training=False)`.
2. Instantiate ALL sublayers (Dense, BatchNormalization, LayerNormalization,
   Activation, ...) in `__init__` — never inside `call`.
3. `call` MUST form a sequential dependency chain where the output of one
   op feeds the next.
4. Include at least ONE of these API dependency contexts:
   - `tf.GradientTape()` scope around ops (computational-graph dependency)
   - training vs inference switch via the `training` kwarg (mode dependency)
   - `tf.device(...)` scope around ops (concurrency/device dependency)
5. Define `USED_APIS = [...]` listing the APIs from the candidate set that
   are actually used.
6. Do NOT import anything — `import tensorflow as tf` is already available.
7. Do NOT include a `if __name__ == "__main__":` block.

### Determinism (CRITICAL — violations would false-positive the oracle)
- `Model.__init__` and `Model.call()` MUST be fully deterministic.
- NEVER use any stochastic API: `tf.random.*`, `tf.keras.layers.Dropout`,
  `tf.keras.layers.GaussianNoise`, `tf.experimental.numpy.random.*`.
- If a candidate API is stochastic, skip it silently.

### Skipping Hard-to-Use APIs
If a candidate API is hard to integrate with `[4, 8]` float32 input
(e.g., needs 4D, integer indices, or a specific mask shape), skip it silently
and do not list it in `USED_APIS`. Better to skip a few APIs than ship a
broken class.

**Reference examples** (each illustrates one dependency type):
{few_shot}

Write the complete class definition now. Output ONLY valid Python code — no markdown, no explanation, no imports.
"""


TF_REPAIR_TEMPLATE = """The following TensorFlow model class failed during execution.

**Error** (read carefully):
{error}

**Original class**:
{original_code}

**Fix instructions** (apply ALL that are relevant):
1. Correct the error with minimal edits — do not rewrite the whole class.
2. Keep `class Model(tf.keras.Model)` with `__init__` and
   `call(self, x, training=False)` only. Do not add imports.
3. Input is fixed at shape `[4, 8]` float32. If a shape mismatch is raised,
   adjust Dense/BatchNorm/LayerNorm units so dims stay consistent, and
   avoid reshapes to 4D/3D tensors.
4. If the error mentions stochastic ops (tf.random, Dropout, GaussianNoise),
   remove them entirely from the class.
5. Preserve the original dependency context (GradientTape / training mode /
   device scope).
6. `USED_APIS` must list the APIs actually used after repair.

Output ONLY the corrected Python class — no markdown, no explanation, no imports.
"""


def build_tf_synthesis_prompt(api_list: list[str]) -> str:
    api_block = "\n".join(f"  - {a}" for a in api_list)
    return TF_SYNTHESIS_TEMPLATE.format(
        api_set=api_block,
        few_shot=TF_FEW_SHOT_EXAMPLES,
    )


def build_tf_repair_prompt(original_code: str, error: str) -> str:
    return TF_REPAIR_TEMPLATE.format(
        error=error[:2000],
        original_code=original_code,
    )
