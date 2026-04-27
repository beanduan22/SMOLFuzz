from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skeleton:
    skeleton_id: str
    framework: str
    dimension: str
    description: str
    template: str


PYTORCH_SKELETONS: list[Skeleton] = [
    Skeleton(
        skeleton_id="PT-G1",
        framework="pytorch",
        dimension="gradient_tracking",
        description="requires_grad + backward",
        template="""import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
x.requires_grad_(True)
out = model(x)
out.sum().backward()


def make_inputs():
    return [x]
""",
    ),
    Skeleton(
        skeleton_id="PT-G2",
        framework="pytorch",
        dimension="gradient_tracking",
        description="torch.no_grad scope",
        template="""import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
with torch.no_grad():
    out = model(x)


def make_inputs():
    return [x]
""",
    ),
    Skeleton(
        skeleton_id="PT-M1",
        framework="pytorch",
        dimension="execution_mode",
        description="train/eval switch",
        template="""import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sens = None
        # MODE_LAYER_SLOT
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
model.train()
y_train = model(x)
model.eval()
y_eval = model(x)


def make_inputs():
    return [x]
""",
    ),
    Skeleton(
        skeleton_id="PT-M2",
        framework="pytorch",
        dimension="execution_mode",
        description="torch.jit.trace",
        template="""import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
y_eager = model(x)
traced = torch.jit.trace(model, x)
y_traced = traced(x)


def make_inputs():
    return [x]
""",
    ),
    Skeleton(
        skeleton_id="PT-D1",
        framework="pytorch",
        dimension="distribution_strategy",
        description="DistributedDataParallel wrap",
        template="""import os
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        return h


_tmpdir = tempfile.TemporaryDirectory()
_init_file = os.path.join(_tmpdir.name, "dist_init")
dist.init_process_group(
    backend="gloo",
    init_method=f"file://{_init_file}",
    rank=0,
    world_size=1,
)
device = "cpu"
model = Model().to(device)
ddp = DDP(model)
x = None
# INPUT_SLOT
out = ddp(x)
out.sum().backward()


def make_inputs():
    return [x]
""",
    ),
    Skeleton(
        skeleton_id="PT-D2",
        framework="pytorch",
        dimension="distribution_strategy",
        description="torch.distributed collective",
        template="""import os
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def forward(self, x):
        h = x
        # BODY_SLOT
        dist.all_reduce(h, op=REDUCE_OP_SLOT)
        return h


_tmpdir = tempfile.TemporaryDirectory()
_init_file = os.path.join(_tmpdir.name, "dist_init")
dist.init_process_group(
    backend="gloo",
    init_method=f"file://{_init_file}",
    rank=0,
    world_size=1,
)
model = Model()
x = None
# INPUT_SLOT
out = model(x)


def make_inputs():
    return [x]
""",
    ),
]


TENSORFLOW_SKELETONS: list[Skeleton] = [
    Skeleton(
        skeleton_id="TF-G1",
        framework="tensorflow",
        dimension="gradient_tracking",
        description="GradientTape + jacobian",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
with tf.GradientTape() as tape:
    tape.watch(x)
    out = model(x)
grad = tape.jacobian(out, x)
""",
    ),
    Skeleton(
        skeleton_id="TF-G2",
        framework="tensorflow",
        dimension="gradient_tracking",
        description="stop_gradient inside tape",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_A_SLOT
        h = tf.stop_gradient(h)
        # BODY_B_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
with tf.GradientTape() as tape:
    tape.watch(x)
    out = model(x)
grad = tape.gradient(out, x)
""",
    ),
    Skeleton(
        skeleton_id="TF-M1",
        framework="tensorflow",
        dimension="execution_mode",
        description="tf.function tracing",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
y_eager = model(x)
graph_fn = tf.function(model.__call__)
y_graph = graph_fn(x)
""",
    ),
    Skeleton(
        skeleton_id="TF-M2",
        framework="tensorflow",
        dimension="execution_mode",
        description="tf.function(jit_compile=True)",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        return h


model = Model()
x = None
# INPUT_SLOT
y_eager = model(x)


@tf.function(jit_compile=True)
def xla_fn(x):
    return model(x)
""",
    ),
    Skeleton(
        skeleton_id="TF-D1",
        framework="tensorflow",
        dimension="distribution_strategy",
        description="MirroredStrategy scope",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        return h


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = None
# INPUT_SLOT
out = strategy.run(
    lambda x: model(x),
    args=(x,),
)
""",
    ),
    Skeleton(
        skeleton_id="TF-D2",
        framework="tensorflow",
        dimension="distribution_strategy",
        description="tf.distribute collective",
        template="""import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.* = LAYER_SLOT

    def call(self, x):
        h = x
        # BODY_SLOT
        ctx = tf.distribute.get_replica_context()
        h = ctx.all_reduce(REDUCE_OP_SLOT, h)
        return h


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Model()
x = None
# INPUT_SLOT
out = strategy.run(
    lambda x: model(x),
    args=(x,),
)
""",
    ),
]


def get_skeletons(framework: str) -> list[Skeleton]:
    if framework == "pytorch":
        return list(PYTORCH_SKELETONS)
    if framework == "tensorflow":
        return list(TENSORFLOW_SKELETONS)
    raise ValueError(f"unsupported framework: {framework}")
