from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Set


def _build_exclusions() -> set:
    excluded = {
        "torch.nn.functional.instance_norm",
        "torch.nn.functional.upsample_nearest", "torch.nn.functional.upsample_bilinear",
        "torch.nn.functional.avg_pool1d", "torch.nn.functional.avg_pool2d",
        "torch.nn.functional.avg_pool3d", "torch.nn.functional.max_pool1d",
        "torch.nn.functional.max_pool2d", "torch.nn.functional.max_pool3d",
        "torch.nn.functional.adaptive_avg_pool1d",
        "torch.nn.functional.adaptive_avg_pool2d",
        "torch.nn.functional.adaptive_max_pool1d",
        "torch.nn.functional.adaptive_max_pool2d",
        "torch.nn.functional.conv1d", "torch.nn.functional.conv2d",
        "torch.nn.functional.conv3d", "torch.nn.functional.conv_transpose1d",
        "torch.nn.functional.conv_transpose2d", "torch.nn.functional.conv_transpose3d",
        "torch.nn.functional.fold", "torch.nn.functional.unfold",
        "torch.nn.Softmax2d",
        "torch.nn.Conv1d", "torch.nn.Conv2d", "torch.nn.Conv3d",
        "torch.nn.ConvTranspose1d", "torch.nn.ConvTranspose2d", "torch.nn.ConvTranspose3d",
        "torch.nn.AvgPool1d", "torch.nn.AvgPool2d", "torch.nn.AvgPool3d",
        "torch.nn.MaxPool1d", "torch.nn.MaxPool2d", "torch.nn.MaxPool3d",
        "torch.nn.AdaptiveAvgPool1d", "torch.nn.AdaptiveAvgPool2d", "torch.nn.AdaptiveAvgPool3d",
        "torch.nn.AdaptiveMaxPool1d", "torch.nn.AdaptiveMaxPool2d", "torch.nn.AdaptiveMaxPool3d",
        "torch.nn.MaxUnpool1d", "torch.nn.MaxUnpool2d", "torch.nn.MaxUnpool3d",
        "torch.nn.Upsample",
        "torch.nn.RNN", "torch.nn.LSTM", "torch.nn.GRU",
        "torch.nn.RNNCell", "torch.nn.LSTMCell", "torch.nn.GRUCell",
        "torch.nn.BatchNorm2d", "torch.nn.BatchNorm3d",
        "torch.nn.InstanceNorm1d", "torch.nn.InstanceNorm2d", "torch.nn.InstanceNorm3d",
        "torch.nn.Dropout2d", "torch.nn.Dropout3d", "torch.nn.FeatureAlphaDropout",
        "torch.nn.AlphaDropout",
        "torch.nn.Embedding", "torch.nn.EmbeddingBag",
        "torch.nn.ReflectionPad1d", "torch.nn.ReflectionPad2d", "torch.nn.ReflectionPad3d",
        "torch.nn.ReplicationPad1d", "torch.nn.ReplicationPad2d", "torch.nn.ReplicationPad3d",
        "torch.nn.ZeroPad1d", "torch.nn.ZeroPad2d", "torch.nn.ZeroPad3d",
        "torch.nn.ConstantPad1d", "torch.nn.ConstantPad2d", "torch.nn.ConstantPad3d",
        "torch.Tensor.gcd_", "torch.Tensor.lcm_",
        "torch.gcd", "torch.lcm",
        "torch.Tensor.bitwise_and_", "torch.Tensor.bitwise_or_",
        "torch.Tensor.bitwise_xor_", "torch.Tensor.bitwise_not_",
        "torch.bitwise_and", "torch.bitwise_or", "torch.bitwise_not",
        "torch.Tensor.bitwise_right_shift", "torch.Tensor.bitwise_left_shift",
        "torch.nn.MultiheadAttention", "torch.nn.Transformer",
        "torch.nn.TransformerEncoder", "torch.nn.TransformerDecoder",
        "torch.nn.TransformerEncoderLayer", "torch.nn.TransformerDecoderLayer",
        "torch.nn.functional.embedding",
        "torch.nn.functional.embedding_bag",
        "torch.nn.functional.nll_loss",
        "torch.nn.functional.cross_entropy",
        "torch.nn.functional.ctc_loss",
        "torch.nn.functional.one_hot",
        "torch.nn.functional.pairwise_distance",
        "torch.nn.functional.cosine_similarity",
        "torch.Tensor.bool", "torch.Tensor.byte",
        "torch.nn.functional.rnn_tanh_cell", "torch.nn.functional.rnn_relu_cell",
        "torch.nn.functional.lstm_cell", "torch.nn.functional.gru_cell",
        "torch.Tensor.coalesce", "torch.Tensor.dequantize",
        "torch.Tensor.dense_dim", "torch.Tensor.sparse_dim",
        "torch.distributed.is_torchelastic_launched",
        "torch.distributed.is_available",
        "torch.distributed.is_initialized",
        "torch.distributed.get_rank",
        "torch.distributed.get_world_size",
        "torch.random.get_rng_state", "torch.random.set_rng_state",
        "torch.random.initial_seed",
        "torch.utils.data.Sampler", "torch.utils.data.Subset",
        "torch.utils.data.DataLoader", "torch.utils.data.Dataset",
        "torch.utils.data.TensorDataset", "torch.utils.data.ConcatDataset",
        "torch.utils.data.random_split",
    }
    return excluded

_EXCLUDED_APIS = _build_exclusions() | {
    "tf.keras.layers.Conv1D", "tf.keras.layers.Conv2D", "tf.keras.layers.Conv3D",
    "tf.keras.layers.Conv1DTranspose", "tf.keras.layers.Conv2DTranspose",
    "tf.keras.layers.Conv3DTranspose",
    "tf.keras.layers.SeparableConv1D", "tf.keras.layers.SeparableConv2D",
    "tf.keras.layers.DepthwiseConv1D", "tf.keras.layers.DepthwiseConv2D",
    "tf.keras.layers.MaxPool1D", "tf.keras.layers.MaxPool2D", "tf.keras.layers.MaxPool3D",
    "tf.keras.layers.AveragePooling1D", "tf.keras.layers.AveragePooling2D",
    "tf.keras.layers.AveragePooling3D",
    "tf.keras.layers.GlobalMaxPool1D", "tf.keras.layers.GlobalMaxPool2D",
    "tf.keras.layers.GlobalMaxPool3D",
    "tf.keras.layers.GlobalAveragePooling1D", "tf.keras.layers.GlobalAveragePooling2D",
    "tf.keras.layers.GlobalAveragePooling3D",
    "tf.keras.layers.UpSampling1D", "tf.keras.layers.UpSampling2D",
    "tf.keras.layers.UpSampling3D",
    "tf.keras.layers.ZeroPadding1D", "tf.keras.layers.ZeroPadding2D",
    "tf.keras.layers.ZeroPadding3D",
    "tf.keras.layers.Cropping1D", "tf.keras.layers.Cropping2D",
    "tf.keras.layers.Cropping3D",
    "tf.keras.layers.LocallyConnected1D", "tf.keras.layers.LocallyConnected2D",
    "tf.keras.layers.RNN", "tf.keras.layers.LSTM", "tf.keras.layers.GRU",
    "tf.keras.layers.LSTMCell", "tf.keras.layers.GRUCell",
    "tf.keras.layers.SimpleRNN", "tf.keras.layers.SimpleRNNCell",
    "tf.keras.layers.Bidirectional", "tf.keras.layers.TimeDistributed",
    "tf.keras.layers.ConvLSTM1D", "tf.keras.layers.ConvLSTM2D",
    "tf.keras.layers.ConvLSTM3D", "tf.keras.layers.ConvLSTMCell",
    "tf.keras.layers.StackedRNNCells",
    "tf.keras.layers.Embedding",
    "tf.keras.layers.Attention", "tf.keras.layers.AdditiveAttention",
    "tf.keras.layers.MultiHeadAttention",
    "tf.keras.layers.Dropout", "tf.keras.layers.SpatialDropout1D",
    "tf.keras.layers.SpatialDropout2D", "tf.keras.layers.SpatialDropout3D",
    "tf.keras.layers.AlphaDropout",
    "tf.keras.layers.GaussianDropout", "tf.keras.layers.GaussianNoise",
    "tf.nn.conv1d", "tf.nn.conv2d", "tf.nn.conv3d",
    "tf.nn.conv1d_transpose", "tf.nn.conv2d_transpose", "tf.nn.conv3d_transpose",
    "tf.nn.depthwise_conv2d", "tf.nn.separable_conv2d",
    "tf.nn.atrous_conv2d", "tf.nn.atrous_conv2d_transpose",
    "tf.nn.max_pool", "tf.nn.max_pool1d", "tf.nn.max_pool2d", "tf.nn.max_pool3d",
    "tf.nn.avg_pool", "tf.nn.avg_pool1d", "tf.nn.avg_pool2d", "tf.nn.avg_pool3d",
    "tf.nn.fractional_avg_pool", "tf.nn.fractional_max_pool",
    "tf.nn.ctc_loss", "tf.nn.ctc_beam_search_decoder",
    "tf.nn.ctc_greedy_decoder", "tf.nn.collapse_repeated",
    "tf.nn.embedding_lookup", "tf.nn.embedding_lookup_sparse",
    "tf.nn.sampled_softmax_loss", "tf.nn.nce_loss",
    "tf.signal.stft", "tf.signal.inverse_stft",
    "tf.keras.losses.CategoricalCrossentropy",
    "tf.keras.losses.SparseCategoricalCrossentropy",
    "tf.keras.losses.CategoricalHinge",
    "tf.keras.losses.BinaryFocalCrossentropy",
    "tf.keras.losses.CategoricalFocalCrossentropy",
    "tf.experimental.numpy.random.standard_normal",
    "tf.experimental.numpy.random.rand",
    "tf.experimental.numpy.random.randn",
    "tf.keras.backend.set_epsilon", "tf.keras.backend.set_floatx",
    "tf.keras.backend.set_image_data_format",
    "tf.keras.backend.set_learning_phase",
    "tf.keras.backend.clear_session",
    "tf.keras.backend.manual_variable_initialization",
    "tf.keras.losses.serialize", "tf.keras.losses.deserialize",
    "tf.keras.metrics.serialize", "tf.keras.metrics.deserialize",
    "tf.keras.optimizers.serialize", "tf.keras.optimizers.deserialize",
    "tf.keras.activations.serialize", "tf.keras.activations.deserialize",
    "tf.keras.regularizers.serialize", "tf.keras.regularizers.deserialize",
    "tf.keras.constraints.serialize", "tf.keras.constraints.deserialize",
    "tf.keras.initializers.serialize", "tf.keras.initializers.deserialize",
    "tf.keras.layers.serialize", "tf.keras.layers.deserialize",
    "tf.keras.models.clone_model", "tf.keras.models.model_from_config",
    "tf.keras.models.model_from_json",
    "tf.nondifferentiable_batch_function",
    "tf.recompute_grad", "tf.custom_gradient",
    "tf.nn.RNNCellResidualWrapper", "tf.nn.RNNCellDeviceWrapper",
    "tf.nn.RNNCellDropoutWrapper",
}
_EXCLUDED_PREFIXES = (
    "torch.optim.",
    "torch.distributed.",
    "torch.utils.data.",
    "torch.hub.",
    "torch.fx.",
    "torch.profiler.",
    "torch.futures.",
    "torch.package.",
    "torch.jit.",
    "torch.overrides.",
    "torch.nn.Lazy",
    "torch.nn.LPPool",
    "torch.nn.FractionalMaxPool",
    "tf.experimental.numpy.random",
    "tf.keras.applications.",
    "tf.keras.datasets.",
    "tf.keras.preprocessing.",
    "tf.keras.callbacks.",
    "tf.keras.mixed_precision.",
    "tf.keras.utils.",
    "tf.keras.estimator.",
)


class MultiRouletteSelector:
    def __init__(self, groups: Dict[str, List[str]]) -> None:
        def _keep(api: str) -> bool:
            if api in _EXCLUDED_APIS:
                return False
            if any(api.startswith(p) for p in _EXCLUDED_PREFIXES):
                return False
            return True

        _SKIP_GROUPS = {"random_generation", "_excluded"}
        self._groups = {
            g: [a for a in apis if _keep(a)]
            for g, apis in groups.items()
            if g not in _SKIP_GROUPS and any(_keep(a) for a in apis)
        }
        self._usage: Dict[str, int] = defaultdict(int)
        self._bug_apis: Set[str] = set()

    def select(self, n: int = 30) -> List[str]:
        groups = self._groups
        if not groups:
            return []

        unused_counts = {
            g: sum(1 for api in apis if self._usage[api] == 0)
            for g, apis in groups.items()
        }
        total_unused = sum(unused_counts.values()) or 1

        quotas_real = {
            g: n * unused_counts[g] / total_unused
            for g in groups
        }
        floors = {g: math.floor(q) for g, q in quotas_real.items()}
        remainder_budget = n - sum(floors.values())
        fractions = sorted(
            groups.keys(),
            key=lambda g: -(quotas_real[g] - floors[g])
        )
        allocations: Dict[str, int] = dict(floors)
        for g in fractions:
            if remainder_budget <= 0:
                break
            group_size = len(groups[g])
            if allocations[g] < group_size:
                allocations[g] += 1
                remainder_budget -= 1

        selected: List[str] = []
        for g, apis in groups.items():
            k = min(allocations[g], len(apis))
            if k <= 0:
                continue
            selected.extend(self._roulette(apis, k))

        return selected

    def record_usage(self, apis: List[str], triggered_bug: bool = False) -> None:
        for api in apis:
            if triggered_bug:
                self._bug_apis.add(api)
            if api not in self._bug_apis:
                self._usage[api] += 1

    def stats(self) -> Dict[str, int]:
        return {
            "total_unique_used": sum(1 for v in self._usage.values() if v > 0),
            "bug_exempt_apis": len(self._bug_apis),
        }

    def _score(self, api: str) -> float:
        return 1.0 / (self._usage[api] + 1)

    def _roulette(self, apis: List[str], k: int) -> List[str]:
        pool = list(apis)
        chosen: List[str] = []
        for _ in range(k):
            if not pool:
                break
            scores = [self._score(a) for a in pool]
            total = sum(scores)
            if total == 0:
                idx = random.randrange(len(pool))
            else:
                r = random.uniform(0, total)
                cumulative = 0.0
                idx = len(pool) - 1
                for i, s in enumerate(scores):
                    cumulative += s
                    if cumulative >= r:
                        idx = i
                        break
            chosen.append(pool[idx])
            pool.pop(idx)
        return chosen
