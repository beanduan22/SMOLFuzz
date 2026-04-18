"""
Load and classify PyTorch / TensorFlow APIs from the text files into the 11
functional groups defined in the SMOLFuzz paper.

Groups (from paper):
  1.  creation_conversion   - tensor creation and dtype/device conversion       PT=127  TF=38
  2.  mathematics           - arithmetic, linear algebra, trigonometry           PT=577  TF=306
  3.  reshaping             - reshape, view, slice, transpose, concat            PT=115  TF=44
  4.  logical               - comparison, boolean, conditional                   PT=86   TF=18
  5.  distributions         - probability distributions (parametric families)    PT=13   TF=9
  6.  forward_layers        - nn layers (Conv, Linear, BN, RNN, Attention …)    PT=278  TF=566
  7.  gradients_optim       - autograd, optimizers, gradient utilities           PT=28   TF=44
  8.  storage_serial        - save/load, serialization                           PT=11   TF=3
  9.  random_generation     - seeded random tensor generation (EXCLUDED)         PT=23   TF=37
  10. model_io              - hub, checkpoint, scripting helpers                 PT=16   TF=125
  11. misc                  - everything else                                    PT=55   TF=1025

  Selectable pool (random_generation excluded): PyTorch=1226, TensorFlow=1135
  See docs/api_classification.md for full definitions and rationale.

Exclusion policy (applied before classification, per paper §3.1.1):

    • random_generation ops are EXCLUDED from the selectable pool because they
      destroy CPU/GPU determinism and produce false positives in differential
      testing — every CPU vs GPU comparison would diverge by construction.
    • Experimental / deprecated APIs (private namespaces ``torch._*``,
      ``tf.compat.v1.*``, ``tf.experimental.*``) are excluded.
    • Compiler-oriented front-ends (``torch.jit``, ``torch.fx``,
      ``torch._dynamo``, ``torch._inductor``, ``torch.compile``,
      ``torch.export``, ``torch.onnx``, ``tf.function``, ``tf.autograph``,
      ``tf.tpu.*``, ``tf.xla.*``, ``tf.lite.*``) are excluded — they belong
      to the compilation stack, not the runtime tensor layer we fuzz.
    • Distributed / data-loading / device-utility / profiler infrastructure
      (``torch.distributed``, ``torch.utils.data``, ``torch.cuda.*``,
      ``torch.backends.*``, ``torch.profiler``, ``torch.futures``,
      ``torch.overrides``, ``tf.distribute.*``, ``tf.io.*``, ``tf.data.*``,
      ``tf.summary.*``, ``tf.tpu.*``, ``tf.config.*``) are excluded — they
      are not numerical operators.

The excluded APIs are still returned in the ``"_excluded"`` group for
inspection so callers can audit what was filtered.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------- #
# Exclusion patterns — applied BEFORE classification.                   #
# Anything matching here is dropped from the selectable pool.           #
# --------------------------------------------------------------------- #

# Random number generation — would make CPU/GPU differential testing
# trivially false-positive. EXCLUDED per user requirement and paper text
# ("we exclude APIs explicitly marked as experimental or deprecated...
#  random generation is in its own group but is not part of the
#  numerical fuzzing pool").
_RANDOM_OP_REGEX = re.compile(
    r"^("
    r"torch\.(rand|randn|randint|randperm|seed|multinomial|poisson|bernoulli|normal)$"
    r"|torch\.(rand_like|randn_like|randint_like)$"
    r"|torch\.random\."
    r"|torch\.manual_seed$"
    r"|torch\.initial_seed$"
    r"|torch\.set_rng_state$"
    r"|torch\.get_rng_state$"
    r"|torch\.Tensor\.(random_|bernoulli_?|normal_|uniform_|cauchy_|exponential_|geometric_|log_normal_|multinomial)$"
    r"|tf\.random\."
    r"|tf\.compat\.v1\.random\."
    r"|tf\.compat\.v1\.set_random_seed$"
    r"|tf\.set_random_seed$"
    r"|tf\.random_normal$"
    r"|tf\.random_uniform$"
    r"|tf\.random_shuffle$"
    r"|tf\.random_crop$"
    r"|torch\.quasirandom($|\.)"   # Sobol/Halton sequence generators
    r"|tf\.random_normal_initializer$"
    r"|tf\.random_uniform_initializer$"
    r")"
)

# Experimental / deprecated / private namespaces.
_PRIVATE_REGEX = re.compile(
    r"^("
    r"torch\._"                  # torch._dynamo, torch._inductor, torch._refs ...
    r"|torch\.testing\."          # internal testing helpers
    r"|tf\.compat\.v1\."          # legacy v1 surface
    r"|tf\.experimental\."        # experimental
    r"|tf\.compat\.v2\.experimental\."
    r"|tf\.contrib\."             # removed/legacy
    r")"
)

# Compiler / tracing front-ends — not the runtime tensor layer.
_COMPILER_REGEX = re.compile(
    r"^("
    r"torch\.jit($|\.)"
    r"|torch\.fx($|\.)"
    r"|torch\.compile$"
    r"|torch\.compiler($|\.)"
    r"|torch\.export($|\.)"
    r"|torch\.onnx($|\.)"
    r"|torch\.package($|\.)"
    r"|tf\.function$"
    r"|tf\.autograph($|\.)"
    r"|tf\.xla($|\.)"
    r"|tf\.tpu($|\.)"
    r"|tf\.lite($|\.)"
    r"|tf\.saved_model($|\.)"
    r"|tf\.compat\.v1\.lite($|\.)"
    r")"
)

# Distributed / data / device / profiler infrastructure — not numerical ops.
_INFRA_REGEX = re.compile(
    r"^("
    r"torch\.distributed($|\.)"
    r"|torch\.utils\."
    r"|torch\.cuda($|\.)"
    r"|torch\.backends($|\.)"
    r"|torch\.profiler($|\.)"
    r"|torch\.futures($|\.)"
    r"|torch\.overrides($|\.)"
    r"|torch\.hub($|\.)"
    r"|torch\.multiprocessing($|\.)"
    r"|torch\.compiled_with_cxx11_abi$"
    r"|torch\.are_deterministic_algorithms_enabled$"
    r"|torch\.use_deterministic_algorithms$"
    r"|torch\.set_default_"
    r"|torch\.get_default_"
    r"|torch\.set_num_"
    r"|torch\.get_num_"
    r"|torch\.set_warn_always$"
    r"|torch\.set_printoptions$"
    r"|torch\.set_flush_denormal$"
    r"|tf\.distribute($|\.)"
    r"|tf\.io($|\.)"
    r"|tf\.data($|\.)"
    r"|tf\.summary($|\.)"
    r"|tf\.config($|\.)"
    r"|tf\.train($|\.)"
    r"|tf\.profiler($|\.)"
    r"|tf\.queue($|\.)"
    r"|tf\.feature_column($|\.)"
    r"|tf\.lookup($|\.)"
    r"|tf\.errors($|\.)"
    r"|tf\.debugging($|\.)"
    r"|tf\.autodiff($|\.)"
    r"|tf\.types($|\.)"
    r"|tf\.test($|\.)"
    r"|tf\.raw_ops($|\.)"          # low-level ops; redundant with tf.* aliases
    r"|tf\.strings($|\.)"          # string processing, not numerical
    r"|tf\.audio($|\.)"            # audio I/O, not core numerical
    r"|tf\.image($|\.)"            # image I/O / preprocessing
    r"|tf\.ragged($|\.)"           # ragged-tensor utilities (structural)
    r"|tf\.sparse($|\.)"           # sparse-tensor utilities (structural)
    r"|tf\.compat($|\.)"           # compatibility shims
    r"|tf\.mlir($|\.)"             # MLIR compiler bridge
    r"|tf\.nest($|\.)"             # python nest utilities
    r"|tf\.sets($|\.)"             # set ops on tensors (structural)
    r"|tf\.quantization($|\.)"     # quantization stack
    r"|tf\.keras\.preprocessing"   # data preprocessing
    r"|tf\.keras\.datasets"        # dataset loaders
    r"|tf\.keras\.utils"           # python helpers
    r"|tf\.keras\.callbacks"       # training callbacks
    r"|tf\.assert_"                # assertion helpers
    r")"
)


def is_excluded(api: str) -> Tuple[bool, str]:
    """Return (True, reason) if the API should be excluded from the fuzzing pool."""
    if _RANDOM_OP_REGEX.match(api):
        return True, "random_generation"
    if _PRIVATE_REGEX.match(api):
        return True, "experimental_or_private"
    if _COMPILER_REGEX.match(api):
        return True, "compiler_frontend"
    if _INFRA_REGEX.match(api):
        return True, "infrastructure"
    return False, ""


# --------------------------------------------------------------------- #
# Keyword / prefix rules per group (checked in order; first match wins) #
# --------------------------------------------------------------------- #

_RULES: Dict[str, List[str]] = {
    "gradients_optim": [
        "torch.optim.", "torch.autograd.", "torch.enable_grad",
        "torch.no_grad", "torch.is_grad_enabled", "torch.set_grad_enabled",
        "torch.gradient", "torch.Tensor.grad", "torch.Tensor.backward",
        "torch.Tensor.requires_grad", "torch.Tensor.retain_grad",
        "torch.Tensor.register_hook", "torch.nn.utils.clip_grad",
    ],
    "distributions": [
        "torch.distributions.",
    ],
    "storage_serial": [
        "torch.save", "torch.load", "torch.jit.", "torch.package.",
        "torch.serialization", "torch.onnx", "torch.export",
    ],
    "model_io": [
        "torch.hub.", "torch.fx.", "torch.profiler.", "torch.futures.",
        "torch.overrides.",
    ],
    "random_generation": [
        # NB: most of these never appear in practice because is_excluded()
        # drops them upstream of classification. The bucket is kept so the
        # paper's 11-group taxonomy is preserved structurally.
        "torch.rand", "torch.randn", "torch.randint", "torch.randperm",
        "torch.manual_seed", "torch.seed", "torch.random.",
        "torch.bernoulli", "torch.poisson", "torch.multinomial",
        "torch.normal", "torch.Tensor.random_", "torch.Tensor.bernoulli",
        "torch.Tensor.uniform_", "torch.Tensor.normal_",
        # Windowing functions are deterministic (not random) — moved to
        # mathematics below.
    ],
    "forward_layers": [
        # Catch-all for Keras-style nn.* modules and any nn.functional.* call.
        # Every class in torch.nn that is a layer/loss/activation lives here.
        "torch.nn.functional.",
        "torch.nn.Sequential", "torch.nn.ModuleList", "torch.nn.ModuleDict",
        "torch.nn.ParameterList", "torch.nn.ParameterDict", "torch.nn.Parameter",
        "torch.nn.Identity", "torch.nn.Flatten", "torch.nn.Unflatten",
        # Convolutions
        "torch.nn.Conv", "torch.nn.ConvTranspose",
        "torch.nn.Linear", "torch.nn.Bilinear",
        "torch.nn.LazyLinear", "torch.nn.LazyConv1d", "torch.nn.LazyConv2d",
        "torch.nn.LazyConv3d", "torch.nn.LazyConvTranspose1d",
        "torch.nn.LazyConvTranspose2d", "torch.nn.LazyConvTranspose3d",
        "torch.nn.LazyBilinear",
        "torch.nn.LazyBatchNorm", "torch.nn.LazyInstanceNorm",
        "torch.nn.RNNBase",
        # Normalisation
        "torch.nn.BatchNorm", "torch.nn.LayerNorm", "torch.nn.GroupNorm",
        "torch.nn.InstanceNorm", "torch.nn.LocalResponseNorm",
        "torch.nn.CrossMapLRN2d", "torch.nn.RMSNorm",
        # Recurrent
        "torch.nn.RNN", "torch.nn.LSTM", "torch.nn.GRU",
        # Attention / Transformer
        "torch.nn.Transformer", "torch.nn.TransformerEncoder",
        "torch.nn.TransformerEncoderLayer",
        "torch.nn.TransformerDecoder", "torch.nn.TransformerDecoderLayer",
        "torch.nn.MultiheadAttention",
        # Embedding
        "torch.nn.Embedding", "torch.nn.EmbeddingBag",
        # Activations
        "torch.nn.ReLU", "torch.nn.ReLU6", "torch.nn.RReLU",
        "torch.nn.LeakyReLU", "torch.nn.PReLU", "torch.nn.ELU",
        "torch.nn.CELU", "torch.nn.SELU", "torch.nn.GELU",
        "torch.nn.Sigmoid", "torch.nn.LogSigmoid",
        "torch.nn.Tanh", "torch.nn.Tanhshrink", "torch.nn.Hardtanh",
        "torch.nn.Hardshrink", "torch.nn.Hardswish", "torch.nn.Hardsigmoid",
        "torch.nn.SiLU", "torch.nn.Mish", "torch.nn.Softplus",
        "torch.nn.Softshrink", "torch.nn.Softsign",
        "torch.nn.Softmax", "torch.nn.Softmax2d", "torch.nn.LogSoftmax",
        "torch.nn.Softmin", "torch.nn.AdaptiveLogSoftmaxWithLoss",
        "torch.nn.GLU", "torch.nn.Threshold",
        # Dropout
        "torch.nn.Dropout", "torch.nn.AlphaDropout",
        "torch.nn.FeatureAlphaDropout",
        # Pooling (includes Adaptive, Fractional, LP variants)
        "torch.nn.MaxPool", "torch.nn.AvgPool",
        "torch.nn.AdaptiveMaxPool", "torch.nn.AdaptiveAvgPool",
        "torch.nn.MaxUnpool", "torch.nn.FractionalMaxPool",
        "torch.nn.LPPool",
        # Padding
        "torch.nn.ReflectionPad", "torch.nn.ReplicationPad",
        "torch.nn.ZeroPad", "torch.nn.ConstantPad",
        "torch.nn.CircularPad",
        # Upsampling / shuffle
        "torch.nn.Upsample", "torch.nn.UpsamplingNearest",
        "torch.nn.UpsamplingBilinear",
        "torch.nn.PixelShuffle", "torch.nn.PixelUnshuffle",
        "torch.nn.ChannelShuffle",
        "torch.nn.Unfold", "torch.nn.Fold",
        # Losses (BCE, CTC, Huber, Hinge, KL, L1, MSE, NLL, Multi*, Poisson, ...)
        "torch.nn.BCELoss", "torch.nn.BCEWithLogitsLoss",
        "torch.nn.CrossEntropyLoss", "torch.nn.CTCLoss",
        "torch.nn.L1Loss", "torch.nn.MSELoss", "torch.nn.NLLLoss",
        "torch.nn.PoissonNLLLoss", "torch.nn.GaussianNLLLoss",
        "torch.nn.HuberLoss", "torch.nn.SmoothL1Loss",
        "torch.nn.HingeEmbeddingLoss", "torch.nn.KLDivLoss",
        "torch.nn.MarginRankingLoss", "torch.nn.TripletMarginLoss",
        "torch.nn.TripletMarginWithDistanceLoss",
        "torch.nn.CosineEmbeddingLoss",
        "torch.nn.MultiLabelMarginLoss", "torch.nn.MultiLabelSoftMarginLoss",
        "torch.nn.MultiMarginLoss", "torch.nn.SoftMarginLoss",
        # Cell variants (GRU / LSTM / RNN)
        "torch.nn.RNNCell", "torch.nn.GRUCell", "torch.nn.LSTMCell",
        # Utilities that are still layer-like
        "torch.nn.ZeroPad", "torch.nn.utils.rnn.",
        "torch.nn.utils.spectral_norm", "torch.nn.utils.weight_norm",
        "torch.nn.utils.parametrize", "torch.nn.utils.parametrizations.",
        "torch.nn.utils.remove_spectral_norm",
        "torch.nn.utils.remove_weight_norm",
        "torch.nn.utils.prune", "torch.nn.utils.parameters_to_vector",
        "torch.nn.utils.vector_to_parameters", "torch.nn.utils.skip_init",
        # Weight-init helpers
        "torch.nn.init.",
        # Layer-adjacent utilities
        "torch.nn.CosineSimilarity", "torch.nn.PairwiseDistance",
        "torch.nn.SyncBatchNorm", "torch.nn.modules.",
        "torch.nn.parameter.",
    ],
    "logical": [
        "torch.eq", "torch.ne", "torch.lt", "torch.le", "torch.gt",
        "torch.ge", "torch.equal", "torch.greater", "torch.greater_equal",
        "torch.less", "torch.less_equal", "torch.not_equal",
        "torch.logical_", "torch.logical.",
        "torch.any", "torch.all", "torch.where", "torch.masked_",
        "torch.isnan", "torch.isinf", "torch.isfinite", "torch.isreal",
        "torch.isneginf", "torch.isposinf", "torch.isin",
        "torch.isclose", "torch.allclose",
        "torch.bitwise_", "torch.Tensor.bitwise_",
        "torch.Tensor.logical_",
        "torch.masked_select",
        "torch.Tensor.eq", "torch.Tensor.ne", "torch.Tensor.lt",
        "torch.Tensor.le", "torch.Tensor.gt", "torch.Tensor.ge",
        "torch.Tensor.any", "torch.Tensor.all", "torch.Tensor.bool",
        "torch.Tensor.masked",
    ],
    "reshaping": [
        "torch.reshape", "torch.view", "torch.squeeze", "torch.unsqueeze",
        "torch.permute", "torch.transpose", "torch.swapaxes",
        "torch.swapdims", "torch.movedim", "torch.moveaxis",
        "torch.flatten", "torch.unflatten", "torch.ravel",
        "torch.cat", "torch.concat", "torch.stack", "torch.hstack",
        "torch.vstack", "torch.dstack", "torch.column_stack",
        "torch.hsplit", "torch.vsplit", "torch.dsplit",
        "torch.row_stack", "torch.nonzero",
        "torch.unique", "torch.unique_consecutive",
        "torch.chunk", "torch.split", "torch.unbind", "torch.tensor_split",
        "torch.Tensor.tensor_split",
        "torch.narrow", "torch.select", "torch.index_select",
        "torch.gather", "torch.scatter", "torch.scatter_",
        "torch.roll", "torch.rot90", "torch.flip", "torch.fliplr",
        "torch.flipud", "torch.tile", "torch.repeat_interleave",
        "torch.broadcast_to", "torch.broadcast_shapes",
        "torch.Tensor.reshape", "torch.Tensor.view", "torch.Tensor.expand",
        "torch.Tensor.squeeze", "torch.Tensor.unsqueeze",
        "torch.Tensor.permute", "torch.Tensor.transpose",
        "torch.Tensor.flatten", "torch.Tensor.unflatten",
        "torch.Tensor.contiguous", "torch.Tensor.stride",
        "torch.Tensor.storage", "torch.Tensor.as_strided",
        "torch.as_strided",
    ],
    "creation_conversion": [
        "torch.zeros", "torch.ones", "torch.empty", "torch.full",
        "torch.eye", "torch.arange", "torch.linspace", "torch.logspace",
        "torch.tensor", "torch.as_tensor", "torch.from_numpy",
        "torch.from_dlpack", "torch.frombuffer",
        "torch.can_cast", "torch.clone", "torch.dequantize",
        "torch.is_complex", "torch.is_conj", "torch.is_floating_point",
        "torch.is_nonzero", "torch.is_signed", "torch.is_storage",
        "torch.is_tensor", "torch.is_inference_mode_enabled",
        "torch.is_warn_always_enabled", "torch.inference_mode",
        "torch.fake_quantize_per_tensor_affine",
        "torch.fake_quantize_per_channel_affine",
        "torch.quantize_per_tensor", "torch.quantize_per_channel",
        "torch.quantized_batch_norm", "torch.quantized_max_pool1d",
        "torch.quantized_max_pool2d",
        "torch.numel", "torch.range",
        "torch.promote_types", "torch.result_type",
        "torch.sparse_coo_tensor", "torch.sparse_csr_tensor",
        "torch.sparse_csc_tensor", "torch.sparse_bsr_tensor",
        "torch.sparse_bsc_tensor", "torch.vander",
        "torch.Tensor.to", "torch.Tensor.type", "torch.Tensor.float",
        "torch.Tensor.double", "torch.Tensor.half", "torch.Tensor.int",
        "torch.Tensor.long", "torch.Tensor.short", "torch.Tensor.bool",
        "torch.Tensor.byte", "torch.Tensor.char", "torch.Tensor.bfloat16",
        "torch.Tensor.cpu", "torch.Tensor.cuda", "torch.Tensor.clone",
        "torch.Tensor.detach", "torch.Tensor.numpy", "torch.Tensor.item",
        "torch.Tensor.tolist",
        "torch.zeros_like", "torch.ones_like", "torch.empty_like",
        "torch.full_like", "torch.rand_like", "torch.randn_like",
        "torch.randint_like",
        # Storage types (deprecated but in API list)
        "Storage", "torch.Generator",
    ],
    "mathematics": [
        # Catch the broad math namespace – checked after specific groups
        "torch.add", "torch.sub", "torch.mul", "torch.div",
        "torch.matmul", "torch.mm", "torch.bmm", "torch.mv",
        "torch.dot", "torch.vdot", "torch.inner", "torch.outer",
        "torch.cross", "torch.kron",
        "torch.abs", "torch.absolute", "torch.neg", "torch.positive",
        "torch.reciprocal", "torch.sign", "torch.sgn", "torch.signbit",
        "torch.sqrt", "torch.rsqrt", "torch.square", "torch.pow",
        "torch.exp", "torch.exp2", "torch.expm1", "torch.log",
        "torch.log2", "torch.log10", "torch.log1p", "torch.logaddexp",
        "torch.logaddexp2", "torch.ceil", "torch.floor", "torch.round",
        "torch.trunc", "torch.frac", "torch.fmod", "torch.remainder",
        "torch.clamp", "torch.clip",
        "torch.sin", "torch.cos", "torch.tan", "torch.asin", "torch.acos",
        "torch.atan", "torch.atan2", "torch.sinh", "torch.cosh",
        "torch.tanh", "torch.asinh", "torch.acosh", "torch.atanh",
        "torch.arcsin", "torch.arccos", "torch.arctan", "torch.arctan2",
        "torch.arcsinh", "torch.arccosh", "torch.arctanh",
        "torch.hypot", "torch.deg2rad", "torch.rad2deg",
        "torch.sum", "torch.prod", "torch.mean", "torch.std", "torch.var",
        "torch.max", "torch.min", "torch.amax", "torch.amin",
        "torch.argmax", "torch.argmin", "torch.cumsum", "torch.cumprod",
        "torch.cummax", "torch.cummin", "torch.diff",
        "torch.sort", "torch.argsort", "torch.topk", "torch.kthvalue",
        "torch.median", "torch.nanmedian", "torch.nansum", "torch.nanmean",
        "torch.linalg.", "torch.fft.", "torch.special.",
        # Tensor method equivalents
        "torch.Tensor.abs", "torch.Tensor.add", "torch.Tensor.sub",
        "torch.Tensor.mul", "torch.Tensor.div", "torch.Tensor.pow",
        "torch.Tensor.sqrt", "torch.Tensor.log", "torch.Tensor.exp",
        "torch.Tensor.sin", "torch.Tensor.cos", "torch.Tensor.tan",
        "torch.Tensor.asin", "torch.Tensor.acos", "torch.Tensor.atan",
        "torch.Tensor.sum", "torch.Tensor.mean", "torch.Tensor.std",
        "torch.Tensor.var", "torch.Tensor.max", "torch.Tensor.min",
        "torch.Tensor.sort", "torch.Tensor.topk", "torch.Tensor.clamp",
        "torch.Tensor.matmul", "torch.Tensor.mm", "torch.Tensor.bmm",
        "torch.Tensor.dot", "torch.Tensor.cross", "torch.Tensor.cumsum",
        "torch.Tensor.cumprod", "torch.Tensor.neg", "torch.Tensor.reciprocal",
        "torch.Tensor.ceil", "torch.Tensor.floor", "torch.Tensor.round",
        "torch.Tensor.tanh", "torch.Tensor.sigmoid",
        "torch.addmm", "torch.addmv", "torch.addbmm", "torch.addr",
        "torch.addcdiv", "torch.addcmul", "torch.baddbmm",
        "torch.chain_matmul", "torch.einsum",
        "torch.angle", "torch.imag", "torch.real", "torch.view_as_real",
        "torch.view_as_complex", "torch.polar", "torch.conj",
        "torch.bincount", "torch.histc", "torch.histogram",
        "torch.norm", "torch.dist", "torch.trace",
        # Signal-processing windows (deterministic, not random)
        "torch.bartlett_window", "torch.blackman_window",
        "torch.hamming_window", "torch.hann_window", "torch.kaiser_window",
        # Linear algebra and statistics on tensors
        "torch.cholesky", "torch.cholesky_inverse", "torch.cholesky_solve",
        "torch.det", "torch.slogdet", "torch.logdet",
        "torch.qr", "torch.svd", "torch.symeig", "torch.lu",
        "torch.lu_solve", "torch.lu_unpack", "torch.triangular_solve",
        "torch.geqrf", "torch.orgqr", "torch.ormqr", "torch.lobpcg",
        "torch.matrix_power", "torch.matrix_rank", "torch.matrix_exp",
        "torch.pinverse", "torch.inverse", "torch.solve", "torch.lstsq",
        "torch.eig", "torch.diag", "torch.diag_embed", "torch.diagflat",
        "torch.diagonal", "torch.tril", "torch.triu",
        "torch.tril_indices", "torch.triu_indices",
        "torch.corrcoef", "torch.cov", "torch.count_nonzero",
        "torch.copysign", "torch.heaviside", "torch.lerp", "torch.lgamma",
        "torch.mvlgamma", "torch.digamma", "torch.polygamma",
        "torch.erf", "torch.erfc", "torch.erfinv", "torch.i0", "torch.sinc",
        "torch.softmax", "torch.log_softmax",
        "torch.true_divide", "torch.floor_divide", "torch.float_power",
        "torch.lcm", "torch.gcd", "torch.nan_to_num",
        "torch.amin", "torch.amax", "torch.aminmax", "torch.quantile",
        "torch.nanquantile", "torch.bucketize", "torch.searchsorted",
        "torch.atleast_", "torch.broadcast_tensors",
        "torch.einsum", "torch.tensordot",
        "torch.conj_physical", "torch.complex", "torch.resolve_conj",
        "torch.resolve_neg",
        # More math ops living in the top-level torch namespace
        "torch.block_diag", "torch.cdist", "torch.cartesian_prod",
        "torch.combinations", "torch.meshgrid",
        "torch.fmax", "torch.fmin", "torch.fix", "torch.frexp",
        "torch.ldexp", "torch.msort", "torch.mode",
        "torch.cumulative_trapezoid", "torch.trapezoid",
        "torch.igamma", "torch.igammac",
        # Tensor methods still living in misc
        "torch.Tensor.solve", "torch.Tensor.sspaddmm",
        "torch.Tensor.tril_", "torch.Tensor.triu_",
        "torch.pca_lowrank", "torch.sigmoid", "torch.renorm",
        "torch.sparse.", "torch.nanmean", "torch.nanmedian",
        "torch.nanquantile", "torch.nansum",
        "torch.stft", "torch.xlogy",
        # Missing atomic math ops
        "torch.divide", "torch.multiply", "torch.subtract",
        "torch.negative", "torch.maximum", "torch.minimum",
        "torch.ger", "torch.take", "torch.trapz",
        "torch.logsumexp", "torch.logcumsumexp", "torch.logit",
        "torch.nextafter", "torch.t",
    ],
}

# fallback
_MISC = "misc"

# Method-name suffixes used to reclassify torch.Tensor.* that fall through
_TENSOR_MATH_SUFFIXES = {
    "abs", "absolute", "acos", "acosh", "add", "addbmm", "addcdiv",
    "addcmul", "addmm", "addmv", "addr", "angle", "arccos", "arccosh",
    "arcsin", "arcsinh", "arctan", "arctanh", "asin", "asinh", "atan",
    "atanh", "baddbmm", "bincount", "ceil", "clamp", "clip", "conj",
    "conj_physical", "copysign", "corrcoef", "cos", "cosh", "count_nonzero",
    "cov", "cross", "cummax", "cummin", "cumprod", "cumsum",
    "deg2rad", "det", "diag", "diag_embed", "diagflat", "digamma",
    "diff", "dist", "div", "divide", "dot", "dsplit",
    "eig", "einsum", "erf", "erfc", "erfinv", "exp", "exp2",
    "expm1", "fix", "float_power", "floor", "floor_divide",
    "fmax", "fmin", "fmod", "frac", "frexp",
    "gcd", "geqrf", "ger", "heaviside", "histc", "histogram",
    "hypot", "i0", "igamma", "igammac", "imag", "inner", "inverse",
    "isfinite", "isinf", "isnan", "isneginf", "isposinf", "isreal",
    "kron", "kthvalue", "lcm", "ldexp", "lerp", "lgamma",
    "log", "log10", "log1p", "log2", "logaddexp", "logaddexp2",
    "logcumsumexp", "logdet", "logit", "logsumexp",
    "lstsq", "lu", "lu_solve",
    "matmul", "matrix_exp", "matrix_power", "max", "maximum", "mean",
    "median", "min", "minimum", "mm", "mode", "msort",
    "mul", "multiply", "mv", "mvlgamma", "nan_to_num",
    "nanmean", "nanmedian", "nanquantile", "nansum", "narrow_copy",
    "neg", "negative", "nextafter", "norm", "outer",
    "pinverse", "polygamma", "positive", "pow", "prod", "qr",
    "quantile", "rad2deg", "real", "reciprocal", "remainder",
    "renorm", "round", "rsqrt", "sgn", "sigmoid", "sign", "signbit",
    "sin", "sinc", "sinh", "slogdet", "softmax", "sort", "special",
    "sqrt", "square", "std", "std_mean", "sub", "subtract", "sum",
    "svd", "symeig", "t", "tan", "tanh", "topk", "trace",
    "trapz", "triangular_solve", "true_divide", "trunc",
    "var", "var_mean", "vdot", "xlogy",
    "amax", "amin", "aminmax", "argsort", "argmax", "argmin",
    "all_reduce", "any",
    "cholesky", "cholesky_inverse", "cholesky_solve",
    "hardshrink", "istft", "stft",
    "orgqr", "ormqr", "geqrf", "lobpcg",
    "resolve_conj", "resolve_neg",
    "isclose", "allclose",
}
_TENSOR_MATH_SUFFIXES |= {s + "_" for s in _TENSOR_MATH_SUFFIXES}

_TENSOR_RESHAPE_SUFFIXES = {
    "as_strided", "broadcast_to", "chunk", "coalesce", "col2im",
    "contiguous", "diagonal", "diagonal_scatter", "expand", "expand_as",
    "flatten", "flip", "fliplr", "flipud", "gather", "hsplit", "hstack",
    "im2col", "index_add", "index_copy", "index_fill", "index_put",
    "index_reduce", "index_select", "movedim", "moveaxis", "narrow",
    "nonzero", "permute", "put", "ravel", "repeat", "repeat_interleave",
    "reshape", "reshape_as", "resize_", "roll", "rot90",
    "scatter", "scatter_add", "scatter_reduce", "select",
    "select_scatter", "slice_scatter", "split", "split_with_sizes",
    "squeeze", "storage", "stride",
    "sum_to_size", "swapaxes", "swapdims", "take", "take_along_dim",
    "tile", "transpose", "tril", "triu", "unbind", "unflatten",
    "unfold", "unique", "unique_consecutive", "unsqueeze", "vsplit",
    "vstack", "view", "view_as", "where", "as_strided_scatter",
    "fill_diagonal_", "index_add_", "index_copy_", "index_fill_",
    "index_put_", "put_", "scatter_", "scatter_add_", "scatter_reduce_",
    "set_", "resize_as_",
}

_TENSOR_CREATION_SUFFIXES = {
    "bfloat16", "bool", "byte", "cdouble", "cfloat", "char", "clone",
    "copy_", "cpu", "cuda", "data_ptr", "dense_dim", "dequantize",
    "detach", "detach_", "device", "dim", "double",
    "element_size", "fill_", "float", "get_device", "grad_layout",
    "half", "int", "is_complex", "is_conj", "is_contiguous",
    "is_cuda", "is_floating_point", "is_inference", "is_leaf",
    "is_meta", "is_neg", "is_nonzero", "is_pinned", "is_quantized",
    "is_set_to", "is_shared", "is_signed", "is_sparse", "is_sparse_csr",
    "item", "layout", "long", "nbytes", "ndim", "ndimension",
    "nelement", "new", "new_empty", "new_empty_strided", "new_full",
    "new_ones", "new_tensor", "new_zeros", "numel", "numpy",
    "output_nr", "pin_memory", "q_per_channel_axis",
    "q_per_channel_scales", "q_per_channel_zero_points",
    "q_scale", "q_zero_point", "qscheme", "real_t",
    "record_stream", "refine_names", "rename", "rename_",
    "requires_grad_", "share_memory_", "short", "size",
    "smm", "sparse_dim", "sparse_mask", "sparse_resize_",
    "sparse_resize_and_clear_", "stft", "storage_offset",
    "storage_type", "to", "to_dense", "to_mkldnn", "to_padded_tensor",
    "to_sparse", "to_sparse_csr", "tolist",
    "type", "type_as", "untyped_storage", "values", "indices",
    "zero_",
}

_TENSOR_LOGICAL_SUFFIXES = {
    "all", "any", "eq", "equal", "ge", "gt", "le", "lt", "ne",
    "greater", "greater_", "greater_equal", "greater_equal_",
    "less", "less_", "less_equal", "less_equal_",
    "not_equal", "not_equal_",
    "masked_fill", "masked_fill_", "masked_scatter", "masked_scatter_",
    "masked_select",
}

_TENSOR_GRAD_SUFFIXES = {
    "backward", "grad", "grad_fn", "register_hook",
    "retain_grad", "requires_grad", "retains_grad",
}


def _classify_tensor_method(api: str) -> str:
    """Classify torch.Tensor.<method> by method name."""
    method = api.split(".")[-1]
    if method in _TENSOR_MATH_SUFFIXES:
        return "mathematics"
    if method in _TENSOR_RESHAPE_SUFFIXES:
        return "reshaping"
    if method in _TENSOR_CREATION_SUFFIXES:
        return "creation_conversion"
    if method in _TENSOR_LOGICAL_SUFFIXES:
        return "logical"
    if method in _TENSOR_GRAD_SUFFIXES:
        return "gradients_optim"
    return _MISC


def _matches(api: str, kw: str) -> bool:
    """Match `api` against keyword `kw`.

    Rules (in order):
      • Exact match: ``api == kw``                              → True
      • ``kw`` ends with '.': module prefix match
          e.g. ``torch.fft.`` matches ``torch.fft.fft``
      • ``kw`` ends with '_': snake-case family prefix match
          e.g. ``torch.logical_`` matches
          ``torch.logical_and`` / ``torch.logical_or`` / ``torch.logical_not``.
      • Otherwise: require a word boundary after ``kw`` — the next character
        must be non-alphabetic (digit, punctuation, underscore, or end).
        ``torch.nn.Conv``    matches ``torch.nn.Conv2d`` (next '2' is digit) ✓
        ``torch.Tensor.abs`` matches ``torch.Tensor.abs_`` (next '_') ✓
        ``torch.t``          does NOT match ``torch.tensor`` (next 'e' alpha) ✗
    """
    if api == kw:
        return True
    if kw.endswith(".") or kw.endswith("_"):
        return api.startswith(kw)
    if not api.startswith(kw):
        return False
    next_char = api[len(kw) : len(kw) + 1]
    return next_char == "" or not next_char.isalpha()


def _classify_api(api: str) -> str:
    for group, keywords in _RULES.items():
        for kw in keywords:
            if _matches(api, kw):
                return group
    # Fallback: torch.Tensor.* method name matching
    if api.startswith("torch.Tensor."):
        return _classify_tensor_method(api)
    # Storage types → creation/conversion
    if api.endswith("Storage"):
        return "creation_conversion"
    return _MISC


# --------------------------------------------------------------------- #
# TensorFlow rule set                                                   #
# --------------------------------------------------------------------- #

_TF_RULES: Dict[str, List[str]] = {
    "gradients_optim": [
        "tf.keras.optimizers.", "tf.optimizers.", "tf.GradientTape",
        "tf.gradients", "tf.stop_gradient", "tf.custom_gradient",
        "tf.recompute_grad", "tf.hessians",
    ],
    "distributions": [
        "tf.distributions.", "tfp.distributions.",
        # TF2 native sampling ops that represent parametric distributions
        "tf.random.categorical", "tf.random.gamma", "tf.random.poisson",
        "tf.random.stateless_binomial", "tf.random.stateless_gamma",
        "tf.random.stateless_poisson",
        "tf.raw_ops.RandomGamma", "tf.raw_ops.RandomPoissonV2",
        "tf.raw_ops.StatelessRandomGammaV2",
    ],
    "random_generation": [
        "tf.random.", "tf.random_normal_initializer",
        "tf.random_uniform_initializer",
        "tf.raw_ops.RandomStandardNormal", "tf.raw_ops.RandomUniform",
        "tf.raw_ops.RandomUniformInt", "tf.raw_ops.RandomShuffle",
        "tf.raw_ops.RandomShuffleQueueV2",
        "tf.raw_ops.StatelessRandomNormal", "tf.raw_ops.StatelessRandomUniform",
        "tf.raw_ops.StatelessRandomUniformFullInt",
        "tf.raw_ops.StatelessRandomUniformInt",
        "tf.raw_ops.StatelessRandomGetAlg",
        "tf.experimental.numpy.random.",
    ],
    "storage_serial": [
        "tf.io.serialize_tensor", "tf.io.parse_tensor",
        "tf.train.Checkpoint", "tf.keras.models.save",
        "tf.keras.models.load",
    ],
    "model_io": [
        "tf.keras.Model", "tf.keras.Sequential", "tf.keras.Input",
        "tf.keras.callbacks.", "tf.keras.utils.",
        "tf.keras.applications.",
        "tf.keras.models.", "tf.keras.estimator.",
        "tf.keras.experimental.",
    ],
    "forward_layers": [
        "tf.keras.layers.", "tf.keras.activations.",
        "tf.keras.losses.", "tf.keras.metrics.",
        "tf.keras.regularizers.", "tf.keras.constraints.",
        "tf.keras.initializers.",
        "tf.keras.backend.",
        "tf.losses.", "tf.metrics.", "tf.initializers.", "tf.nn.",
        "tf.nn.conv", "tf.nn.depthwise_conv", "tf.nn.separable_conv",
        "tf.nn.atrous_conv", "tf.nn.embedding_lookup",
        "tf.nn.relu", "tf.nn.relu6", "tf.nn.leaky_relu", "tf.nn.elu",
        "tf.nn.selu", "tf.nn.gelu", "tf.nn.crelu", "tf.nn.silu",
        "tf.nn.swish", "tf.nn.sigmoid", "tf.nn.tanh", "tf.nn.softmax",
        "tf.nn.softplus", "tf.nn.softsign", "tf.nn.log_softmax",
        "tf.nn.dropout", "tf.nn.l2_normalize", "tf.nn.local_response_normalization",
        "tf.nn.batch_normalization", "tf.nn.fused_batch_norm",
        "tf.nn.moments", "tf.nn.weighted_moments",
        "tf.nn.avg_pool", "tf.nn.max_pool", "tf.nn.fractional",
        "tf.nn.pool", "tf.nn.bias_add",
        "tf.nn.ctc_", "tf.nn.rnn_cell", "tf.nn.dynamic_rnn",
        "tf.nn.static_rnn", "tf.nn.bidirectional",
        "tf.nn.sampled_softmax", "tf.nn.nce_loss",
        "tf.nn.sigmoid_cross_entropy_with_logits",
        "tf.nn.softmax_cross_entropy_with_logits",
        "tf.nn.sparse_softmax_cross_entropy_with_logits",
        "tf.nn.weighted_cross_entropy_with_logits",
    ],
    "logical": [
        "tf.equal", "tf.not_equal", "tf.less", "tf.less_equal",
        "tf.greater", "tf.greater_equal", "tf.logical_and",
        "tf.logical_or", "tf.logical_not", "tf.logical_xor",
        "tf.where", "tf.cond", "tf.case", "tf.boolean_mask",
        "tf.math.is_nan", "tf.math.is_inf", "tf.math.is_finite",
        "tf.reduce_all", "tf.reduce_any",
    ],
    "reshaping": [
        "tf.reshape", "tf.transpose", "tf.expand_dims", "tf.squeeze",
        "tf.concat", "tf.stack", "tf.unstack", "tf.split", "tf.tile",
        "tf.gather", "tf.gather_nd", "tf.scatter_nd",
        "tf.tensor_scatter_nd_add", "tf.tensor_scatter_nd_max",
        "tf.tensor_scatter_nd_min", "tf.tensor_scatter_nd_sub",
        "tf.tensor_scatter_nd_update",
        "tf.sequence_mask", "tf.unravel_index",
        "tf.slice",
        "tf.strided_slice", "tf.pad", "tf.repeat", "tf.reverse",
        "tf.roll", "tf.broadcast_to", "tf.fill", "tf.unique",
        "tf.unique_with_counts", "tf.dynamic_partition",
        "tf.dynamic_stitch", "tf.identity", "tf.identity_n",
        "tf.parallel_stack", "tf.broadcast_dynamic_shape",
        "tf.broadcast_static_shape", "tf.size", "tf.shape",
        "tf.shape_n", "tf.rank", "tf.one_hot", "tf.searchsorted",
        "tf.space_to_batch", "tf.batch_to_space",
        "tf.depth_to_space", "tf.space_to_depth",
    ],
    "creation_conversion": [
        "tf.zeros", "tf.ones", "tf.constant", "tf.Variable",
        "tf.zeros_like", "tf.ones_like", "tf.eye", "tf.linspace",
        "tf.range", "tf.cast", "tf.bitcast", "tf.convert_to_tensor",
        "tf.dtypes.", "tf.fill",
        "tf.complex", "tf.real", "tf.imag",
        "tf.numpy_function", "tf.py_function",
        "tf.as_dtype", "tf.is_tensor", "tf.ensure_shape",
        "tf.get_static_value", "tf.IndexedSlices", "tf.IndexedSlicesSpec",
        "tf.TensorSpec", "tf.TensorArraySpec", "tf.RaggedTensor",
        "tf.RaggedTensorSpec", "tf.SparseTensor", "tf.SparseTensorSpec",
        "tf.Tensor", "tf.Module", "tf.name_scope",
        "tf.type_spec_from_value", "tf.TensorArray",
    ],
    "mathematics": [
        "tf.add", "tf.subtract", "tf.multiply", "tf.divide", "tf.mod",
        "tf.floormod", "tf.floordiv", "tf.realdiv", "tf.truediv",
        "tf.matmul", "tf.einsum", "tf.tensordot",
        "tf.abs", "tf.negative", "tf.sign", "tf.reciprocal",
        "tf.sqrt", "tf.rsqrt", "tf.square", "tf.pow",
        "tf.exp", "tf.expm1", "tf.log", "tf.log1p", "tf.log_sigmoid",
        "tf.sin", "tf.cos", "tf.tan", "tf.asin", "tf.acos", "tf.atan",
        "tf.atan2", "tf.sinh", "tf.cosh", "tf.tanh",
        "tf.asinh", "tf.acosh", "tf.atanh",
        "tf.ceil", "tf.floor", "tf.round", "tf.rint",
        "tf.maximum", "tf.minimum", "tf.argmax", "tf.argmin",
        "tf.reduce_sum", "tf.reduce_mean", "tf.reduce_prod",
        "tf.reduce_max", "tf.reduce_min", "tf.reduce_std",
        "tf.reduce_variance", "tf.reduce_logsumexp",
        "tf.cumsum", "tf.cumprod", "tf.cumulative_logsumexp",
        "tf.norm", "tf.linalg.", "tf.signal.", "tf.math.",
        "tf.bincount", "tf.histogram_fixed_width",
        "tf.sort", "tf.top_k", "tf.nn.top_k",
        "tf.clip_by_value", "tf.clip_by_norm", "tf.clip_by_global_norm",
        "tf.segment_max", "tf.segment_min", "tf.segment_mean",
        "tf.segment_prod", "tf.segment_sum",
        "tf.unsorted_segment_max", "tf.unsorted_segment_min",
        "tf.unsorted_segment_mean", "tf.unsorted_segment_prod",
        "tf.unsorted_segment_sum", "tf.unsorted_segment_sqrt_n",
        "tf.bitwise.",
        "tf.argsort", "tf.edit_distance", "tf.eigvals",
        "tf.extract_volume_patches", "tf.foldl", "tf.foldr",
        "tf.scan", "tf.fingerprint",
        "tf.matrix_square_root", "tf.meshgrid", "tf.sigmoid",
        "tf.truncatediv", "tf.truncatemod", "tf.scalar_mul",
        "tf.saturate_cast",
    ],
}


def _classify_tf_api(api: str) -> str:
    for group, keywords in _TF_RULES.items():
        for kw in keywords:
            if _matches(api, kw):
                return group
    return _MISC


# --------------------------------------------------------------------- #
# Public loader                                                         #
# --------------------------------------------------------------------- #


_GROUP_ORDER = list(_RULES.keys()) + [_MISC]
# Random generation lives only in the excluded bucket — no in-pool group.
# The 11 paper groups are: the 10 above + "random_generation" (excluded pool).


def _classify(api: str, framework: str) -> str:
    if framework == "torch":
        return _classify_api(api)
    return _classify_tf_api(api)


def load_and_classify(
    api_file: str | Path,
    *,
    framework: str | None = None,
    drop_excluded: bool = True,
) -> Dict[str, List[str]]:
    """Return ``{group_name: [api, ...]}``.

    Parameters
    ----------
    api_file
        Path to a newline-delimited list of API strings.
    framework
        ``"torch"`` or ``"tf"``. If ``None``, inferred from the filename.
    drop_excluded
        When True (default), excluded APIs (random ops, compiler front-ends,
        infrastructure, private namespaces) are placed in the ``"_excluded"``
        bucket and NOT in any selectable group. When False, they are
        classified anyway — useful for diagnostics only.
    """
    path = Path(api_file)
    if framework is None:
        name = path.name.lower()
        framework = "tf" if name.startswith("tf") else "torch"

    apis = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    groups: Dict[str, List[str]] = {g: [] for g in _GROUP_ORDER}
    groups["_excluded"] = []
    for api in apis:
        excluded, _reason = is_excluded(api)
        if excluded and drop_excluded:
            groups["_excluded"].append(api)
            continue
        groups[_classify(api, framework)].append(api)

    return groups


def group_summary(groups: Dict[str, List[str]]) -> str:
    lines = []
    selectable = 0
    excluded = 0
    for g, apis in groups.items():
        lines.append(f"  {g:25s}: {len(apis):4d}")
        if g == "_excluded":
            excluded += len(apis)
        else:
            selectable += len(apis)
    lines.append(f"  {'SELECTABLE':25s}: {selectable:4d}")
    lines.append(f"  {'EXCLUDED':25s}: {excluded:4d}")
    lines.append(f"  {'TOTAL':25s}: {selectable + excluded:4d}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    files = sys.argv[1:] or ["torch_valid_apis.txt", "tf_valid_apis.txt"]
    for f in files:
        if not Path(f).exists():
            continue
        print(f"=== {f} ===")
        print(group_summary(load_and_classify(f)))
        print()
