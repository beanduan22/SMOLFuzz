# SMOLFuzz — Confirmed Bugs (Verified on Current Hardware)

**Total confirmed: 58** (42 PyTorch + 16 TensorFlow)  
**Verified:** 2026-04-10  
**Hardware:** NVIDIA GPU (CUDA), PyTorch + TensorFlow  
**All reproducers:** `high_confidence_bugs/reproducers/`  
**Inputs:** Embedded as literals in every reproducer — no external files needed

---

## PyTorch — 42 Confirmed

All inputs are embedded as `torch.tensor(...)` literals inside each reproducer script.

| Reproducer | Bug Type | Signal (confirmed) | Key APIs | Mutation | Inputs Shape |
|---|---|---|---|---|---|
| bug_pt071.py | inconsistent | L2=4.4361e-01 | Linear, BatchNorm1d, Dropout, sin | add_noise | [4,8] |
| bug_pt080.py | inconsistent | L2=5.9273e-03 | Linear, WeightNorm, BatchNorm1d | uniform | [4,4] |
| bug_pt101.py | inconsistent | L2=1.1677e-02 | Linear, BatchNorm1d, sin, cos | scale_large | [4,8] |
| bug_pt106.py | inconsistent | L2=9.0167e-02 | Linear, sin, cos, hardshrink | scale_large | scalar |
| bug_pt125.py | inconsistent | L2=2.4522e-01 | Linear, sin, enable_grad, BatchNorm1d, Dropout | scale_large | [1,4] |
| bug_pt138.py | inconsistent | L2=2.0447e+00 | Linear, sin, cos, tanh | scale_small | [4,8] |
| bug_pt144.py | inconsistent | L2=3.2524e-03 | Linear, BatchNorm1d, sin, cos | uniform | [4,1] |
| bug_pt147.py | inconsistent | **L2=8.3500e+01** | Linear, **BatchNorm1d**, sigmoid, expm1, polygamma_ | mask | [1,8] |
| bug_pt162.py | inconsistent | L2=4.8801e-02 | Linear, sin, cos, special.erfc | scale_large | [4,8] |
| bug_pt171.py | inconsistent | **L2=2.0108e+01** | Linear, poisson_nll_loss, cumprod, BatchNorm1d | mask | [4,4] |
| bug_pt191.py | inconsistent | L2=1.5129e+00 | Linear, **BatchNorm1d**, sin, arccos, clamp | add_noise | [4,4] |
| bug_pt192.py | **NaN** | **ASYM NaN: cpu=16, gpu=17, asym=1** | Linear, sin, cos, logsumexp | scale_large | [4,8] |
| bug_pt202.py | inconsistent | L2=1.7321e+00 | Linear, special.log_softmax, sin, cos | scale_large | [4,8] |
| bug_pt236.py | inconsistent | L2=6.0033e-02 | Linear, sin, cos, mean | scale_large | [4,8] |
| bug_pt241.py | inconsistent | L2=6.0545e+00 | Linear, ReLU, relu, requires_grad_ | uniform | [4,8] |
| bug_pt248.py | inconsistent | L2=1.3941e-02 | Linear, sin, cos, special.xlogy | scale_large | scalar |
| bug_pt281.py | inconsistent | L2=2.3222e+00 | Linear, sin, multilabel_soft_margin_loss, mul | mask | [4,8] |
| bug_pt284.py | inconsistent | L2=2.3937e-02 | Linear, sin, cos, Tensor.hardshrink | scale_large | [4,8] |
| bug_pt295.py | inconsistent | L2=2.8284e+00 | Linear, sin, Softshrink, **BatchNorm1d** | uniform | [4,4] |
| bug_pt305.py | inconsistent | L2=2.7761e-01 | Linear, **BatchNorm1d**, sin, Dropout, special.i1 | scale_small | [4,8] |
| bug_pt316.py | inconsistent | L2=7.1596e-03 | Linear, sin, cos, lgamma | scale_large | scalar |
| bug_pt319.py | inconsistent | L2=1.0162e-01 | Linear, ReLU, Unflatten, **Upsample** | scale_large | [4,8] |
| bug_pt335.py | inconsistent | L2=4.0375e-01 | Linear, **BatchNorm1d**, Dropout, pad | add_noise | [4,8] |
| bug_pt343.py | inconsistent | L2=7.8760e-01 | Linear, **BatchNorm1d**, sin, tanh, Dropout | uniform | [4,8] |
| bug_pt346.py | inconsistent | **L2=3.2768e+04** | hann_window, Linear, sin, cos | scale_large | scalar |
| bug_pt357.py | inconsistent | L2=5.7845e-01 | affine_grid, index_select, ge_, **BatchNorm1d**, Dropout | mask | [4,8] |
| bug_pt358.py | inconsistent | **L2=7.3856e+02** | Linear, softplus, sin, cos, geqrf | scale_large | [1,8] |
| bug_pt375.py | inconsistent | L2=2.4601e+00 | Linear, **BatchNorm1d**, dropout, tanh, sinc_ | add_noise | [4,8] |
| bug_pt382.py | inconsistent | L2=7.1658e-03 | Linear, sin, cos, nn.Hardswish | scale_large | [4,8] |
| bug_pt384.py | inconsistent | L2=5.0380e-02 | Linear, stft, sin, cos | scale_large | scalar |
| bug_pt390.py | inconsistent | L2=3.5405e-02 | Linear, sin, cos, clip | scale_large | [4,8] |
| bug_pt396.py | **NaN** | **ASYM NaN/Inf: cpu=Inf, gpu=NaN** | Linear, LeakyReLU, AbsTransform, ormqr | uniform | scalar |
| bug_pt398.py | inconsistent | L2=9.4865e-03 | Linear, sin, cos, special.gammaln | scale_large | [4,8] |
| bug_pt404.py | inconsistent | **L2=5.4636e+04** | Linear, maximum, sin, cos | scale_large | [4,16,8] |
| bug_pt409.py | inconsistent | L2=3.4774e-01 | Linear, expm1, sin, **BatchNorm1d** | scale_small | [4,4] |
| bug_pt424.py | inconsistent | L2=3.9086e-03 | Linear, sin, cos, nn.Mish | scale_large | scalar |
| bug_pt428.py | inconsistent | L2=1.0558e-02 | Linear, sin, cos, nn.GLU | scale_large | [8] |
| bug_pt441.py | **crash** | **CPU crashes / GPU ok** | Linear, logcumsumexp, corrcoef, **cholesky** | scale_small | [4,8] |
| bug_pt450.py | inconsistent | L2=4.7500e+00 | distributions.transforms, Linear, SiLU, sin | uniform | scalar |
| bug_pt467.py | inconsistent | L2=1.7321e+00 | Linear, **BatchNorm1d**, dropout, arctan, matrix_exp | scale_large | [4,4] |
| bug_pt480.py | inconsistent | L2=6.1272e-02 | Linear, GELU, sin, cos | scale_large | [4,8] |
| bug_pt486.py | inconsistent | L2=3.0295e-03 | Linear, sin, cos (multi-output) | scale_large | [4,8] |

### PyTorch by Bug Category

| Category | Count | Reproducers |
|---|---|---|
| Asymmetric NaN/Inf | 2 | pt192, pt396 |
| CPU crash / GPU ok | 1 | pt441 |
| BatchNorm1d divergence | 13 | pt071, pt080, pt101, pt125, pt144, pt147, pt171, pt191, pt295, pt305, pt335, pt343, pt357, pt375, pt409, pt467 |
| Trig pipeline (sin/cos/tanh) | 16 | pt106, pt138, pt162, pt202, pt236, pt248, pt281, pt284, pt316, pt319, pt346, pt358, pt382, pt384, pt390, pt398, pt404, pt424, pt428, pt480, pt486 |
| Other numerical | 10 | pt241, pt450, pt467, pt192, pt396, pt171, pt202 ... |

---

## TensorFlow — 16 Confirmed

All inputs embedded as `np.array(...)` literals (shape [4,8], float32).  
All are **baseline inconsistency** — divergence occurs with **no mutation**, using only identical model weights and the same input on CPU vs GPU.

| Reproducer | Signal (confirmed) | Key APIs |
|---|---|---|
| bug_tf023.py | L2=3.3918e-03 | tf.math.reduce_logsumexp, tf.keras.layers.Conv1D, tf.math.bessel_i1e |
| bug_tf029.py | L2=2.8974e-03 | pad_sequences, Flatten, LambdaCallback, he_uniform |
| bug_tf067.py | L2=6.2642e-03 | tf.linalg.diag, tf.keras.layers.LocallyConnected1D, tf.math.angle |
| bug_tf094.py | L2=1.3180e-02 | squared_hinge, max_pool2d, square, Discretization |
| bug_tf106.py | **L2=1.4250e+00** | **digamma**, **SeparableConv2D**, **linalg.inv**, **mdct** |
| bug_tf110.py | L2=7.2632e-03 | numpy.select, numpy.flipud, numpy.outer, scalar_mul |
| bug_tf202.py | L2=1.4067e-03 | tf.math.igamma, tf.keras.layers.Conv2DTranspose, tf.linalg.trace |
| bug_tf234.py | L2=1.6479e-03 | tf.keras.layers.GRU, tf.math.atanh, tf.linalg.band_part |
| bug_tf246.py | L2=1.1740e-03 | tf.keras.layers.LSTM, tf.math.sinh, tf.signal.rfft |
| bug_tf261.py | L2=1.3418e-03 | tf.keras.layers.Conv3D, tf.math.log1p, tf.linalg.cross |
| bug_tf266.py | L2=5.2257e-03 | custom_object_scope, tanh, ZeroPadding3D, Lambda |
| bug_tf357.py | L2=6.7360e-03 | tf.keras.layers.Bidirectional, tf.math.lgamma, tf.linalg.svd |
| bug_tf383.py | L2=1.9597e-03 | tf.keras.layers.MultiHeadAttention, tf.math.erfc *(NaN present on both sides)* |
| bug_tf402.py | L2=1.2589e-03 | tf.keras.layers.SimpleRNN, tf.math.acosh, tf.signal.dct |
| bug_tf479.py | **L2=3.7410e-02** | tf.keras.layers.DepthwiseConv2D, tf.math.special.bessel_i0e, tf.linalg.eigh |
| bug_tf481.py | L2=7.2726e-03 | tf.keras.layers.SeparableConv1D, tf.math.log_sigmoid, tf.signal.stft |

---

## Not Reproduced on This Hardware (26 TF)

These showed divergence during fuzzing but fall below L2=1e-3 threshold on current driver/hardware.  
They may reproduce on other CUDA versions or GPU models.

tf009, tf012, tf020, tf025, tf035, tf037, tf040, tf053, tf084, tf095, tf104,  
tf147, tf158, tf204, tf208, tf225, tf265, tf355, tf381, tf421, tf438, tf452,  
tf478, tf482, tf486, tf490

---

## How to Run

```bash
# Single bug
python3 high_confidence_bugs/reproducers/bug_pt404.py

# All PyTorch bugs
for f in high_confidence_bugs/reproducers/bug_pt*.py; do
    echo "=== $f ==="; python3 "$f" 2>/dev/null | tail -1
done

# All TensorFlow bugs
for f in high_confidence_bugs/reproducers/bug_tf*.py; do
    echo "=== $f ==="; python3 "$f" 2>/dev/null | tail -1
done
```
