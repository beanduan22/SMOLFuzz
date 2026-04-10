# SMOLFuzz — Confirmed Bugs

**Total: 44 bugs** (42 PyTorch + 2 TensorFlow)  
**Verified:** 2026-04-10 — confirmed stable across 5 consecutive runs on current hardware  
**All reproducers:** `high_confidence_bugs/reproducers/`  
**Inputs:** Embedded as literals — no external `.pt` files needed

---

## PyTorch — 42 bugs

| Reproducer | Bug Type | Confirmed L2 / Signal | Key APIs | Mutation |
|---|---|---|---|---|
| bug_pt071.py | inconsistent | L2=4.4361e-01 | Linear, BatchNorm1d, Dropout, sin | add_noise |
| bug_pt080.py | inconsistent | L2=5.9273e-03 | Linear, WeightNorm, BatchNorm1d | uniform |
| bug_pt101.py | inconsistent | L2=1.1677e-02 | Linear, BatchNorm1d, sin, cos | scale_large |
| bug_pt106.py | inconsistent | L2=9.0167e-02 | Linear, sin, cos, hardshrink | scale_large |
| bug_pt125.py | inconsistent | L2=2.4522e-01 | Linear, sin, enable_grad, BatchNorm1d, Dropout | scale_large |
| bug_pt138.py | inconsistent | L2=2.0447e+00 | Linear, sin, cos, tanh | scale_small |
| bug_pt144.py | inconsistent | L2=3.2524e-03 | Linear, BatchNorm1d, sin, cos | uniform |
| bug_pt147.py | inconsistent | **L2=8.3500e+01** | Linear, **BatchNorm1d**, sigmoid, expm1, polygamma_ | mask |
| bug_pt162.py | inconsistent | L2=4.8801e-02 | Linear, sin, cos, special.erfc | scale_large |
| bug_pt171.py | inconsistent | **L2=2.0108e+01** | Linear, poisson_nll_loss, cumprod, BatchNorm1d | mask |
| bug_pt191.py | inconsistent | L2=1.5129e+00 | Linear, **BatchNorm1d**, sin, arccos, clamp | add_noise |
| bug_pt192.py | **NaN** | **ASYM NaN: cpu=16, gpu=17, asym=1** | Linear, sin, cos, logsumexp | scale_large |
| bug_pt202.py | inconsistent | L2=1.7321e+00 | Linear, special.log_softmax, sin, cos | scale_large |
| bug_pt236.py | inconsistent | L2=6.0033e-02 | Linear, sin, cos, mean | scale_large |
| bug_pt241.py | inconsistent | L2=6.0545e+00 | Linear, ReLU, relu, requires_grad_ | uniform |
| bug_pt248.py | inconsistent | L2=1.3941e-02 | Linear, sin, cos, special.xlogy | scale_large |
| bug_pt281.py | inconsistent | L2=2.3222e+00 | Linear, sin, multilabel_soft_margin_loss, mul | mask |
| bug_pt284.py | inconsistent | L2=2.3937e-02 | Linear, sin, cos, Tensor.hardshrink | scale_large |
| bug_pt295.py | inconsistent | L2=2.8284e+00 | Linear, sin, Softshrink, **BatchNorm1d** | uniform |
| bug_pt305.py | inconsistent | L2=2.7761e-01 | Linear, **BatchNorm1d**, sin, Dropout, special.i1 | scale_small |
| bug_pt316.py | inconsistent | L2=7.1596e-03 | Linear, sin, cos, lgamma | scale_large |
| bug_pt319.py | inconsistent | L2=1.0162e-01 | Linear, ReLU, Unflatten, **Upsample** | scale_large |
| bug_pt335.py | inconsistent | L2=4.0375e-01 | Linear, **BatchNorm1d**, Dropout, pad | add_noise |
| bug_pt343.py | inconsistent | L2=7.8760e-01 | Linear, **BatchNorm1d**, sin, tanh, Dropout | uniform |
| bug_pt346.py | inconsistent | **L2=3.2768e+04** | hann_window, Linear, sin, cos | scale_large |
| bug_pt357.py | inconsistent | L2=5.7845e-01 | affine_grid, index_select, ge_, **BatchNorm1d**, Dropout | mask |
| bug_pt358.py | inconsistent | **L2=7.3856e+02** | Linear, softplus, sin, cos, geqrf | scale_large |
| bug_pt375.py | inconsistent | L2=2.4601e+00 | Linear, **BatchNorm1d**, dropout, tanh, sinc_ | add_noise |
| bug_pt382.py | inconsistent | L2=7.1658e-03 | Linear, sin, cos, nn.Hardswish | scale_large |
| bug_pt384.py | inconsistent | L2=5.0380e-02 | Linear, stft, sin, cos | scale_large |
| bug_pt390.py | inconsistent | L2=3.5405e-02 | Linear, sin, cos, clip | scale_large |
| bug_pt396.py | **NaN** | **cpu=Inf / gpu=NaN (scalar)** | Linear, LeakyReLU, AbsTransform, ormqr | uniform |
| bug_pt398.py | inconsistent | L2=9.4865e-03 | Linear, sin, cos, special.gammaln | scale_large |
| bug_pt404.py | inconsistent | **L2=5.4636e+04** | Linear, maximum, sin, cos | scale_large |
| bug_pt409.py | inconsistent | L2=3.4774e-01 | Linear, expm1, sin, BatchNorm1d | scale_small |
| bug_pt424.py | inconsistent | L2=3.9086e-03 | Linear, sin, cos, nn.Mish | scale_large |
| bug_pt428.py | inconsistent | L2=1.0558e-02 | Linear, sin, cos, nn.GLU | scale_large |
| bug_pt441.py | **crash** | **CPU crashes / GPU ok** | Linear, logcumsumexp, corrcoef, cholesky | scale_small |
| bug_pt450.py | inconsistent | L2=4.7500e+00 | distributions.transforms, Linear, SiLU, sin | uniform |
| bug_pt467.py | inconsistent | L2=1.7321e+00 | Linear, **BatchNorm1d**, dropout, arctan, matrix_exp | scale_large |
| bug_pt480.py | inconsistent | L2=6.1272e-02 | Linear, GELU, sin, cos | scale_large |
| bug_pt486.py | inconsistent | L2=3.0295e-03 | Linear, sin, cos (multi-output) | scale_large |

## TensorFlow — 2 bugs

Both are **baseline inconsistency** (no mutation — divergence from CPU/GPU kernel differences alone).

| Reproducer | Confirmed L2 | Key APIs |
|---|---|---|
| bug_tf067.py | L2=6.2642e-03 | linalg.diag, LocallyConnected1D, math.angle |
| bug_tf106.py | **L2=1.4250e+00** | digamma, SeparableConv2D, linalg.inv, mdct |

---

## How to Run

```bash
# Single bug
python3 high_confidence_bugs/reproducers/bug_pt404.py

# All bugs — every line should end with BUG CONFIRMED
for f in high_confidence_bugs/reproducers/*.py; do
    echo -n "$(basename $f): "
    python3 "$f" 2>/dev/null | grep -E "BUG CONFIRMED|not reproduced" | head -1
done
```
