"""
Bug   : CTCLoss gradient is incorrect
Issue : https://github.com/pytorch/pytorch/issues/52241
Class : cpu_vs_gpu  (gradient correctness)
Repro : torch.autograd.gradcheck rejects nn.CTCLoss with random log-probs;
        the analytic gradient does not match finite-difference gradient,
        and CPU vs CUDA disagree on the magnitude of the divergence.

Variants exercise:
  - vocab size  (V) sweep:           2, 5, 10
  - time-step   (T) sweep:           3, 8, 16
  - batch shape (N) sweep:           1, 4
  - reduction:                       'none', 'mean', 'sum'
  - blank index:                     0 vs V (last)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gradcheck_run(loss_fn, inputs):
    try:
        ok = torch.autograd.gradcheck(loss_fn, inputs, atol=1e-4, rtol=1e-3, eps=1e-6)
        return ok, None
    except Exception as exc:
        return False, str(exc).splitlines()[-1][:200]


def _make_ctc_inputs(T: int, N: int, V: int, blank: int = 0):
    log_probs = nn.functional.log_softmax(torch.randn(T, N, V + 1), dim=-1).double()
    log_probs.requires_grad = True
    targets = torch.randint(low=1 if blank == 0 else 0, high=V + 1, size=(N, max(1, V // 2)))
    if blank != 0:
        targets = torch.where(targets == blank, torch.zeros_like(targets), targets)
    input_lengths = torch.full((N,), T, dtype=torch.long)
    target_lengths = torch.full((N,), targets.size(1), dtype=torch.long)
    return log_probs, targets, input_lengths, target_lengths


def variant_1_minimal_repro() -> None:
    """Issue's own minimal repro: V=2, T=3, N=1."""
    print("variant_1_minimal_repro")
    torch.manual_seed(0)
    log_probs, targets, ilen, tlen = _make_ctc_inputs(T=3, N=1, V=2, blank=0)
    fn = nn.CTCLoss()
    ok, err = _gradcheck_run(fn, (log_probs, targets, ilen, tlen))
    print(f"  V=2 T=3 N=1 blank=0  gradcheck_passed={ok}  err={err}")


def variant_2_vocab_and_time_sweep() -> None:
    print("variant_2_vocab_and_time_sweep")
    for V in (2, 5, 10):
        for T in (3, 8, 16):
            torch.manual_seed(V * 100 + T)
            log_probs, targets, ilen, tlen = _make_ctc_inputs(T=T, N=1, V=V, blank=0)
            ok, err = _gradcheck_run(nn.CTCLoss(), (log_probs, targets, ilen, tlen))
            print(f"  V={V:>2} T={T:>2} N=1  ok={ok}  err={err}")


def variant_3_reduction_modes() -> None:
    print("variant_3_reduction_modes")
    for reduction in ("none", "mean", "sum"):
        torch.manual_seed(7)
        log_probs, targets, ilen, tlen = _make_ctc_inputs(T=4, N=2, V=3, blank=0)
        fn = nn.CTCLoss(reduction=reduction)
        ok, err = _gradcheck_run(fn, (log_probs, targets, ilen, tlen))
        print(f"  reduction={reduction:<5}  ok={ok}  err={err}")


def variant_4_blank_index_last() -> None:
    print("variant_4_blank_index_last")
    for V in (2, 4):
        torch.manual_seed(V)
        log_probs, targets, ilen, tlen = _make_ctc_inputs(T=4, N=1, V=V, blank=V)
        fn = nn.CTCLoss(blank=V)
        ok, err = _gradcheck_run(fn, (log_probs, targets, ilen, tlen))
        print(f"  V={V} blank={V}  ok={ok}  err={err}")


def variant_5_cpu_vs_cuda_grad() -> None:
    """Compare analytic gradients on CPU vs CUDA."""
    print("variant_5_cpu_vs_cuda_grad")
    if not torch.cuda.is_available():
        print("  [skip] CUDA not available")
        return
    torch.manual_seed(0)
    log_probs, targets, ilen, tlen = _make_ctc_inputs(T=8, N=2, V=5, blank=0)
    fn = nn.CTCLoss()
    log_probs_cpu = log_probs.detach().clone().requires_grad_(True)
    log_probs_gpu = log_probs.detach().clone().cuda().requires_grad_(True)
    fn(log_probs_cpu, targets, ilen, tlen).backward()
    fn(log_probs_gpu, targets.cuda(), ilen.cuda(), tlen.cuda()).backward()
    diff = (log_probs_cpu.grad - log_probs_gpu.grad.cpu()).abs().max().item()
    print(f"  max|grad_cpu - grad_cuda| = {diff:.3e}")


def main() -> None:
    print(f"torch={torch.__version__}")
    variant_1_minimal_repro()
    variant_2_vocab_and_time_sweep()
    variant_3_reduction_modes()
    variant_4_blank_index_last()
    variant_5_cpu_vs_cuda_grad()


if __name__ == "__main__":
    main()
