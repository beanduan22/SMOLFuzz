"""Fix the non-confirmed repros: training mode + remove model 220 (noise-level diff)."""
import torch, json, re
from pathlib import Path

bugs_dir   = Path('results/run_500_pytorch/full/bugs')
models_dir = Path('results/run_500_pytorch/full/models')
out_dir    = Path('high_confidence_bugs/reproducers')

# models that need TRAINING mode (BatchNorm batch-stats divergence)
training_mode_models = {125, 147, 191, 305, 343, 357, 375, 467}
# model 420, 433, 453: non-BN issues — just need correct inputs embedded
# model 220: epsilon-level complex noise — remove
remove_models = {220}


def tensor_to_literal(t):
    if not isinstance(t, torch.Tensor):
        return repr(t)
    dtype_str = str(t.dtype).split('.')[1]
    return f'torch.tensor({t.tolist()}, dtype=torch.{dtype_str})'


TEMPLATE = '''#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : {mid}
Bug Type: {bug_type}
Root Cause: {root_cause}
Detail  : {detail_short}
APIs    : {apis}
Mutation: {mut}

Run: python3 bug_pt{mid:03d}.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model ─────────────────────────────────────────────────────────────────────
{model_part}

# ── Reproducer ────────────────────────────────────────────────────────────────
def run():
    # Exact mutated inputs that triggered the bug (embedded from saved .pt file)
{inp_lines}
    inputs = {inp_return}

    torch.manual_seed(42)
    cpu_model = Model()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gpu_model = Model().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    {eval_line}

    {grad_ctx_open}
    cpu_out = cpu_model(*inputs)
    gpu_out = gpu_model(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs])
    {grad_ctx_close}

    def to_list(o):
        if isinstance(o, torch.Tensor):
            return [o.detach()]
        return [x.detach() for x in (o if isinstance(o, (list, tuple)) else [])
                if isinstance(x, torch.Tensor)]

    found = False
    for i, (c, g) in enumerate(zip(to_list(cpu_out), to_list(gpu_out))):
        g = g.cpu()
        if c.shape != g.shape:
            print(f"output[{{i}}]: shape mismatch cpu={{c.shape}} gpu={{g.shape}}")
            found = True
            continue
        cf, gf = c.float(), g.float()
        asym = (torch.isnan(cf) & ~torch.isnan(gf)) | (torch.isnan(gf) & ~torch.isnan(cf))
        asinf = (torch.isinf(cf) & ~torch.isinf(gf)) | (torch.isinf(gf) & ~torch.isinf(cf))
        nan_inf = (torch.isnan(cf) & torch.isinf(gf)) | (torch.isinf(cf) & torch.isnan(gf))
        if (asym | asinf | nan_inf).any():
            n = int((asym | asinf | nan_inf).sum().item())
            print(f"output[{{i}}]: ASYMMETRIC NaN/Inf — "
                  f"cpu_nan={{torch.isnan(cf).sum().item()}} "
                  f"gpu_nan={{torch.isnan(gf).sum().item()}} asym={{n}}")
            found = True
        else:
            fin = ~(torch.isnan(cf) | torch.isinf(cf) | torch.isnan(gf) | torch.isinf(gf))
            if fin.any():
                l2 = float((cf[fin] - gf[fin]).pow(2).sum().sqrt().item())
                if l2 > 1e-3:
                    print(f"output[{{i}}]: L2={{l2:.4e}}  shape={{list(c.shape)}}")
                    found = True

    if found:
        print("BUG CONFIRMED: CPU and GPU produce different results for the same model and input")
    else:
        print("Inconsistency not reproduced on this hardware/driver version")


if __name__ == "__main__":
    run()
'''


def get_best_bug(mid):
    jsons = sorted(bugs_dir.glob(f'bug_*m{mid:04d}*.json'))
    if not jsons:
        return None, None
    best_j, best_l2 = None, -1
    for jf in jsons:
        d = json.loads(jf.read_text())
        m = re.search(r'l2=([\d.e+-]+)', d.get('detail', ''))
        l2 = float(m.group(1)) if m else 0
        if d.get('bug_type') == 'nan':
            l2 = 999
        if l2 > best_l2:
            best_l2, best_j = l2, (jf, d)
    return best_j


def get_model_part(mid):
    src = models_dir / f'model_{mid:04d}.py'
    code_lines = [l for l in src.read_text().split('\n') if not l.startswith('#')]
    clean = '\n'.join(code_lines)
    m = re.search(
        r'((?:^import[^\n]*\n|^from[^\n]*\n)*.*?class Model.*?)'
        r'(?=^def make_inputs|^USED_APIS|\Z)',
        clean, re.DOTALL | re.MULTILINE,
    )
    return (m.group(0).strip() if m else clean.strip())


# Remove model 220 (epsilon-level complex noise)
for mid in remove_models:
    p = out_dir / f'bug_pt{mid:03d}.py'
    if p.exists():
        p.unlink()
        print(f'Removed bug_pt{mid:03d}.py (epsilon-level noise, not a real bug)')


# Fix training-mode models
for mid in training_mode_models:
    result = get_best_bug(mid)
    if result is None:
        continue
    jf, d = result

    inp_files = sorted(bugs_dir.glob(f'bug_*m{mid:04d}*.inputs.pt'))
    if not inp_files:
        print(f'  model {mid}: no inputs.pt')
        continue
    inputs = torch.load(str(inp_files[0]), weights_only=False)

    inp_lines_list = [f'    inp{i} = {tensor_to_literal(t)}' for i, t in enumerate(inputs)]
    inp_lines = '\n'.join(inp_lines_list)
    inp_return = '[' + ', '.join(f'inp{i}' for i in range(len(inputs))) + ']'

    out_path = out_dir / f'bug_pt{mid:03d}.py'
    out_path.write_text(TEMPLATE.format(
        mid=mid,
        bug_type=d['bug_type'],
        root_cause='BatchNorm1d CPU (sequential Welford) vs GPU (cuDNN parallel) batch-stat reduction divergence',
        detail_short=d.get('detail', '')[:100],
        apis=', '.join(d.get('used_apis', [])[:5]),
        mut=d.get('mutation_name', '?'),
        model_part=get_model_part(mid),
        inp_lines=inp_lines,
        inp_return=inp_return,
        eval_line='# NOTE: keep training mode — BatchNorm uses batch statistics (source of the bug)',
        grad_ctx_open='',
        grad_ctx_close='',
    ))
    print(f'  Fixed bug_pt{mid:03d}.py (training mode, BatchNorm batch-stats divergence)')


# Fix model 420 and 433 (need saved inputs, not BN issue)
for mid in [420, 433, 453]:
    result = get_best_bug(mid)
    if result is None:
        continue
    jf, d = result

    inp_files = sorted(bugs_dir.glob(f'bug_*m{mid:04d}*.inputs.pt'))
    if not inp_files:
        print(f'  model {mid}: no inputs.pt')
        continue
    inputs = torch.load(str(inp_files[0]), weights_only=False)

    inp_lines_list = [f'    inp{i} = {tensor_to_literal(t)}' for i, t in enumerate(inputs)]
    inp_lines = '\n'.join(inp_lines_list)
    inp_return = '[' + ', '.join(f'inp{i}' for i in range(len(inputs))) + ']'

    out_path = out_dir / f'bug_pt{mid:03d}.py'
    out_path.write_text(TEMPLATE.format(
        mid=mid,
        bug_type=d['bug_type'],
        root_cause='Numerical divergence between CPU and CUDA kernels',
        detail_short=d.get('detail', '')[:100],
        apis=', '.join(d.get('used_apis', [])[:5]),
        mut=d.get('mutation_name', '?'),
        model_part=get_model_part(mid),
        inp_lines=inp_lines,
        inp_return=inp_return,
        eval_line='cpu_model.eval(); gpu_model.eval()',
        grad_ctx_open='with torch.no_grad():',
        grad_ctx_close='',
    ))
    print(f'  Fixed bug_pt{mid:03d}.py (saved inputs embedded)')

print('Done')
