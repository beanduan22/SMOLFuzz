import torch; import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        x = self.fc1(x)
        torch.nn.init.kaiming_normal_(x)
        with torch.enable_grad():
            x.requires_grad_(True)
            x = self.bn(x)
            x = torch.nn.functional.glu(x)
            x = torch.unique_consecutive(x, return_inverse=False, return_counts=False, dim=1)
            x = torch.tensor_split(x, 2, dim=1)[0]
            x = float_power_(x, 2.0)
            x = torch.erf(x)
            x = torch.sort(x, dim=1).values
            x = torch.abs(x)
            x = torch.log1p(x)
            x = torch.asinh(x)
            x = torch.expm1(x)
            x = torch.amin(x, dim=1, keepdim=True)
            x = torch.negative(x)
        return x

def make_inputs():
    return [torch.randn(4, 8)]

USED_APIS = ["torch.nn.init.kaiming_normal_", "torch.nn.functional.glu", 
             "torch.unique_consecutive", "torch.Tensor.tensor_split", 
             "float_power_", "torch.erf", "torch.Tensor.sort", "torch.abs",
             "torch.log1p", "torch.asinh", "torch.expm1",
             "torch.amin", "torch.negative", "torch.enable_grad"]

def float_power_(tensor, exponent):
    return tensor ** exponent

# ===================== SMOLFuzz runner =====================
# False-positive prevention:
#   • .eval() disables Dropout/BatchNorm stochastic behavior
#   • TF32 off → GPU matmul matches CPU float32 (not ~1e-3 looser)
#   • cuDNN deterministic=True, benchmark=False → stable kernel choice
#   • CUBLAS_WORKSPACE_CONFIG :4096:8 → required for deterministic matmul
#   • use_deterministic_algorithms(True, warn_only=True) → catches non-det ops
#   • The same run is executed TWICE on GPU; a non-matching pair is marked
#     "nondet" so the oracle can drop it instead of reporting a false bug.
if __name__ == "__main__":
    import argparse, os, sys, traceback
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--double-run", action="store_true",
                        help="Run forward twice and include both outputs")
    args = parser.parse_args()

    # ----- Determinism / precision knobs -----
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    try:
        if args.input_file:
            inputs = torch.load(args.input_file, weights_only=False)
        else:
            torch.manual_seed(42)
            inputs = make_inputs()
            save_path = args.output_file + ".inputs.pt"
            torch.save(inputs, save_path)

        device = args.device
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = Model().to(device)
        model.eval()                       # disable Dropout / BatchNorm updates

        device_inputs = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                device_inputs.append(x.to(device))
            else:
                device_inputs.append(x)

        def to_cpu_list(out):
            if isinstance(out, torch.Tensor):
                return [out.detach().cpu()]
            elif isinstance(out, (list, tuple)):
                flat = []
                for o in out:
                    if isinstance(o, torch.Tensor):
                        flat.append(o.detach().cpu())
                return flat
            else:
                try:
                    return [torch.tensor(out)]
                except Exception:
                    return []

        # NB: not `torch.inference_mode()` — the paper explicitly requires
        # computational-graph / autograd dependencies to be exercised, which
        # inference_mode() forbids (in-place ops, grad, etc.). We leave grad
        # tracking enabled and rely on the model's own context managers
        # (no_grad, enable_grad, GradientTape) to drive the semantics.
        with torch.set_grad_enabled(True):
            output_a = model(*device_inputs)
        outputs_a = to_cpu_list(output_a)

        outputs_b = None
        if args.double_run:
            # Second run on the same device with identical inputs.
            # Any difference between outputs_a and outputs_b is
            # DEVICE-LEVEL non-determinism, not a CPU/GPU library bug.
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            model2 = Model().to(device)
            model2.eval()
            device_inputs2 = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    device_inputs2.append(x.to(device))
                else:
                    device_inputs2.append(x)
            with torch.set_grad_enabled(True):
                output_b = model2(*device_inputs2)
            outputs_b = to_cpu_list(output_b)

        result = {"status": "ok", "outputs": outputs_a, "outputs_repeat": outputs_b}
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=10),
        }

    torch.save(result, args.output_file)
