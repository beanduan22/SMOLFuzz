from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_both")

HERE    = Path(__file__).parent
PARENT  = HERE.parent


def _tail_summary(summary_file: Path, max_lines: int = 40) -> str:
    if not summary_file.exists():
        return "  (summary not found)"
    lines = summary_file.read_text().splitlines()
    return "\n".join(lines[:max_lines])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SMOLFuzz: run PyTorch + TF campaigns in parallel"
    )
    ap.add_argument("--models", type=int, default=500,
                    help="Number of models per framework")
    ap.add_argument("--budget", type=int, default=60,
                    help="Mutation fuzzing budget in seconds per model")
    ap.add_argument("--api-set-size", type=int, default=30,
                    help="APIs per PyTorch model")
    ap.add_argument("--tf-api-set-size", type=int, default=30,
                    help="APIs per TF model")
    ap.add_argument("--out-base", default=str(HERE / "results"),
                    help="Base output directory")
    args = ap.parse_args()

    out_base  = Path(args.out_base)
    pt_outdir = out_base / f"run_{args.models}_pytorch"
    tf_outdir = out_base / f"run_{args.models}_tf"

    log.info("=" * 70)
    log.info("SMOLFuzz: PyTorch + TensorFlow | models=%d budget=%ds",
             args.models, args.budget)
    log.info("PyTorch output → %s", pt_outdir)
    log.info("TF      output → %s", tf_outdir)
    log.info("=" * 70)

    pt_cmd = [
        sys.executable, "-m", "smolfuzz.main",
        "--mode",         "full",
        "--models",       str(args.models),
        "--budget",       str(args.budget),
        "--api-set-size", str(args.api_set_size),
        "--output-dir",   str(pt_outdir),
    ]

    tf_cmd = [
        sys.executable, "-m", "smolfuzz.run_tf",
        "--models",       str(args.models),
        "--budget",       str(args.budget),
        "--api-set-size", str(args.tf_api_set_size),
        "--out",          str(tf_outdir),
    ]

    log.info("Launching PyTorch fuzzer…")
    out_base.mkdir(parents=True, exist_ok=True)
    pt_log_path = out_base / f"run_{args.models}_pytorch.log"
    pt_outdir.mkdir(parents=True, exist_ok=True)
    pt_proc = subprocess.Popen(
        pt_cmd,
        stdout=open(pt_log_path, "w"), stderr=subprocess.STDOUT,
        cwd=str(PARENT),
    )
    log.info("PyTorch PID %d  log → %s", pt_proc.pid, pt_log_path)

    time.sleep(5)

    log.info("Launching TensorFlow fuzzer…")
    tf_log_path = out_base / f"run_{args.models}_tf.log"
    tf_proc = subprocess.Popen(
        tf_cmd,
        stdout=open(tf_log_path, "w"), stderr=subprocess.STDOUT,
        cwd=str(PARENT),
    )
    log.info("TF      PID %d  log → %s", tf_proc.pid, tf_log_path)

    start = time.time()
    poll_interval = 60

    while True:
        pt_done = pt_proc.poll() is not None
        tf_done = tf_proc.poll() is not None

        elapsed_h = (time.time() - start) / 3600
        log.info(
            "Progress | elapsed=%.1fh | PyTorch=%s (rc=%s) | TF=%s (rc=%s)",
            elapsed_h,
            "DONE" if pt_done else "running", pt_proc.returncode,
            "DONE" if tf_done else "running", tf_proc.returncode,
        )

        if pt_done and tf_done:
            break

        time.sleep(poll_interval)

    pt_rc = pt_proc.returncode
    tf_rc = tf_proc.returncode
    total_time = time.time() - start

    log.info("=" * 70)
    log.info("Both campaigns finished | total_time=%.1fh", total_time / 3600)
    log.info("PyTorch exit code: %d", pt_rc)
    log.info("TF      exit code: %d", tf_rc)

    tf_summary_file = tf_outdir / "summary.txt"

    def _log_tail(log_path: Path, max_lines: int = 30) -> str:
        if not log_path.exists():
            return "  (log not found)"
        lines = log_path.read_text().splitlines()
        return "\n".join(lines[-max_lines:])

    combined_path = out_base / f"run_{args.models}_combined_summary.txt"
    combined_lines = [
        "SMOLFuzz Combined Run — COMPLETED",
        f"Finished:        {datetime.now()}",
        f"Total wall time: {total_time / 3600:.2f} h",
        f"Models per fw:   {args.models}",
        f"Budget/model:    {args.budget}s",
        "",
        "=" * 60,
        "PYTORCH (CPU vs CUDA)",
        f"  Exit code: {pt_rc}",
        f"  Log:       {pt_log_path}",
        f"  Bugs dir:  {pt_outdir}/bugs",
        "  Last log lines:",
        _log_tail(pt_log_path, 20),
        "",
        "=" * 60,
        "TENSORFLOW (CPU vs GPU)",
        f"  Exit code: {tf_rc}",
        f"  Log:       {tf_log_path}",
        f"  Summary:",
        _tail_summary(tf_summary_file),
    ]
    combined_path.write_text("\n".join(combined_lines))
    log.info("Combined summary → %s", combined_path)

    if pt_rc != 0:
        log.error("PyTorch fuzzer exited with code %d — check %s", pt_rc, pt_log_path)
    if tf_rc != 0:
        log.error("TF fuzzer exited with code %d — check %s", tf_rc, tf_log_path)

    sys.exit(0 if pt_rc == 0 and tf_rc == 0 else 1)


if __name__ == "__main__":
    main()
