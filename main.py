from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "smolfuzz"

from .core.api_loader import group_summary, load_and_classify
from .core.executor import ModelExecutor, has_randomness, has_nondet_gpu_op
from .backends.llm_client import OllamaClient
from .core.oracle import DifferentialOracle
from .core.selector import MultiRouletteSelector
from .core.skeletons import get_skeletons
from .core.synthesizer import ModelSynthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
DEFAULT_API_FILE = HERE / "torch_valid_apis.txt"
DEFAULT_OUTPUT   = HERE / "results"

_MUTATION_NAMES = {1: "add_noise", 2: "scale_small", 3: "mask",
                   4: "uniform",   5: "scale_large"}

NEAR_MISS_REL_ERR = 1e-2


def run(
    api_file: Path,
    output_dir: Path,
    n_models: int,
    api_set_size: int,
    fuzzing_budget_s: int,
    llm_models: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    logger.info("Loading API list from %s", api_file)
    groups = load_and_classify(api_file)
    logger.info("API classification:\n%s", group_summary(groups))

    selectable_groups = {k: v for k, v in groups.items() if k != "_excluded"}
    selector    = MultiRouletteSelector(selectable_groups)
    client      = OllamaClient(models=llm_models) if llm_models else OllamaClient()
    synthesizer = ModelSynthesizer(client, target_lib="PyTorch")
    executor    = ModelExecutor(output_dir / "workspace")
    oracle      = DifferentialOracle(output_dir)
    skeletons   = get_skeletons("pytorch")

    if n_models > 0:
        logger.info("Starting SMOLFuzz | models=%d api_set=%d budget=%ds",
                    n_models, api_set_size, fuzzing_budget_s)
    else:
        logger.info(
            "Starting SMOLFuzz | models=until_termination api_set=%d budget=%ds",
            api_set_size, fuzzing_budget_s,
        )
    logger.info("LLM models: %s", client._models)

    total_bugs     = 0
    total_clean    = 0
    total_nondet   = 0
    total_gen_fail = 0
    failed_synth   = 0
    invalid_models = 0
    rejected_random = 0

    all_apis_union: set[str] = set()
    for g_name, apis in selectable_groups.items():
        all_apis_union.update(apis)
    apis_attempted: set[str] = set()
    apis_executed:  set[str] = set()
    coverage_history: list[tuple[int, int]] = []

    strategy_counts: dict[int, int] = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    def _pick_strategy() -> int:
        total = sum(strategy_counts.values())
        r = random.random() * total
        for s in sorted(strategy_counts):
            r -= strategy_counts[s]
            if r <= 0:
                return s
        return max(strategy_counts, key=strategy_counts.__getitem__)

    def _is_near_miss(report) -> bool:
        if report.is_bug() or report.is_nondet():
            return True
        detail = (report.detail or "").lower()
        if "rel_err=" in detail:
            try:
                token = detail.split("rel_err=", 1)[1].split()[0]
                rel = float(token)
                return rel >= NEAR_MISS_REL_ERR
            except Exception:
                return False
        return False

    MAX_NO_NEW_API_STREAK = 10
    consecutive_no_new_apis = 0

    def _update_no_new_api_streak(used_apis: list[str]) -> bool:
        nonlocal consecutive_no_new_apis
        new_apis = set(used_apis) - apis_executed
        if new_apis:
            consecutive_no_new_apis = 0
            return False
        consecutive_no_new_apis += 1
        if consecutive_no_new_apis >= MAX_NO_NEW_API_STREAK:
            logger.info(
                "Early stop: %d consecutive models introduced no new APIs",
                MAX_NO_NEW_API_STREAK,
            )
            return True
        return False

    i = 0
    while True:
        if n_models > 0 and i >= n_models:
            break
        model_num = i + 1
        skeleton = skeletons[i % len(skeletons)]
        i += 1
        logger.info("=" * 60)
        if n_models > 0:
            logger.info("Model %d / %d", model_num, n_models)
        else:
            logger.info("Model %d", model_num)
        plan = selector.select_plan(n=api_set_size)
        api_set = plan.all_apis
        selector.record_attempts(api_set)
        apis_attempted.update(api_set)
        logger.info(
            "Selected skeleton=%s (%s), pool=%d",
            skeleton.skeleton_id,
            skeleton.description,
            len(api_set),
        )

        synth = synthesizer.synthesize(skeleton, api_set)

        model_path = models_dir / f"model_{synth.model_id:04d}.py"
        model_path.write_text(
            f"# SMOLFuzz model {synth.model_id} | llm={synth.llm_model}"
            f" | skeleton={synth.skeleton_id}"
            f" | attempts={synth.attempts} | apis={len(synth.used_apis)}"
            f" | error={'yes' if synth.error else 'no'}\n"
            f"# CANDIDATE_POOL = {api_set}\n\n"
            + synth.code
        )

        if synth.error:
            logger.error("Synthesis failed for model %d: %s",
                         synth.model_id, synth.error[:200])
            failed_synth += 1
            selector.record_usage(synth.used_apis, anomaly_detected=False)
            continue

        if has_randomness(synth.code):
            logger.warning(
                "Model %d rejected: LLM inserted random op (dropout/rand/bernoulli)"
                " — comparing CPU vs GPU would false-positive. Skipping.",
                synth.model_id,
            )
            rejected_random += 1
            selector.record_usage(synth.used_apis, anomaly_detected=False)
            continue

        if has_nondet_gpu_op(synth.code):
            logger.info(
                "Model %d uses GPU-nondeterministic op (scatter_add / segment_sum / "
                "grid_sample / interpolate / embedding_bag). Will still fuzz, but "
                "oracle will tag diffs as nondet unless they exceed the noise floor.",
                synth.model_id,
            )

        logger.info("Synthesis OK | llm=%s | attempts=%d | used_apis=%d",
                    synth.llm_model, synth.attempts, len(synth.used_apis))

        logger.info("Validating baseline (CPU + GPU)…")
        pair, inputs = executor.run_baseline(synth.code, synth.model_id)

        cpu_ok = pair.cpu.status == "ok"
        gpu_ok = pair.gpu.status == "ok"

        if not cpu_ok or not gpu_ok:
            logger.warning(
                "Model %d invalid: cpu=%s gpu=%s — skipping",
                synth.model_id, pair.cpu.status, pair.gpu.status,
            )
            if not cpu_ok:
                logger.warning("  CPU error: %s", pair.cpu.error[:200])
            if not gpu_ok:
                logger.warning("  GPU error: %s", pair.gpu.error[:200])
            invalid_models += 1
            selector.record_usage(synth.used_apis, anomaly_detected=False)
            continue

        if inputs is None:
            logger.warning("Model %d: no inputs recovered — skipping", synth.model_id)
            invalid_models += 1
            selector.record_usage(synth.used_apis, anomaly_detected=False)
            continue

        apis_executed.update(synth.used_apis)
        coverage_history.append((i + 1, len(apis_executed)))

        logger.info("Baseline OK — model is valid, starting fuzzing for %ds…",
                    fuzzing_budget_s)

        found_bug = False
        saw_anomaly = False
        mutation_count = 0
        budget_start = time.time()
        tried_in_sweep: set[int] = set()
        sweep_signal = False
        cur_inputs = inputs

        while time.time() - budget_start < fuzzing_budget_s:
            strategy = _pick_strategy()
            mut_name = _MUTATION_NAMES[strategy]
            mutation_count += 1
            elapsed = time.time() - budget_start
            logger.info("  Mutation #%d (%s) at %.1fs", mutation_count, mut_name, elapsed)
            tried_in_sweep.add(strategy)

            mpair = executor.run_mutation(
                synth.code, synth.model_id, cur_inputs, strategy
            )
            mreport = oracle.evaluate(mpair, synth.used_apis, synth.code)

            if mreport.is_bug():
                logger.warning("  BUG found | model=%d mutation=%s",
                               synth.model_id, mut_name)
                strategy_counts[strategy] += 1
                total_bugs += 1
                found_bug = True
                saw_anomaly = True
                sweep_signal = True
                break

            if _is_near_miss(mreport):
                strategy_counts[strategy] += 1
                sweep_signal = True
                if mreport.is_nondet():
                    total_nondet += 1
                    saw_anomaly = True
                    logger.info("  Non-deterministic (near-miss, not a bug)")
                elif mreport.is_generation_failure():
                    total_gen_fail += 1
                    saw_anomaly = True
                    logger.info("  Generation failure (near-miss, not a bug)")
                else:
                    logger.info("  Near-miss (rel_err >= %.0e)", NEAR_MISS_REL_ERR)
            else:
                logger.info("  Clean (no bug)")
                total_clean += 1

            if len(tried_in_sweep) == 5:
                if not sweep_signal:
                    logger.info(
                        "  Full strategy sweep with no anomaly or near-miss — re-sampling inputs"
                    )
                    new_pair, new_inputs = executor.run_baseline(
                        synth.code, synth.model_id
                    )
                    if (new_inputs is not None
                            and new_pair.cpu.status == "ok"
                            and new_pair.gpu.status == "ok"):
                        cur_inputs = new_inputs
                tried_in_sweep = set()
                sweep_signal = False

        elapsed_total = time.time() - budget_start
        if found_bug:
            logger.info("Model %d: bug found after %d mutations (%.1fs)",
                        synth.model_id, mutation_count, elapsed_total)
        else:
            logger.info("Model %d: no bug found after %d mutations (%.1fs budget)",
                        synth.model_id, mutation_count, elapsed_total)

        selector.record_usage(synth.used_apis, anomaly_detected=saw_anomaly)

        if _update_no_new_api_streak(synth.used_apis):
            break

    logger.info("=" * 60)
    logger.info(
        "DONE | models=%d failed_synth=%d invalid=%d rejected_random=%d "
        "bugs=%d clean=%d nondet=%d gen_fail=%d",
        i, failed_synth, invalid_models, rejected_random,
        total_bugs, total_clean, total_nondet, total_gen_fail,
    )

    n_total = len(all_apis_union)
    n_att = len(apis_attempted)
    n_exec = len(apis_executed)
    logger.info("=" * 60)
    logger.info("API COVERAGE")
    logger.info("  Total APIs in selectable pool : %d", n_total)
    logger.info("  APIs attempted  (selected)    : %d  (%.1f%%)",
                n_att, 100.0 * n_att / max(1, n_total))
    logger.info("  APIs executed   (both devices): %d  (%.1f%%)",
                n_exec, 100.0 * n_exec / max(1, n_total))

    logger.info("  Per-group attempted/executed:")
    for gname, apis in selectable_groups.items():
        total = len(apis)
        if total == 0:
            continue
        att_n = sum(1 for a in apis if a in apis_attempted)
        exec_n = sum(1 for a in apis if a in apis_executed)
        conv = (100.0 * exec_n / att_n) if att_n else 0.0
        logger.info(
            "    %-22s : attempted %3d / %3d (%.1f%%) | executed %3d / %3d (%.1f%%) | exec/att %.1f%%",
            gname,
            att_n, total, 100.0 * att_n / total,
            exec_n, total, 100.0 * exec_n / total,
            conv,
        )

    selector_stats = selector.stats()
    zero_exec_groups = selector_stats.get("groups_zero_executed", [])
    low_conversion_groups = selector_stats.get("groups_low_conversion", [])
    logger.info("  Groups with zero executed APIs: %s", zero_exec_groups)
    logger.info("  Groups with low executed/attempted conversion: %s", low_conversion_groups)

    import json as _json
    coverage_data = {
        "n_models": i,
        "n_total_apis": n_total,
        "n_attempted": n_att,
        "n_executed": n_exec,
        "attempted": sorted(apis_attempted),
        "executed": sorted(apis_executed),
        "coverage_history": coverage_history,
        "per_group": {
            g: {"total": len(apis),
                "attempted": sum(1 for a in apis if a in apis_attempted),
                "executed": sum(1 for a in apis if a in apis_executed)}
            for g, apis in selectable_groups.items()
        },
        "totals": {
            "bugs": total_bugs, "clean": total_clean,
            "nondet": total_nondet, "gen_fail": total_gen_fail,
            "failed_synth": failed_synth,
            "invalid_models": invalid_models,
            "rejected_random": rejected_random,
        },
        "selector_stats": selector_stats,
    }
    (output_dir / "coverage.json").write_text(_json.dumps(coverage_data, indent=2))
    logger.info("Coverage → %s", output_dir / "coverage.json")

    logger.info("LLM stats: %s", client.stats())
    logger.info("Selector stats: %s", selector_stats)
    logger.info("Bug reports → %s", output_dir / "bugs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SMOLFuzz: LLM-based DL library fuzzer")
    p.add_argument("--mode", choices=["subset", "full"], default="subset",
                   help="subset = 5 models for validation; full = n_models")
    p.add_argument("--models", type=int, default=0,
                   help="Number of models to synthesise in full mode; 0 means stop only on the no-new-API condition")
    p.add_argument("--api-set-size", type=int, default=12,
                   help="APIs per synthesised model")
    p.add_argument("--budget", type=int, default=60,
                   help="Mutation fuzzing budget in seconds per model")
    p.add_argument("--api-file", type=Path, default=DEFAULT_API_FILE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--llm-models", type=str, default=None,
                   help="Comma-separated Ollama model names (overrides defaults)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    llm_models = [m.strip() for m in args.llm_models.split(",")] \
                 if args.llm_models else None

    if args.mode == "subset":
        logger.info("SUBSET VALIDATION MODE: 5 models")
        run(
            api_file=args.api_file,
            output_dir=args.output_dir / "subset2",
            n_models=5,
            api_set_size=12,
            fuzzing_budget_s=30,
            llm_models=llm_models,
        )
    else:
        run(
            api_file=args.api_file,
            output_dir=args.output_dir / "full",
            n_models=args.models,
            api_set_size=args.api_set_size,
            fuzzing_budget_s=args.budget,
            llm_models=llm_models,
        )
