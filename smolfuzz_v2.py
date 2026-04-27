from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_COUNTS = HERE / "data" / "smolfuzz_v2_counts.json"


@dataclass(frozen=True)
class StratumRow:
    stratum: str
    framework: str
    issues: int
    state: int
    non_state: int
    u_dup: int

    @property
    def pct_state(self) -> float:
        return 100.0 * self.state / self.issues


def pct(part: int, whole: int) -> float:
    return 100.0 * part / whole


def load_counts(path: Path) -> dict:
    with path.open() as fh:
        data = json.load(fh)
    validate_counts(data)
    return data


def validate_counts(data: dict) -> None:
    meta = data["issue_collection"]
    rows = [StratumRow(**row) for row in data["strata"]]

    total_issues = sum(row.issues for row in rows)
    total_state = sum(row.state for row in rows)
    total_non = sum(row.non_state for row in rows)
    total_udup = sum(row.u_dup for row in rows)
    if total_issues != meta["sample_size"]:
        raise ValueError(f"sample_size mismatch: {total_issues} != {meta['sample_size']}")
    if total_state + total_non + total_udup != total_issues:
        raise ValueError("sample totals do not balance")

    fix_verified = sum(row.issues for row in rows if row.stratum == "Fix-verified")
    unverified = sum(row.issues for row in rows if row.stratum == "Unverified")
    if fix_verified != meta["fix_verified_sample"]:
        raise ValueError("fix_verified_sample mismatch")
    if unverified != meta["unverified_sample"]:
        raise ValueError("unverified_sample mismatch")

    dims = data["dimensions"]
    pt_total = sum(dims["PyTorch"].values())
    tf_total = sum(dims["TensorFlow"].values())
    if pt_total != sum(row.state for row in rows if row.framework == "PyTorch"):
        raise ValueError("PyTorch state total mismatch")
    if tf_total != sum(row.state for row in rows if row.framework == "TensorFlow"):
        raise ValueError("TensorFlow state total mismatch")


def subtotal(rows: list[StratumRow], *, stratum: str | None = None, framework: str | None = None) -> StratumRow:
    selected = [
        row for row in rows
        if (stratum is None or row.stratum == stratum)
        and (framework is None or row.framework == framework)
    ]
    return StratumRow(
        stratum=stratum or "Total",
        framework=framework or "All",
        issues=sum(row.issues for row in selected),
        state=sum(row.state for row in selected),
        non_state=sum(row.non_state for row in selected),
        u_dup=sum(row.u_dup for row in selected),
    )


def format_pct(value: float) -> str:
    return f"{value:.1f}\\%"


def build_methodology(meta: dict) -> str:
    agreement_pct = pct(meta["human_agreement"], meta["human_total"])
    return rf"""\subsection{{Methodology}}
\label{{sec:empirical-method}}

\noindent\textbf{{Issue collection.}} We surveyed closed GitHub issues in PyTorch and TensorFlow from January 2021 to April 2026. Because the two repositories use different bug-labeling conventions, we do not rely on a single maintainer-assigned bug label. Instead, we filter by repository-native labels (\texttt{{module:*}} and \texttt{{triaged}} for PyTorch, \texttt{{comp:*}} and \texttt{{type:bug}} for TensorFlow) over five state-relevant subsystems per framework: component labels localize issues to state-relevant subsystems, while \texttt{{triaged}} (PyTorch) and \texttt{{type:bug}} (TensorFlow) indicate that the report has been reviewed by maintainers. This returned {meta["raw_hits"]:,} raw hits ({meta["unique_hits"]:,} unique after URL deduplication).

\noindent\textbf{{Filtering.}} We retained issues whose title or body contains at least one correctness-related symptom keyword (e.g., \emph{{incorrect}}, \emph{{wrong}}, \emph{{NaN}}, \emph{{unexpected}}, \emph{{inconsistent}}) and excluded pure feature requests, documentation fixes, build or install problems, and duplicates. This produced a filtered pool of {meta["filtered_pool"]:,} correctness-relevant reports.

\noindent\textbf{{Sampling.}} An issue is \emph{{fix-verified}} if its GitHub history provides a publicly auditable link to a merged fixing PR or a default-branch commit, and \emph{{unverified}} otherwise. In the {meta["filtered_pool"]:,}-issue pool, {meta["fix_verified_pool"]} issues are fix-verified and {meta["unverified_pool"]} are unverified. We drew a stratified sample of {meta["sample_size"]} issues, sampling proportionally from each stratum, yielding {meta["fix_verified_sample"]} fix-verified and {meta["unverified_sample"]} unverified issues.

\noindent\textbf{{Classification.}} Each sampled issue was assigned a single primary category from a five-category rubric: (A)~gradient tracking, (B)~execution mode, (C)~distribution strategy, (D)~other state, and (E)~non-state. When an issue exhibited signals from more than one state dimension, raters selected the dimension most directly responsible for the reported incorrect behavior. Two auxiliary tags, U (unclassifiable) and DUP (duplicate), were used for quality control.

\noindent\textbf{{Label reliability.}} To scale classification across 200 issues while retaining human oversight, we adopted a hybrid labeling protocol. We partitioned the sample into two equal subsets, each independently classified by a different general-purpose LLM rater (GPT-5 Nano and Claude Sonnet 4.5) under the same rubric and prompt. To assess label quality, one of the authors, blinded to the LLM labels, independently re-classified all {meta["human_total"]} issues; the author's labels agreed with the LLM labels on {meta["human_agreement"]}/{meta["human_total"]} ({agreement_pct:.0f}\%) primary categories, with Cohen's $\kappa = {meta["cohen_kappa"]:.2f}$~\cite{{landis1977measurement}}, indicating substantial human--LLM agreement.
"""


def build_stratum_table(rows: list[StratumRow]) -> str:
    fix_pt = next(row for row in rows if row.stratum == "Fix-verified" and row.framework == "PyTorch")
    fix_tf = next(row for row in rows if row.stratum == "Fix-verified" and row.framework == "TensorFlow")
    unv_pt = next(row for row in rows if row.stratum == "Unverified" and row.framework == "PyTorch")
    unv_tf = next(row for row in rows if row.stratum == "Unverified" and row.framework == "TensorFlow")
    fix_total = subtotal(rows, stratum="Fix-verified")
    unv_total = subtotal(rows, stratum="Unverified")
    total_pt = subtotal(rows, framework="PyTorch")
    total_tf = subtotal(rows, framework="TensorFlow")
    grand = subtotal(rows)

    return rf"""\begin{{table}}[t!]
\small
\centering
\setlength{{\tabcolsep}}{{2.5pt}}
\caption{{Counts for the 200-issue sample.}}
\label{{tab:empirical-per-stratum}}
\begin{{tabular}}{{llrrrrr}}
\toprule
Stratum & Framework & \# Issues & State & Non (E) & U/DUP & \%State \\
\midrule
\multirow{{3}}{{*}}{{Fix-verified}}
 & PyTorch    & {fix_pt.issues}  & {fix_pt.state} & {fix_pt.non_state} & {fix_pt.u_dup} & {format_pct(fix_pt.pct_state)} \\
 & TensorFlow & {fix_tf.issues}  & {fix_tf.state} & {fix_tf.non_state} & {fix_tf.u_dup} & {format_pct(fix_tf.pct_state)} \\
 & Subtotal   & {fix_total.issues} & {fix_total.state} & {fix_total.non_state} & {fix_total.u_dup} & {format_pct(fix_total.pct_state)} \\
\midrule
\multirow{{3}}{{*}}{{Unverified}}
 & PyTorch    & {unv_pt.issues} & {unv_pt.state} & {unv_pt.non_state} & {unv_pt.u_dup} & {format_pct(unv_pt.pct_state)} \\
 & TensorFlow & {unv_tf.issues} & {unv_tf.state} & {unv_tf.non_state} & {unv_tf.u_dup} & {format_pct(unv_tf.pct_state)} \\
 & Subtotal   & {unv_total.issues} & {unv_total.state} & {unv_total.non_state} & {unv_total.u_dup} & {format_pct(unv_total.pct_state)} \\
\midrule
\multirow{{3}}{{*}}{{Total}}
 & PyTorch    & {total_pt.issues} & {total_pt.state} & {total_pt.non_state} & {total_pt.u_dup} & {format_pct(total_pt.pct_state)} \\
 & TensorFlow & {total_tf.issues}  & {total_tf.state} & {total_tf.non_state} & {total_tf.u_dup} & {format_pct(total_tf.pct_state)} \\
 & \textbf{{All}} & \textbf{{{grand.issues}}} & \textbf{{{grand.state}}} & \textbf{{{grand.non_state}}} & \textbf{{{grand.u_dup}}} & \textbf{{{format_pct(grand.pct_state)}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_dimension_table(dimensions: dict[str, dict[str, int]]) -> str:
    labels = [
        ("Gradient tracking", "A.~Gradient tracking"),
        ("Execution mode", "B.~Execution mode"),
        ("Distribution strategy", "C.~Distribution strategy"),
        ("Other state", "D.~Other state"),
    ]
    pt_total = sum(dimensions["PyTorch"].values())
    tf_total = sum(dimensions["TensorFlow"].values())
    all_total = pt_total + tf_total
    abc_pt = sum(dimensions["PyTorch"][name] for name, _ in labels[:3])
    abc_tf = sum(dimensions["TensorFlow"][name] for name, _ in labels[:3])
    abc_all = abc_pt + abc_tf

    body = []
    for key, label in labels:
        pt = dimensions["PyTorch"][key]
        tf = dimensions["TensorFlow"][key]
        total = pt + tf
        body.append(
            f"{label:<24} & {pt:2d} & {format_pct(pct(pt, pt_total))} &"
            f" {tf:2d} & {format_pct(pct(tf, tf_total))} &"
            f" {total:2d} & {format_pct(pct(total, all_total))} \\\\"
        )

    body_text = "\n".join(body)
    return rf"""\begin{{table}}[t!]
\small
\centering
\setlength{{\tabcolsep}}{{3.5pt}}
\caption{{Distribution over the {all_total} state-related issues.}}
\label{{tab:empirical-per-dim}}
\begin{{tabular}}{{lrr|rr|rr}}
\toprule
& \multicolumn{{2}}{{c}}{{PyTorch}} & \multicolumn{{2}}{{c}}{{TensorFlow}} & \multicolumn{{2}}{{c}}{{Total}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}}
Primary dimension & \# & \% & \# & \% & \# & \% \\
\midrule
{body_text}
\midrule
{{Primary category in A--C}} & \textbf{{{abc_pt}}} & \textbf{{{format_pct(pct(abc_pt, pt_total))}}} & \textbf{{{abc_tf}}} & \textbf{{{format_pct(pct(abc_tf, tf_total))}}} & \textbf{{{abc_all}}} & \textbf{{{format_pct(pct(abc_all, all_total))}}} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def build_results(rows: list[StratumRow], dimensions: dict[str, dict[str, int]]) -> str:
    fix_total = subtotal(rows, stratum="Fix-verified")
    unv_total = subtotal(rows, stratum="Unverified")
    total_pt = subtotal(rows, framework="PyTorch")
    total_tf = subtotal(rows, framework="TensorFlow")
    grand = subtotal(rows)

    pt_total_state = sum(dimensions["PyTorch"].values())
    tf_total_state = sum(dimensions["TensorFlow"].values())
    all_state = pt_total_state + tf_total_state
    abc_pt = sum(dimensions["PyTorch"][name] for name in ("Gradient tracking", "Execution mode", "Distribution strategy"))
    abc_tf = sum(dimensions["TensorFlow"][name] for name in ("Gradient tracking", "Execution mode", "Distribution strategy"))
    abc_all = abc_pt + abc_tf

    return rf"""\subsection{{Results}}

{build_stratum_table(rows)}

{build_dimension_table(dimensions)}

Table~\ref{{tab:empirical-per-stratum}} reports per-stratum classification counts. State-related issues are common across both strata and both frameworks: they account for {format_pct(fix_total.pct_state)} of fix-verified issues and {format_pct(unv_total.pct_state)} of unverified issues, and for {format_pct(total_pt.pct_state)} of PyTorch issues and {format_pct(total_tf.pct_state)} of TensorFlow issues. Overall, {grand.state} of the {grand.issues} sampled issues are state-related, indicating that state-related bugs are common in the sampled correctness-oriented issue set rather than confined to isolated edge cases or to a single framework.

Table~\ref{{tab:empirical-per-dim}} reports the primary-category distribution over the {all_state} state-related issues. The three primary dimensions account for {abc_all} of the {all_state} state-related issues ({format_pct(pct(abc_all, all_state))}), and this proportion is stable across frameworks ({format_pct(pct(abc_pt, pt_total_state))} for PyTorch and {format_pct(pct(abc_tf, tf_total_state))} for TensorFlow). The dominant dimension differs between the two ecosystems: PyTorch issues are distributed more evenly across gradient tracking ({format_pct(pct(dimensions["PyTorch"]["Gradient tracking"], pt_total_state))}) and distribution strategy ({format_pct(pct(dimensions["PyTorch"]["Distribution strategy"], pt_total_state))}), whereas TensorFlow issues concentrate on execution mode ({format_pct(pct(dimensions["TensorFlow"]["Execution mode"], tf_total_state))}), consistent with the prominence of \texttt{{tf.function}} tracing and eager/graph transitions. Only {dimensions["PyTorch"]["Other state"] + dimensions["TensorFlow"]["Other state"]} issues ({format_pct(pct(dimensions["PyTorch"]["Other state"] + dimensions["TensorFlow"]["Other state"], all_state))}) fall into the residual category of other state dimensions. In other words, although a long tail of runtime states exists and the framework-level emphases differ, the state-related bug surface in the sampled pool is concentrated on a small number of dimensions in both ecosystems.
"""


def build_document(data: dict) -> str:
    rows = [StratumRow(**row) for row in data["strata"]]
    meta = data["issue_collection"]
    dimensions = data["dimensions"]
    return "\n\n".join([
        build_methodology(meta),
        build_results(rows, dimensions),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the SMOLFuzz v2 empirical-study LaTeX from validated counts."
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=DEFAULT_COUNTS,
        help="Path to the JSON counts file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    data = load_counts(args.counts)
    latex = build_document(data)

    if args.out is None:
        print(latex)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
