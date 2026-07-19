#!/usr/bin/env python3
"""Regenerate the Chapter 5 LaTeX table blocks straight from results/*.json.

Emits full \\begin{table}...\\end{table} environments for the four tables in
main.tex whose numbers are computed rather than authored: the joint threshold
sweep (tab:surface), the draft tree size sweep (tab:trees), and the headline
comparison (tab:headline_values / tab:headline_ratio). Formatting (column
spec, captions, labels) is fixed to match main.tex as of the 2026-07-19
attention-pinning reversal -- paste the printed blocks in to replace the
existing ones, then re-check any prose that cites specific numbers from them.

Reads results/drive_all.json (surface + headline) and
results/pruning_analysis.json (trees). Pure stdlib, run from the capim/ dir:

    python3 scripts/cpu/gen_tables.py                 # all four, to stdout
    python3 scripts/cpu/gen_tables.py --table surface # just one
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Tuple

SIGMAS = [-0.5, -1.0, -1.5, -2.0, -2.5]
MU_THS = [1, 2, 4, 8, 12, 16, 64]
DATASETS = ["alpaca", "gsm8k"]
HEADLINE_DRAFT = "pim"          # draft placement decision: always PIM
HEADLINE_SIGMA = -1.0
HEADLINE_MU = {"Standard": 1, "Low-power": 64}

# LP-Spec's own published Table III figures (not derived from our sim).
LP_SPEC_PUBLISHED = {"tps": 73.4, "tpj": 32.6, "edp": 0.418}


def _sigma_str(s: float) -> str:
    return f"−{abs(s):.1f}"  # unicode minus, matches main.tex's convention


def load_drive_all(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def load_pruning(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _index_capim(records: List[dict]) -> Dict[Tuple[str, float, int, str], dict]:
    out = {}
    for r in records:
        if r["driver"] != "capim":
            continue
        c = r["config"]
        out[(r["dataset"], r["collection_gate"], c["mu_th"], c["draft_device"])] = r
    return out


def _index_lp_spec(records: List[dict]) -> Dict[Tuple[str, int], dict]:
    return {(r["dataset"], r["config"]["L_spec"]): r
            for r in records if r["driver"] == "lp_spec"}


def _index_ar(records: List[dict]) -> Dict[str, dict]:
    return {r["dataset"]: r for r in records if r["driver"] == "ar"}


def gen_surface(records: List[dict]) -> str:
    capim = _index_capim(records)
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4.5pt}")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{7}{c}{μ\textsubscript{th}} \\")
    lines.append(r"\cmidrule(lr){2-8}")
    lines.append("σ\\textsubscript{th} & " + " & ".join(str(m) for m in MU_THS) + r" \\")
    lines.append(r"\midrule")
    for i, ds in enumerate(DATASETS):
        label = "Alpaca" if ds == "alpaca" else "GSM8K"
        lines.append(r"\multicolumn{8}{l}{\emph{%s}} \\" % label)
        for s in SIGMAS:
            cells = []
            for mu in MU_THS:
                r = capim[(ds, s, mu, HEADLINE_DRAFT)]
                cells.append(f"{r['token_per_s_mean']:.1f} / {r['token_per_j_mean']:.1f}")
            lines.append(f"{_sigma_str(s)} & " + " & ".join(cells) + r" \\")
        if i == 0:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("\\caption{The joint threshold sweep. Each cell gives throughput in token/s and "
                 "energy efficiency in token/J for one (σ\\textsubscript{th}, "
                 "μ\\textsubscript{th}) pair.}")
    lines.append(r"\label{tab:surface}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_trees(pruning: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{Alpaca} & \multicolumn{3}{c}{GSM8K} \\")
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    lines.append("σ\\textsubscript{th} & median & mean & largest & median & mean & largest \\\\")
    lines.append(r"\midrule")
    for s in SIGMAS:
        a = pruning["alpaca"]["causal_sweep"][str(s)]
        g = pruning["gsm8k"]["causal_sweep"][str(s)]
        lines.append(
            f"{_sigma_str(s)} & {int(a['mu_p50'])} & {a['mean_tree_size']:.2f} & {int(a['mu_max'])} & "
            f"{int(g['mu_p50'])} & {g['mean_tree_size']:.2f} & {int(g['mu_max'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{The draft tree size in nodes after confidence-based pruning}")
    lines.append(r"\label{tab:trees}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _best_lp_spec(records: List[dict], ds: str) -> dict:
    lp = [r for r in records if r["driver"] == "lp_spec" and r["dataset"] == ds]
    return max(lp, key=lambda r: r["token_per_s_mean"])


def gen_headline_values(records: List[dict]) -> str:
    capim = _index_capim(records)
    ar = _index_ar(records)["alpaca"]
    best = _best_lp_spec(records, "alpaca")

    rows = []
    for label, mu in HEADLINE_MU.items():
        r = capim[("alpaca", HEADLINE_SIGMA, mu, HEADLINE_DRAFT)]
        rows.append((f"CAPIM -- {label}", r["token_per_s_mean"], r["token_per_j_mean"], r["edp_mean"]))
    rows.append(("LP-Spec (simulated, best)", best["token_per_s_mean"], best["token_per_j_mean"], best["edp_mean"]))
    rows.append(("LP-Spec (published)", LP_SPEC_PUBLISHED["tps"], LP_SPEC_PUBLISHED["tpj"], LP_SPEC_PUBLISHED["edp"]))
    rows.append(("Autoregressive", ar["token_per_s_mean"], ar["token_per_j_mean"], ar["edp_mean"]))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r" & Throughput (token/s) & Energy efficiency (token/J) & EDP (s\textperiodcentered mJ) \\")
    lines.append(r"\midrule")
    for name, tps, tpj, edp in rows:
        lines.append(f"{name} & {tps:.1f} & {tpj:.1f} & {edp:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{The values underlying Figure~\ref{fig:headline}, including "
                 r"energy-delay product (EDP), Alpaca.}")
    lines.append(r"\label{tab:headline_values}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def gen_headline_ratio(records: List[dict]) -> str:
    capim = _index_capim(records)
    ar = _index_ar(records)["alpaca"]
    best = _best_lp_spec(records, "alpaca")

    def ratio_pair(r, baseline_tps, baseline_tpj) -> str:
        return f"{r['token_per_s_mean'] / baseline_tps:.2f}x, {r['token_per_j_mean'] / baseline_tpj:.2f}x"

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"CAPIM & LP-Spec (simulated, best) & LP-Spec (published) & Autoregressive \\")
    lines.append(r"\midrule")
    for label, mu in HEADLINE_MU.items():
        r = capim[("alpaca", HEADLINE_SIGMA, mu, HEADLINE_DRAFT)]
        row_label = f"{label} (μ\\textsubscript{{th}} = {mu})"
        vs_best = ratio_pair(r, best["token_per_s_mean"], best["token_per_j_mean"])
        vs_pub = ratio_pair(r, LP_SPEC_PUBLISHED["tps"], LP_SPEC_PUBLISHED["tpj"])
        vs_ar = ratio_pair(r, ar["token_per_s_mean"], ar["token_per_j_mean"])
        lines.append(f"{row_label} & {vs_best} & {vs_pub} & {vs_ar} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{CAPIM's two operating points against LP-Spec and autoregressive "
                 r"baselines, as ratios (throughput, energy efficiency), Alpaca.}")
    lines.append(r"\label{tab:headline_ratio}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-all", default="results/drive_all.json")
    ap.add_argument("--pruning", default="results/pruning_analysis.json")
    ap.add_argument("--table", choices=["surface", "trees", "headline_values", "headline_ratio", "all"],
                     default="all")
    args = ap.parse_args(argv)

    records = load_drive_all(args.drive_all)
    generators = {
        "surface": lambda: gen_surface(records),
        "trees": lambda: gen_trees(load_pruning(args.pruning)),
        "headline_values": lambda: gen_headline_values(records),
        "headline_ratio": lambda: gen_headline_ratio(records),
    }
    tables = generators.keys() if args.table == "all" else [args.table]
    for i, t in enumerate(tables):
        if i:
            print()
        print(f"% --- {t} ---")
        print(generators[t]())


if __name__ == "__main__":
    main()
