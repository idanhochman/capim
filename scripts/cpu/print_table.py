#!/usr/bin/env python3
"""Print a comparison table from one or more `main.py drive` JSON output files."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


_KEY_ABBREV = {
    "sigma_th": "sigma",
    "draft_device": "draft",
    "mu_th": "mu",
    "concurrent_verify": "cv",
    "all_npu": "allnpu",
    "L_spec": "L",
    "medusa_num_heads": "heads",
}


def _config_str(config: Dict[str, Any]) -> str:
    if not config:
        return "-"
    parts = []
    for k, v in config.items():
        if isinstance(v, bool):
            v = int(v)
        parts.append(f"{_KEY_ABBREV.get(k, k)}={v}")
    return " ".join(parts)


def load_records(paths: List[str]) -> List[dict]:
    records: List[dict] = []
    for path in paths:
        with open(path) as f:
            records.extend(json.load(f))
    return records


def format_table(records: List[dict]) -> str:
    rows = []
    for r in records:
        rows.append((
            r["driver"], r["dataset"], _config_str(r.get("config", {})),
            f"{r['token_per_s_mean']:.2f}±{r['token_per_s_std']:.2f}",
            f"{r['token_per_j_mean']:.2f}±{r['token_per_j_std']:.2f}",
            f"{r['edp_mean']:.3g}",
            f"{r['end_to_end_latency_s'] * 1e3:.1f}",
            f"{r.get('mean_tree_size', 0.0):.1f}",
            f"{r.get('mean_accepted_length', 0.0):.2f}",
            f"{r.get('mean_acceptance_rate', 0.0):.1%}",
        ))
    cfg_w = max([len("config")] + [len(row[2]) for row in rows])

    header = (f"{'driver':<8} {'dataset':<8} {'config':<{cfg_w}} "
              f"{'tok/s':>14} {'tok/J':>14} {'EDP(s·mJ)':>12} {'e2e(ms)':>10} "
              f"{'tree':>6} {'acc_len':>8} {'acc_rate':>9}")
    lines = [header, "-" * len(header)]
    for driver, dataset, cfg, tps, tpj, edp, e2e, tree, alen, arate in rows:
        lines.append(f"{driver:<8} {dataset:<8} {cfg:<{cfg_w}} "
                     f"{tps:>14} {tpj:>14} {edp:>12} {e2e:>10} "
                     f"{tree:>6} {alen:>8} {arate:>9}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="JSON files written by `main.py drive`")
    args = ap.parse_args(argv)

    records = load_records(args.results)
    print(format_table(records))


if __name__ == "__main__":
    main()
