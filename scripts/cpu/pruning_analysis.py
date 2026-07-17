"""Section 5.2 — draft-tree behaviour under a confidence gate.

Computes every number the results chapter's pruning section quotes, from the traces
on disk.  CPU-only, no model, no cost model: this section is deliberately
hardware-free, so it reports work in NODES and tokens per iteration, never joules.

Two trace families, with different roles (see Ch. 4):

  * ungated  (eagle_{ds}_s-inf.json)  -- the measurement instrument.  Every node was
    genuinely submitted to the target, so `accepted` is a real verification outcome.
    This is the ONLY trace in which we can see what a gate would have thrown away:
    in a gated run a pruned node is never verified and carries no label.
  * causal   (eagle_{ds}_s-{sigma}.json) -- the ground truth for what a gated system
    actually produces.  The gate fired inside the decode loop, so a pruned node
    genuinely changed the continuation.

The static sweep (sigma applied to the ungated trace after the fact) is a
counterfactual; `agreement()` licenses it by checking it against the causal runs.

Usage:
    .venv/bin/python scripts/cpu/pruning_analysis.py [--json results/pruning_analysis.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.schema import DecodeStep, Trace              # noqa: E402
from capim_ctrl.driver import _effective_accept          # noqa: E402  (the gate's own truncation)

DATASETS = ("alpaca", "gsm8k")
CAUSAL_SIGMAS = (-0.5, -1.0, -1.5, -2.0, -2.5)
SWEEP_SIGMAS = (-0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0, -2.5, -3.0, -4.0, -6.0)
AR_TOKENS_PER_ITER = 1.0          # autoregressive: one token per iteration, by construction
SIGMA_OP = -1.0                   # the operating point the chords are drawn from

# Dense grid for the saturation curve: fine through the knee, coarser down the tail,
# far enough (-12) that the gate stops firing and the curve lands on the ungated point.
CURVE_SIGMAS = tuple(
    [round(-0.1 - 0.1 * i, 2) for i in range(30)]        # -0.1 .. -3.0
    + [round(-3.0 - 0.25 * i, 2) for i in range(1, 29)]  # -3.25 .. -10.0
    + [-11.0, -12.0]
)


def trace_path(ds: str, sigma: float | None) -> Path:
    """sigma=None -> the ungated (s-inf) trace."""
    if sigma is None:
        return REPO / "traces" / f"eagle_{ds}_s-inf.json"
    s = f"{sigma:g}".rstrip("0").rstrip(".") if sigma != int(sigma) else f"{int(sigma)}"
    return REPO / "traces" / f"eagle_{ds}_s{s}.json"


def kept(step: DecodeStep, sigma: float) -> list:
    return [n for n in step.nodes if n.cumulative_log_prob >= sigma]


# ---------------------------------------------------------------------------
# 5.2.1 -- what a fixed draft budget returns
# ---------------------------------------------------------------------------
def fixed_budget(tr: Trace) -> dict:
    """The ungated run: acceptance density, and the structural ceiling that bounds it."""
    n_steps = len(tr.steps)
    nodes = sum(s.tree_size for s in tr.steps)
    accepted = sum(1 for s in tr.steps for n in s.nodes if n.accepted)
    depth = max(s.max_depth for s in tr.steps) + 1        # levels, not 0-based index
    width = nodes / n_steps

    # Only ONE root->leaf path is accepted per iteration, so at most `depth` nodes of a
    # `width`-node tree can ever be accepted.  Most of the rejection rate is that bound.
    ceiling = depth / width
    density = accepted / nodes
    return {
        "steps": n_steps,
        "nodes_verified": nodes,
        "nodes_accepted": accepted,
        "mean_tree_size": width,
        "max_depth": depth,
        "acceptance_density": density,
        "rejection_rate": 1.0 - density,
        "structural_ceiling": ceiling,
        "rejection_structural_pts": 1.0 - ceiling,        # unavoidable share, in points
        "rejection_avoidable_pts": ceiling - density,     # the part a gate can address
        "tokens_per_iter": sum(s.accepted_length for s in tr.steps) / n_steps + 1.0,
    }


def marginal_yield(causal: dict, ungated: dict, knee_sigma: float = SIGMA_OP) -> dict:
    """Tokens returned per verified node, for the first few nodes vs the long tail.

    Causal, not counterfactual: both endpoints are real gated runs.  The AR point
    (mu=0, tokens/iter=1.0) is exact by construction, so the first segment's slope is
    the yield of the nodes the gate actually keeps.
    """
    lo = causal[knee_sigma]
    mu_lo, tpi_lo = lo["mean_tree_size"], lo["tokens_per_iter"]
    mu_hi, tpi_hi = ungated["mean_tree_size"], ungated["tokens_per_iter"]

    head = (tpi_lo - AR_TOKENS_PER_ITER) / mu_lo          # nodes 1 .. mu_lo
    tail = (tpi_hi - tpi_lo) / (mu_hi - mu_lo)            # nodes mu_lo .. 59
    return {
        "knee_sigma": knee_sigma,
        "head_mu": mu_lo, "head_tokens_per_iter": tpi_lo, "head_yield": head,
        "tail_mu": mu_hi, "tail_tokens_per_iter": tpi_hi, "tail_yield": tail,
        "collapse": head / tail,
    }


# ---------------------------------------------------------------------------
# 5.2.2 -- how much can you prune, and what does it cost?
# ---------------------------------------------------------------------------
def static_sweep(tr: Trace, sigmas=SWEEP_SIGMAS) -> list[dict]:
    """Apply sigma to the UNGATED trace after the fact.

    Counterfactual, and unavoidably so: it is the only way to see the acceptance
    outcome of a node a gate would have removed.  Licensed by agreement() below.
    """
    n_steps = len(tr.steps)
    tot_nodes = sum(s.tree_size for s in tr.steps)
    tot_accepted = sum(s.accepted_length for s in tr.steps)
    tot_tpi = tot_accepted + n_steps                      # + bonus token per iteration

    rows = []
    for sig in sigmas:
        n_kept = n_kept_acc = eff_accept = 0
        for s in tr.steps:
            k = kept(s, sig)
            n_kept += len(k)
            n_kept_acc += sum(1 for n in k if n.accepted)
            eff_accept += _effective_accept(s, sig)

        n_dropped = tot_nodes - n_kept
        dropped_acc = tot_accepted - n_kept_acc           # accepted nodes the gate removed
        eff_tpi = eff_accept + n_steps

        rows.append({
            "sigma": sig,
            "mean_tree_size": n_kept / n_steps,
            "nodes_kept_frac": n_kept / tot_nodes,
            "nodes_removed_frac": n_dropped / tot_nodes,
            # of everything the gate threw away, how much was never going to be accepted
            "discarded_doomed_frac": (n_dropped - dropped_acc) / n_dropped if n_dropped else float("nan"),
            # useful work per verified node, among the nodes we keep
            "kept_acceptance_density": n_kept_acc / n_kept if n_kept else float("nan"),
            # the two headline axes: work saved vs tokens lost
            "accepted_retained_frac": eff_accept / tot_accepted,
            "tokens_per_iter": eff_tpi / n_steps,
            "tokens_per_iter_retained_frac": eff_tpi / tot_tpi,
            "work_saving_x": tot_nodes / n_kept if n_kept else float("inf"),
            "throughput_cost_x": tot_tpi / eff_tpi if eff_tpi else float("inf"),
        })
    return rows


def causal_sweep(ds: str, sigmas=CAUSAL_SIGMAS) -> dict:
    """The real gated runs: mu and tokens/iteration, plus the per-iteration mu spread."""
    out = {}
    for sig in sigmas:
        tr = Trace.load(str(trace_path(ds, sig)))
        mus = [s.tree_size for s in tr.steps]
        tpi = [s.accepted_length + 1 for s in tr.steps]
        out[sig] = {
            "sigma": sig,
            "steps": len(tr.steps),
            "mean_tree_size": statistics.fmean(mus),
            "mu_std": statistics.pstdev(mus),
            "mu_p50": statistics.median(mus),
            "mu_p99": sorted(mus)[max(0, math.ceil(0.99 * len(mus)) - 1)],
            "mu_max": max(mus),
            "tokens_per_iter": statistics.fmean(tpi),
            "mean_accepted_length": statistics.fmean([s.accepted_length for s in tr.steps]),
        }
    return out


def agreement(static_rows: list[dict], causal: dict, ungated_tpi: float) -> list[dict]:
    """Licence for the static sweep: does the counterfactual match the real gated run?

    Compared on tokens/iteration retention, which both views can produce.
    """
    by_sigma = {r["sigma"]: r for r in static_rows}
    rows = []
    for sig, c in causal.items():
        if sig not in by_sigma:
            continue
        s = by_sigma[sig]
        pred = s["tokens_per_iter_retained_frac"]
        meas = c["tokens_per_iter"] / ungated_tpi
        rows.append({
            "sigma": sig,
            "static_predicted_retention": pred,
            "causal_measured_retention": meas,
            "abs_error_pts": abs(pred - meas) * 100.0,
            "static_mu": s["mean_tree_size"],
            "causal_mu": c["mean_tree_size"],
        })
    return rows


# ---------------------------------------------------------------------------
# 5.2.3 -- is the gate signal worth what it costs?
# ---------------------------------------------------------------------------
def depth_truncation(tr: Trace, max_depth: int) -> dict:
    """The free gate: cut every branch at depth d.  No score, no softmax, no crossing.

    This is what BiLD / Kangaroo do.  Its accepted-token retention is the bar the
    cumulative-confidence gate has to clear to justify computing a score at all.
    """
    n_steps = len(tr.steps)
    tot_nodes = sum(s.tree_size for s in tr.steps)
    tot_accepted = sum(s.accepted_length for s in tr.steps)

    n_kept = eff = 0
    for s in tr.steps:
        n_kept += sum(1 for n in s.nodes if n.depth <= max_depth)
        eff += min(s.accepted_length, max_depth + 1)      # accepted chain truncated at depth d
    return {
        "rule": f"depth <= {max_depth}",
        "nodes_kept_frac": n_kept / tot_nodes,
        "accepted_retained_frac": eff / tot_accepted,
        "tokens_per_iter": (eff + n_steps) / n_steps,
    }


def signal_value(tr: Trace, static_rows: list[dict]) -> dict:
    """Accepted-token retention at a matched node budget: confidence vs depth vs random.

    Random needs no experiment: selecting a fraction f of nodes blindly retains f of the
    accepted ones in expectation.  It is the null, and it is exact.
    """
    depth_rows = [depth_truncation(tr, d) for d in range(0, 6)]

    out = []
    for d in depth_rows:
        budget = d["nodes_kept_frac"]
        # the confidence gate at the SAME node budget.  static_rows runs sigma high->low,
        # so nodes_kept_frac is already ascending -- interpolate on it directly.
        xs = [r["nodes_kept_frac"] for r in static_rows]
        ys = [r["accepted_retained_frac"] for r in static_rows]
        conf = _interp(budget, xs, ys)
        out.append({
            "budget_nodes_frac": budget,
            "depth_rule": d["rule"],
            "depth_retention": d["accepted_retained_frac"],
            "confidence_retention": conf,
            "random_retention": budget,                    # the null, exact
            "conf_vs_depth_x": conf / d["accepted_retained_frac"] if d["accepted_retained_frac"] else float("nan"),
            "conf_vs_random_x": conf / budget if budget else float("nan"),
        })
    return {"matched_budgets": out}


def _interp(x: float, xs: list[float], ys: list[float]) -> float:
    """Linear interpolation on an ascending-xs curve (no numpy dependency here)."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


# ---------------------------------------------------------------------------
def analyse(ds: str) -> dict:
    ungated = Trace.load(str(trace_path(ds, None)))
    fb = fixed_budget(ungated)
    static = static_sweep(ungated)
    causal = causal_sweep(ds)
    return {
        "dataset": ds,
        "fixed_budget": fb,
        "marginal_yield": marginal_yield(causal, fb),
        "static_sweep": static,
        # dense (mu, tokens/iter) for the saturation curve.  Counterfactual, and slightly
        # optimistic (~3% at the operating point) -- the causal points are plotted on it.
        "static_curve": [
            {k: r[k] for k in ("sigma", "mean_tree_size", "tokens_per_iter")}
            for r in static_sweep(ungated, CURVE_SIGMAS)
        ],
        "causal_sweep": {str(k): v for k, v in causal.items()},
        "agreement": agreement(static, causal, fb["tokens_per_iter"]),
        "signal_value": signal_value(ungated, static),
    }


def report(a: dict) -> None:
    ds, fb, my = a["dataset"], a["fixed_budget"], a["marginal_yield"]
    print(f"\n{'='*78}\n{ds.upper()}\n{'='*78}")

    print("\n-- 5.2.1  what a fixed draft budget returns ------------------------------")
    print(f"  steps {fb['steps']}, nodes verified {fb['nodes_verified']}, accepted {fb['nodes_accepted']}")
    print(f"  tree size mu = {fb['mean_tree_size']:.1f}, depth = {fb['max_depth']}, tokens/iter = {fb['tokens_per_iter']:.3f}")
    print(f"  acceptance density {fb['acceptance_density']*100:.1f}%  -> rejection {fb['rejection_rate']*100:.1f}%")
    print(f"  STRUCTURAL CEILING {fb['structural_ceiling']*100:.1f}% "
          f"({fb['max_depth']}/{fb['mean_tree_size']:.0f}: one accepted path per iteration)")
    print(f"    of the {fb['rejection_rate']*100:.1f}% rejected, "
          f"{fb['rejection_structural_pts']*100:.1f} pts are structurally unavoidable, "
          f"{fb['rejection_avoidable_pts']*100:.1f} pts addressable")
    print(f"  MARGINAL YIELD  first {my['head_mu']:.1f} nodes: {my['head_yield']:.3f} tokens/node")
    print(f"                  nodes {my['head_mu']:.1f}-{my['tail_mu']:.0f}: {my['tail_yield']:.3f} tokens/node")
    print(f"                  -> collapse {my['collapse']:.1f}x")

    print("\n-- 5.2.2  how much can you prune, and what does it cost? -----------------")
    print(f"  {'sigma':>6} {'mu':>6} {'kept':>7} {'removed':>8} {'doomed':>8} {'density':>8} "
          f"{'acc.ret':>8} {'tok/iter':>9} {'work':>7} {'cost':>6}")
    for r in a["static_sweep"]:
        print(f"  {r['sigma']:>6.2f} {r['mean_tree_size']:>6.2f} {r['nodes_kept_frac']*100:>6.1f}% "
              f"{r['nodes_removed_frac']*100:>7.1f}% {r['discarded_doomed_frac']*100:>7.1f}% "
              f"{r['kept_acceptance_density']*100:>7.1f}% {r['accepted_retained_frac']*100:>7.1f}% "
              f"{r['tokens_per_iter']:>9.3f} {r['work_saving_x']:>6.1f}x {r['throughput_cost_x']:>5.2f}x")
    print("  (work = fewer nodes verified; cost = fewer tokens/iteration. NODES, not joules.)")

    print("\n  causal gated runs (ground truth):")
    print(f"  {'sigma':>6} {'mu':>6} {'mu sd':>6} {'mu p99':>7} {'mu max':>7} {'tok/iter':>9}")
    for _, c in a["causal_sweep"].items():
        print(f"  {c['sigma']:>6.2f} {c['mean_tree_size']:>6.2f} {c['mu_std']:>6.2f} "
              f"{c['mu_p99']:>7.0f} {c['mu_max']:>7.0f} {c['tokens_per_iter']:>9.3f}")

    print("\n  licence for the static view (counterfactual vs the real gated run):")
    print(f"  {'sigma':>6} {'static':>8} {'causal':>8} {'error':>8}")
    for r in a["agreement"]:
        print(f"  {r['sigma']:>6.2f} {r['static_predicted_retention']*100:>7.1f}% "
              f"{r['causal_measured_retention']*100:>7.1f}% {r['abs_error_pts']:>6.1f} pts")

    print("\n-- 5.2.3  is the gate signal worth what it costs? -------------------------")
    print(f"  {'budget':>8} {'depth rule':>12} {'depth':>8} {'confidence':>11} {'random':>8} "
          f"{'vs depth':>9} {'vs random':>10}")
    for r in a["signal_value"]["matched_budgets"]:
        print(f"  {r['budget_nodes_frac']*100:>7.1f}% {r['depth_rule']:>12} "
              f"{r['depth_retention']*100:>7.1f}% {r['confidence_retention']*100:>10.1f}% "
              f"{r['random_retention']*100:>7.1f}% {r['conf_vs_depth_x']:>8.2f}x {r['conf_vs_random_x']:>9.2f}x")
    print("  (accepted-token retention at a matched node budget. depth truncation is the")
    print("   free gate -- no score, no softmax, no PIM->NPU crossing -- so it is the bar.)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="results/pruning_analysis.json")
    args = p.parse_args()

    out = {ds: analyse(ds) for ds in DATASETS}
    for ds in DATASETS:
        report(out[ds])

    dest = REPO / args.json
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
