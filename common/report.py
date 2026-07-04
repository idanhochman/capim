"""
Reporting + aggregation.

Aggregation unit = per prompt (group StepRecords by prompt_id).  For each prompt we
sum the decode-step TOTALS (tokens, time, energy); the headline metrics are then
token-WEIGHTED across prompts — i.e. the ratio of the averaged totals
(mean_tokens / mean_time), which equals Σtokens / Σtime — so a long prompt counts
for more than a short one.  The per-prompt spread is reported separately as a SAMPLE
std (n-1) over the per-prompt rates, for error bars only.

Metrics match LP-Spec Table III, all on the DECODE steps only (prefill excluded):
  - throughput        token/s   = tokens / time
  - energy efficiency token/J   = tokens / energy
  - EDP (per token)             = (time / tokens) * (energy_mJ / tokens)   [s·mJ]
        Computed per token so it is comparable across drivers that emit different
        token counts; lower = better.

Prefill is a one-time, per-sequence cost carried on DriverResult and reported only in
the secondary end-to-end latency metric; energy stays decode-only.

A trace covers one dataset (Alpaca or GSM8K); run the two datasets as separate traces
and compare their reports.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List

from common.system import DriverResult, StepRecord


@dataclass
class PromptAgg:
    prompt_id: int
    dataset: str
    tokens: float
    time_s: float
    energy_j: float

    @property
    def token_per_s(self) -> float:
        return self.tokens / self.time_s if self.time_s > 0 else 0.0

    @property
    def token_per_j(self) -> float:
        return self.tokens / self.energy_j if self.energy_j > 0 else 0.0

    @property
    def edp_s_mj(self) -> float:
        # Per-token energy-delay product so it is comparable across drivers that
        # emit different token counts: (s/token) * (mJ/token).
        if self.tokens <= 0 or self.time_s <= 0:
            return 0.0
        lat_per_token = self.time_s / self.tokens
        e_per_token_mj = (self.energy_j * 1e3) / self.tokens
        return lat_per_token * e_per_token_mj


def aggregate_by_prompt(result: DriverResult) -> List[PromptAgg]:
    # A trace covers exactly one dataset, so prompt_id is unique within it and we
    # bucket on prompt_id alone (no dataset slicing needed).
    buckets: Dict[int, PromptAgg] = {}
    for s in result.steps:
        a = buckets.get(s.prompt_id)
        if a is None:
            buckets[s.prompt_id] = PromptAgg(s.prompt_id, s.dataset, s.tokens_emitted,
                                             s.time_s, s.energy_j)
        else:
            a.tokens += s.tokens_emitted
            a.time_s += s.time_s
            a.energy_j += s.energy_j
    return list(buckets.values())


def _sample_std(xs: List[float], mean: float) -> float:
    """Sample std (Bessel-corrected, n-1).  0 for fewer than two points."""
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class Summary:
    driver: str
    dataset: str
    n_prompts: int
    # Headline figures are token-weighted (ratio of averaged per-prompt totals);
    # the std is the per-prompt sample spread, for error bars only.
    token_per_s_mean: float
    token_per_s_std: float
    token_per_j_mean: float
    token_per_j_std: float
    edp_mean: float
    edp_std: float
    # token-weighted totals (averaged per-prompt totals), decode-only
    mean_time_s: float
    mean_energy_j: float
    mean_tokens: float
    # secondary: end-to-end latency = mean per-prompt decode time + one-time prefill
    end_to_end_latency_s: float
    prefill_time_s: float


def summarize(result: DriverResult) -> Summary:
    aggs = aggregate_by_prompt(result)

    # token-weighted totals (decode-only): averaging per-prompt totals across prompts
    # makes the ratios below equal Σtokens/Σtime and Σtokens/Σenergy.
    t_m = _mean([a.time_s for a in aggs])
    e_m = _mean([a.energy_j for a in aggs])
    tok_m = _mean([a.tokens for a in aggs])

    tps_mean = tok_m / t_m if t_m > 0 else 0.0
    tpj_mean = tok_m / e_m if e_m > 0 else 0.0
    edp_mean = ((t_m / tok_m) * (e_m * 1e3 / tok_m)) if tok_m > 0 and t_m > 0 else 0.0

    # per-prompt sample spread (error bars), about the token-weighted mean
    tps_std = _sample_std([a.token_per_s for a in aggs], tps_mean)
    tpj_std = _sample_std([a.token_per_j for a in aggs], tpj_mean)
    edp_std = _sample_std([a.edp_s_mj for a in aggs], edp_mean)

    ds = aggs[0].dataset if aggs else "unknown"
    return Summary(
        driver=result.driver, dataset=ds, n_prompts=len(aggs),
        token_per_s_mean=tps_mean, token_per_s_std=tps_std,
        token_per_j_mean=tpj_mean, token_per_j_std=tpj_std,
        edp_mean=edp_mean, edp_std=edp_std,
        mean_time_s=t_m, mean_energy_j=e_m, mean_tokens=tok_m,
        end_to_end_latency_s=t_m + result.prefill_time_s,
        prefill_time_s=result.prefill_time_s,
    )


def comparison_table(results: List[DriverResult],
                     baseline_driver: str = "LP-Spec") -> str:
    """Format a comparison table with speedup/efficiency vs a baseline driver."""
    summaries = [summarize(r) for r in results]
    base = next((s for s in summaries if s.driver == baseline_driver), summaries[0])

    rows = []
    header = f"{'Driver':<16} {'token/s':>14} {'token/J':>14} {'EDP(s·mJ)':>14} {'vs '+base.driver:>22}"
    rows.append(header)
    rows.append("-" * len(header))
    for s in summaries:
        spd = s.token_per_s_mean / base.token_per_s_mean if base.token_per_s_mean else 0.0
        eff = s.token_per_j_mean / base.token_per_j_mean if base.token_per_j_mean else 0.0
        edp_imp = base.edp_mean / s.edp_mean if s.edp_mean else 0.0
        rows.append(
            f"{s.driver:<16} "
            f"{s.token_per_s_mean:>10.1f}±{s.token_per_s_std:<3.0f} "
            f"{s.token_per_j_mean:>10.1f}±{s.token_per_j_std:<3.0f} "
            f"{s.edp_mean:>14.3g} "
            f"{spd:>6.2f}x sp {eff:>5.2f}x eff {edp_imp:>5.2f}x edp"
        )
    # secondary metric: end-to-end latency (decode + one-time prefill)
    rows.append("")
    rows.append("end-to-end latency (decode + prefill):")
    for s in summaries:
        rows.append(
            f"  {s.driver:<16} {s.end_to_end_latency_s * 1e3:>8.1f} ms "
            f"(decode {s.mean_time_s * 1e3:.1f} + prefill {s.prefill_time_s * 1e3:.1f})")
    return "\n".join(rows)


# --- per-component breakdown (PAPI Fig. 12 analog) -----------------------------
#
# Token-weighted decomposition of each driver's per-token cost into typed ops and
# energy components.  Diagnostic AND thesis figure: it shows WHERE latency/energy
# go (FC vs attention vs nonlinear vs PIM<->NPU communication) and, for the
# concurrent LP-Spec driver, where its column-split makespan sits in the
# [all-PIM, all-NPU] band -- the check on whether PIM gets compute-bound credit.
#
# NOTE on time_by_device: for sequential drivers (AR/CAPIM) NPU+PIM sums to the
# wall clock (a true partition); for the concurrent driver (LP-Spec) BOTH devices
# are charged the full makespan, so it is device-BUSY-time (utilization) and sums
# to MORE than wall clock.  The wall-clock latency partition is time_by_type.

# LayerType.name -> PAPI-style rollup bucket (FC / Attn / NL / Comm).
_TYPE_ROLLUP = {
    "FC": "FC", "MATMUL": "Attn",
    "SOFTMAX": "NL", "ACT": "NL", "NORM": "NL",
    "COMM": "Comm",
}
_ROLLUP_ORDER = ["FC", "Attn", "NL", "Comm"]


@dataclass
class Breakdown:
    driver: str
    dataset: str
    tokens: float
    time_s: float
    energy_j: float
    time_by_type: Dict[str, float]       # rolled-up wall-clock seconds
    time_by_device: Dict[str, float]     # device busy-seconds (>wallclock if concurrent)
    energy_by_component: Dict[str, float]
    crossings: float
    split_t_npu_bound: float             # all-NPU time of split kernels (concurrent)
    split_t_pim_bound: float             # all-PIM time of split kernels (concurrent)
    split_makespan: float                # actual FC+Attn time (= makespan if concurrent)


def breakdown(result: DriverResult) -> Breakdown:
    tokens = time_s = energy_j = crossings = 0.0
    s_npu = s_pim = 0.0
    ttype: Dict[str, float] = {}
    tdev: Dict[str, float] = {}
    ecomp: Dict[str, float] = {}
    dataset = "unknown"
    for st in result.steps:
        dataset = st.dataset
        tokens += st.tokens_emitted
        time_s += st.time_s
        energy_j += st.energy_j
        crossings += st.crossings
        s_npu += st.split_t_npu_bound
        s_pim += st.split_t_pim_bound
        for k, v in st.time_by_type.items():
            b = _TYPE_ROLLUP.get(k, k)
            ttype[b] = ttype.get(b, 0.0) + v
        for k, v in st.time_by_device.items():
            tdev[k] = tdev.get(k, 0.0) + v
        for k, v in st.energy_by_component.items():
            ecomp[k] = ecomp.get(k, 0.0) + v
    split_makespan = ttype.get("FC", 0.0) + ttype.get("Attn", 0.0)
    return Breakdown(result.driver, dataset, tokens, time_s, energy_j,
                     ttype, tdev, ecomp, crossings, s_npu, s_pim, split_makespan)


def breakdown_table(results: List[DriverResult]) -> str:
    """Per-driver, token-weighted latency/energy breakdown (PAPI Fig. 12 style)."""
    lines: List[str] = []
    for r in results:
        b = breakdown(r)
        if b.tokens <= 0:
            continue
        perk = lambda x: 1e3 * x / b.tokens   # seconds->ms or joules->mJ per token
        lines.append(
            f"### {b.driver}  ({b.dataset})  --  {perk(b.time_s):.1f} ms/token, "
            f"{perk(b.energy_j):.1f} mJ/token, {b.crossings / b.tokens:.1f} crossings/token")

        lines.append("  latency by op (wall-clock):")
        for k in _ROLLUP_ORDER:
            v = b.time_by_type.get(k, 0.0)
            if v:
                lines.append(f"    {k:<6}{perk(v):8.2f} ms/tok ({v / b.time_s:5.1%})")

        lines.append("  energy by component:")
        for k in ("off_mem", "on_chip", "alu", "comm"):
            v = b.energy_by_component.get(k, 0.0)
            if v:
                lines.append(f"    {k:<8}{perk(v):7.2f} mJ/tok ({v / b.energy_j:5.1%})")

        npu_t = b.time_by_device.get("NPU", 0.0)
        pim_t = b.time_by_device.get("PIM", 0.0)
        concurrent = b.split_t_npu_bound > 0
        tag = " (busy-time; >wallclock, concurrent)" if concurrent else ""
        lines.append(f"  device:  NPU {perk(npu_t):.2f}  PIM {perk(pim_t):.2f} ms/tok{tag}")

        if concurrent:
            r_eff = b.split_makespan / b.split_t_npu_bound if b.split_t_npu_bound else 0.0
            lines.append(
                f"  split FC+Attn makespan {perk(b.split_makespan):.2f} ms/tok in band "
                f"[all-PIM {perk(b.split_t_pim_bound):.2f}, all-NPU {perk(b.split_t_npu_bound):.2f}]"
                f"  (eff r={r_eff:.2f})")
        lines.append("")
    return "\n".join(lines)


def export_csv(results: List[DriverResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["driver", "dataset", "n_prompts", "token_per_s", "token_per_s_std",
                    "token_per_j", "token_per_j_std", "edp_s_mj", "edp_std",
                    "mean_time_s", "mean_energy_j", "mean_tokens",
                    "end_to_end_latency_s", "prefill_time_s"])
        for r in results:
            s = summarize(r)
            w.writerow([s.driver, s.dataset, s.n_prompts,
                        s.token_per_s_mean, s.token_per_s_std,
                        s.token_per_j_mean, s.token_per_j_std,
                        s.edp_mean, s.edp_std, s.mean_time_s, s.mean_energy_j, s.mean_tokens,
                        s.end_to_end_latency_s, s.prefill_time_s])
