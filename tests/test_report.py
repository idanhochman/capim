"""
GPU-free smoke tests for common/report.py — the three properties it guarantees:
  1. token-WEIGHTED headline means (ratio of averaged totals == Sigma/Sigma),
     not a mean of per-prompt rates;
  2. SAMPLE std (n-1), not population;
  3. per-token EDP + a separate end-to-end latency that adds the one-time prefill.

Run from the repo root:
  python3 -m tests.test_report
"""

from __future__ import annotations

import math

from common.report import aggregate_by_prompt, comparison_table, summarize
from common.system import DriverResult, StepRecord


def _two_prompt_result() -> DriverResult:
    # prompt 0: two steps (7 tok, 0.09 s, 0.0038 J); prompt 1: one step (10 tok, ...)
    steps = [
        StepRecord(prompt_id=0, dataset="alpaca", step_id=0, tokens_emitted=4.0,
                   time_s=0.05, energy_j=0.002,
                   time_by_device={"NPU": 0.03, "PIM": 0.02},
                   energy_by_component={"off_mem": 0.001, "alu": 0.001},
                   time_by_type={"FC": 0.03, "MATMUL": 0.01, "NORM": 0.01}, crossings=2),
        StepRecord(prompt_id=0, dataset="alpaca", step_id=1, tokens_emitted=3.0,
                   time_s=0.04, energy_j=0.0018,
                   time_by_device={"NPU": 0.025, "PIM": 0.015},
                   energy_by_component={"off_mem": 0.001, "alu": 0.0008},
                   time_by_type={"FC": 0.025, "MATMUL": 0.01, "NORM": 0.005}, crossings=2),
        StepRecord(prompt_id=1, dataset="alpaca", step_id=0, tokens_emitted=10.0,
                   time_s=0.12, energy_j=0.006,
                   time_by_device={"NPU": 0.07, "PIM": 0.05},
                   energy_by_component={"off_mem": 0.003, "alu": 0.003},
                   time_by_type={"FC": 0.07, "MATMUL": 0.03, "NORM": 0.02}, crossings=2),
    ]
    return DriverResult(driver="CAPIM", model="Vicuna-7B-v1.3", steps=steps,
                        prefill_time_s=0.015, prefill_energy_j=0.05)


def test_token_weighted_not_mean_of_rates():
    r = _two_prompt_result()
    s = summarize(r)
    tot_tok, tot_time, tot_e = 17.0, 0.21, 0.0098
    assert math.isclose(s.token_per_s_mean, tot_tok / tot_time, rel_tol=1e-9)
    assert math.isclose(s.token_per_j_mean, tot_tok / tot_e, rel_tol=1e-9)
    # a naive mean-of-rates would differ (it weights both prompts equally)
    aggs = aggregate_by_prompt(r)
    mean_of_rates = sum(a.token_per_s for a in aggs) / len(aggs)
    assert not math.isclose(s.token_per_s_mean, mean_of_rates, rel_tol=1e-3)


def test_sample_std_not_population():
    r = _two_prompt_result()
    s = summarize(r)
    aggs = aggregate_by_prompt(r)
    rates = [a.token_per_s for a in aggs]
    m = s.token_per_s_mean
    pop = math.sqrt(sum((x - m) ** 2 for x in rates) / len(rates))
    samp = math.sqrt(sum((x - m) ** 2 for x in rates) / (len(rates) - 1))
    assert math.isclose(s.token_per_s_std, samp, rel_tol=1e-9)
    assert not math.isclose(s.token_per_s_std, pop, rel_tol=1e-6)


def test_edp_per_token_and_end_to_end_latency():
    r = _two_prompt_result()
    s = summarize(r)
    # per-token EDP = (mean_time/mean_tok) * (mean_energy_mJ/mean_tok)
    exp_edp = (s.mean_time_s / s.mean_tokens) * (s.mean_energy_j * 1e3 / s.mean_tokens)
    assert math.isclose(s.edp_mean, exp_edp, rel_tol=1e-9)
    # end-to-end latency adds the one-time prefill; decode-only excludes it
    assert math.isclose(s.end_to_end_latency_s, s.mean_time_s + 0.015, rel_tol=1e-9)
    assert s.end_to_end_latency_s > s.mean_time_s


def test_comparison_table_renders():
    r = _two_prompt_result()
    out = comparison_table([r], baseline_driver="CAPIM")
    assert "token/s" in out and "end-to-end latency" in out


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} report tests passed.")


if __name__ == "__main__":
    _run_all()
