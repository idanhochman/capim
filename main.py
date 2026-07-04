#!/usr/bin/env python3
"""
CLI entry point.

Two subcommands:
  collect   GPU-only. Runs instrumented inference (EAGLE or MEDUSA) over a prompt
            set and saves one Trace to a JSON file.
  drive     CPU-only. Loads trace(s), drives one or more drivers (ar / capim /
            lp_spec) over a sweep of their knobs, and writes one JSON list of
            per-run summary records.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from typing import Dict, List

from common.config import VICUNA_7B
from common.schema import Trace
from common.type import Device as Dev

NEG_INF = float("-inf")


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> None:
    import prompts as prompt_lib

    if args.sanity:
        raw_prompts = prompt_lib.SANITY_PROMPTS
        dataset = "sanity"
    else:
        dataset = args.dataset
        raw_prompts = prompt_lib.load_prompts(dataset, args.n_prompts)
    formatted = [prompt_lib.format_vicuna_prompt(p) for p in raw_prompts]

    load_in_4bit = args.precision == "int4"
    load_in_8bit = args.precision == "int8"

    if args.method == "eagle":
        from capim_ctrl.collector import collect, load_eagle_model

        model, tokenizer = load_eagle_model(
            args.base_model, args.ea_model,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
        )
        trace = collect(
            model, tokenizer, formatted,
            dataset=dataset, sigma_th=args.sigma_th,
            max_new_tokens=args.max_new_tokens,
            draft_head=args.ea_model,
        )
    else:
        from baselines.lp_spec.collector import collect, load_medusa_model

        model, tokenizer = load_medusa_model(
            args.base_model, args.medusa_model,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit,
        )
        trace = collect(
            model, tokenizer, formatted,
            dataset=dataset, L=args.L, selection="greedy_headk",
            max_new_tokens=args.max_new_tokens,
            draft_head=args.medusa_model,
        )

    trace.save(args.out)
    print(f"wrote {args.out}  ({len(trace.steps)} steps, "
          f"mean_accepted_length={trace.mean_accepted_length:.2f}, "
          f"mean_acceptance_rate={trace.mean_acceptance_rate:.2%})")


# --------------------------------------------------------------------------
# drive
# --------------------------------------------------------------------------

def _trace_stats(trace: Trace) -> dict:
    """Trace-level summary stats echoed into every drive record (for the table).
    These describe the underlying trace, not the driver run, so they are identical
    across a driver's knob sweep on the same trace."""
    return dict(
        mean_tree_size=trace.mean_tree_size,
        mean_accepted_length=trace.mean_accepted_length,
        mean_acceptance_rate=trace.mean_acceptance_rate,
    )


def _drive_ar(trace: Trace, trace_path: str, trace_source: str) -> List[dict]:
    from baselines import autoregressive
    from common.report import summarize

    result = autoregressive.drive(VICUNA_7B, trace)
    record = asdict(summarize(result))
    record.update(driver="ar", trace=trace_path, trace_source=trace_source, config={},
                  **_trace_stats(trace))
    return [record]


def _drive_capim(trace: Trace, trace_path: str, args: argparse.Namespace) -> List[dict]:
    from capim_ctrl import driver as capim
    from common.report import summarize

    records: List[dict] = []
    draft_cache_by_key: Dict[tuple, dict] = {}
    device_of = {"npu": Dev.NPU, "pim": Dev.PIM}
    for sigma_th, draft_device_name, mu_th in itertools.product(
        args.sigma_th, args.draft_device, args.mu_th
    ):
        draft_device = device_of[draft_device_name]
        cache = draft_cache_by_key.setdefault((sigma_th, draft_device), {})
        cfg = capim.CapimConfig(
            sigma_th=sigma_th, mu_th=mu_th,
            all_npu=args.all_npu,
            concurrent_verify=not args.sequential_verify,
            draft_device=draft_device,
        )
        result = capim.drive(VICUNA_7B, trace, cfg, draft_cache=cache)
        record = asdict(summarize(result))
        record.update(
            driver="capim", trace=trace_path, trace_source="eagle",
            config=dict(sigma_th=sigma_th, draft_device=draft_device_name, mu_th=mu_th,
                        concurrent_verify=cfg.concurrent_verify, all_npu=args.all_npu),
            **_trace_stats(trace),
        )
        records.append(record)
    return records


def _drive_lp_spec(trace: Trace, trace_path: str, args: argparse.Namespace) -> List[dict]:
    from baselines.lp_spec import driver as lp_spec
    from common.report import summarize

    records: List[dict] = []
    for L_spec in args.L_spec:
        cfg = lp_spec.LPSpecConfig(L_spec=L_spec, medusa_num_heads=args.medusa_num_heads)
        result = lp_spec.drive(VICUNA_7B, trace, cfg)
        record = asdict(summarize(result))
        record.update(
            driver="lp_spec", trace=trace_path, trace_source="medusa",
            config=dict(L_spec=L_spec, medusa_num_heads=args.medusa_num_heads),
            **_trace_stats(trace),
        )
        records.append(record)
    return records


def cmd_drive(args: argparse.Namespace) -> None:
    if "capim" in args.driver and not args.eagle_trace:
        raise SystemExit("--driver capim requires --eagle-trace")
    if "lp_spec" in args.driver and not args.medusa_trace:
        raise SystemExit("--driver lp_spec requires --medusa-trace")

    eagle_trace = Trace.load(args.eagle_trace) if args.eagle_trace else None
    medusa_trace = Trace.load(args.medusa_trace) if args.medusa_trace else None

    records: List[dict] = []
    if "ar" in args.driver:
        if eagle_trace is not None:
            records += _drive_ar(eagle_trace, args.eagle_trace, "eagle")
        if medusa_trace is not None:
            records += _drive_ar(medusa_trace, args.medusa_trace, "medusa")
    if "capim" in args.driver:
        records += _drive_capim(eagle_trace, args.eagle_trace, args)
    if "lp_spec" in args.driver:
        records += _drive_lp_spec(medusa_trace, args.medusa_trace, args)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {args.out}  ({len(records)} runs)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("collect", help="GPU: run instrumented inference, save a trace")
    pc.add_argument("--method", choices=["eagle", "medusa"], required=True)
    pc.add_argument("--dataset", choices=["alpaca", "gsm8k"], default="alpaca")
    pc.add_argument("--sanity", action="store_true",
                    help="use 5 built-in prompts instead of downloading a dataset")
    pc.add_argument("--n-prompts", type=int, default=100)
    pc.add_argument("--max-new-tokens", type=int, default=200)
    pc.add_argument("--sigma-th", type=float, default=-1.5,
                    help="EAGLE cumulative-log-prob gate threshold (--method eagle); "
                         "-inf records the ungated full tree")
    pc.add_argument("--L", type=int, default=4,
                    help="MEDUSA DTP keep count (--method medusa)")
    pc.add_argument("--base-model", default="lmsys/vicuna-7b-v1.3")
    pc.add_argument("--ea-model", default="yuhuili/EAGLE-Vicuna-7B-v1.3")
    pc.add_argument("--medusa-model", default="FasterDecoding/medusa-vicuna-7b-v1.3")
    pc.add_argument("--precision", choices=["int8", "int4", "fp16"], default="int8")
    pc.add_argument("--out", required=True)
    pc.set_defaults(func=cmd_collect)

    pd = sub.add_parser("drive", help="CPU: re-cost trace(s) through one or more drivers")
    pd.add_argument("--eagle-trace", default=None)
    pd.add_argument("--medusa-trace", default=None)
    pd.add_argument("--driver", nargs="+", choices=["ar", "capim", "lp_spec"], required=True)
    # capim sweep knobs
    pd.add_argument("--mu-th", type=int, nargs="+", default=[4])
    pd.add_argument("--draft-device", choices=["npu", "pim"], nargs="+", default=["pim"])
    pd.add_argument("--sigma-th", type=float, nargs="+", default=[NEG_INF],
                    help="re-gate threshold; default -inf assumes the trace is "
                         "already gated by the collector")
    pd.add_argument("--all-npu", action="store_true",
                    help="CAPIM ablation: force the all-NPU route regardless of mu_th")
    pd.add_argument("--sequential-verify", action="store_true",
                    help="disable the mu>=mu_th concurrent verify route")
    # lp_spec sweep knobs
    pd.add_argument("--L-spec", type=int, nargs="+", default=[4])
    pd.add_argument("--medusa-num-heads", type=int, default=5)
    pd.add_argument("--out", required=True)
    pd.set_defaults(func=cmd_drive)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
