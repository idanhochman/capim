# CAPIM — Confidence-Aware PIM Architecture for LLM Inference on Mobile Devices

Simulation artefact for an MSc thesis (King's College London, 2026). CAPIM proposes a
mobile inference architecture that gates speculative decoding on **live EAGLE-2 confidence
scores**: as the draft tree grows, a hardware comparator kills any branch whose cumulative
log-probability falls below a threshold σ<sub>th</sub>, before the tokens on it are ever
verified. The result is a draft tree small enough that verification stays resident in
LPDDR5-PIM, with no off-chip weight traffic.

This repository contains everything needed to reproduce the evaluation: instrumented trace
collection from real EAGLE-2 and MEDUSA inference, an analytical PIM+NPU cost model, three
drivers (autoregressive, LP-Spec, CAPIM) scored on that one shared model, and the scripts
that turn the results into the thesis figures and tables.

The full write-up is `doc/main.tex` (build with `doc/build.sh`).

## What this is, and is not

- **Analytical simulation, not silicon.** No RTL, no fabrication, no cycle-accurate DRAM
  simulator. Latency comes from a per-layer roofline over published LPDDR5-PIM and mobile
  NPU constants; energy comes from published J/op figures. This is the same methodology as
  the LP-Spec baseline it is compared against.
- **Real traces, not synthetic ones.** The confidence scores, tree shapes and acceptance
  outcomes are recorded from actual GPU inference of EAGLE-2 and MEDUSA on Vicuna-7B-v1.3,
  with the gate firing *inside* the decode loop so a pruned node genuinely changes the
  continuation.
- **Batch size 1.** The target is a single mobile user stream, so batching is unavailable
  and speculative decoding is the only source of parallelism.
- **Energy first.** The headline metrics are LP-Spec's: throughput (token/s), energy
  efficiency (token/J) and energy-delay product (s·mJ).

## The two-phase workflow

Everything in the repository is a wrapper around one split — an expensive GPU phase that
records what the models actually did, and a cheap CPU phase that re-costs that recording
through different architectures.

```
  GPU     main.py collect  ──▶  traces/*.json     instrumented EAGLE / MEDUSA inference
  CPU     main.py drive    ──▶  results/*.json    re-cost through ar | capim | lp_spec
  CPU     scripts/cpu/*.py ──▶  figures + LaTeX tables
```

The drive phase is **pure standard library** — no GPU, no PyTorch, no dependencies at all.
Once the traces exist, the entire design-space sweep runs on a laptop in minutes, and every
figure in the thesis is regenerated from a single self-describing results artefact.

Traces and results are not committed (`traces/` and `results/` are git-ignored, and the
traces are large). **A fresh clone therefore has to run the GPU phase first** — see
[Reproducing the thesis results](#reproducing-the-thesis-results).

## Quick start

Clone **with submodules**. The collectors import from the pinned EAGLE and Medusa repos
under `sd_repos/`; a plain `git clone` leaves those directories empty and collection fails
at import.

```bash
git clone --recurse-submodules https://github.com/idanhochman/capim.git
cd capim
```

(Already cloned without them: `git submodule update --init --recursive`, which
`scripts/gpu/env.sh` also does for you.)

The shortest end-to-end path, using the five built-in sanity prompts instead of a dataset —
this needs a GPU for the first two commands only:

```bash
bash scripts/gpu/eagle/sanity.sh          # -> traces/eagle_sanity_s-1.5.json
bash scripts/gpu/medusa/sanity.sh         # -> traces/medusa_sanity_L4.json

python3 main.py drive \
    --eagle-trace  traces/eagle_sanity_s-1.5.json \
    --medusa-trace traces/medusa_sanity_L4.json \
    --driver ar capim lp_spec --out results/sanity.json

python3 scripts/cpu/print_table.py results/sanity.json
```

`main.py --help`, and `main.py {collect,drive} --help`, document every flag.

## Repository layout

```
main.py                  CLI: the `collect` (GPU) and `drive` (CPU) subcommands

common/                  the shared cost plane — all three drivers are scored on this
  config.py                LPDDR5-PIM + mobile-NPU constants, energy model, model dims
  model.py                 the typed-layer atom (FC / MATMUL / SOFTMAX / ACT / comm)
                           and the workload builders that emit layer lists
  devices/pim.py           near-bank PIM roofline: max(compute, memory) per layer
  devices/npu.py           mobile NPU roofline
  system.py                composition: device-tagged layers -> step latency + energy
  schema.py                Trace / DecodeStep / TokenNode — the GPU↔CPU contract
  gating.py                path invalidation, shared by both collectors
  report.py                per-prompt aggregation into token/s, token/J, EDP
  sd_repos.py              locates and imports the sd_repos/ submodules

capim_ctrl/              CAPIM itself
  collector.py             GPU: EAGLE-2 with the σ gate firing inside the draft loop
  eagle_topk.py            vendored EAGLE topK_genrate, gate hook added
  sequencer.py             the controller policy — σ gate and μ route, pure algorithm
  driver.py                CPU: re-costs an EAGLE trace into per-step latency/energy

baselines/
  autoregressive.py        no speculation; one target forward per token (lower bound)
  lp_spec/collector.py     GPU: MEDUSA with LP-Spec's retrospective pruner in the loop
  lp_spec/dtp.py           the Draft Token Pruner, trace-replay model
  lp_spec/driver.py        CPU: MEDUSA + DTP + concurrent NPU‖PIM verification

prompts/                 Alpaca, GSM8K and sanity prompt loaders + Vicuna formatting
scripts/gpu/             trace collection (needs a GPU and per-method deps)
scripts/cpu/             sweeps, analysis, figures, LaTeX tables (stdlib + matplotlib)
tests/                   pytest suite; no GPU, no traces required
doc/                     the thesis: main.tex, references.bib, Figures/, build.sh
sd_repos/                EAGLE and Medusa upstream repos, as pinned git submodules
```

## Reproducing the thesis results

### 1. Collect the traces (GPU)

Each script takes its swept knob as an argument and writes one trace file per point, with
the knob in the filename so a sweep never overwrites itself. Defaults are 100 prompts per
dataset at INT8.

```bash
bash scripts/gpu/eagle/collect.sh                       # both datasets, σ = -1.5
SIGMA_TH="-0.5 -1 -1.5 -2 -2.5 -inf" bash scripts/gpu/eagle/collect.sh
L="2 4 8 12 16 64" bash scripts/gpu/medusa/collect.sh
bash scripts/gpu/eagle/topm.sh "2 4 8 12"               # fixed-budget control
```

`σ = -inf` records the ungated full tree. That run has two jobs: it is the fixed-budget
control that isolates the gate from the drafter, and it is the only trace in which every
node was genuinely verified — so it is the only place a gate's discarded nodes carry a real
accept/reject label. `topm.sh` holds the drafter fixed and varies only the budget rule,
separating "EAGLE is a better drafter than MEDUSA" from "the live gate beats a retrospective
ranker."

The two stacks pin **mutually incompatible** `transformers` versions (EAGLE needs 4.53.1,
MEDUSA needs 4.36.2), so each `collect.sh` installs its own pins before running. Do not try
to share one environment between them.

### 2. Drive the sweep (CPU)

```bash
bash scripts/cpu/drive_all.sh
```

This is the canonical run and the single source of truth for every figure and table. It
re-costs every collected trace across the full design grid — σ × μ<sub>th</sub> × draft
device for CAPIM, the L sweep for LP-Spec, plus the autoregressive baseline — and merges
the parts into `results/drive_all.json` (182 records at the default grid).

Replay is causal: each EAGLE trace is driven at the σ it was *collected* at, never
re-thresholded after the fact. Each record carries its own dataset, driver, config and
collection gate, so the plotting scripts never re-drive and never parse a filename.

Subsets via environment variables, e.g.
`DATASETS=alpaca SIGMAS="-1.5 -inf" MU_THS="1 4 64" bash scripts/cpu/drive_all.sh`.

### 3. Analyse and plot (CPU)

```bash
python3 scripts/cpu/validate_cost_model.py                       # five checks, no CAPIM
python3 scripts/cpu/pruning_analysis.py                          # -> results/pruning_analysis.json
python3 scripts/cpu/plots.py --outdir doc/Figures
python3 scripts/cpu/gen_tables.py
bash    scripts/cpu/operation_modes.sh                           # headline modes table
```

`validate_cost_model.py` runs first for a reason: it checks the shared cost model against
things already known — LP-Spec's published Table III, the analytical batch-1 roofline bound,
and whether the published throughput is even reachable under our PIM constant — none of it
involving CAPIM. A cost model CAPIM is scored on has to reproduce the baseline's own results
before its verdict on CAPIM means anything.

`pruning_analysis.py` is deliberately hardware-free: it reports draft-tree behaviour in
nodes and tokens per iteration, never joules, so the gate's effect on the *algorithm* can be
read independently of the cost model.

`plots.py` writes straight into `doc/Figures/` under the filenames `main.tex` already
includes, so there is no rename-on-copy step. `gen_tables.py` prints complete LaTeX table
environments to stdout for pasting into the results chapter.

| Thesis artefact | Produced by | Reads |
|---|---|---|
| `draft_tree_saturation_plot.png` | `plots.py --only saturation` | `results/pruning_analysis.json` |
| `cost_model_validation_plot.png` | `plots.py --only validation` | `results/drive_all.json` |
| `threshold_tradeoff_{alpaca,gsm8k}_plot.png` | `plots.py --only surface` | `results/drive_all.json` |
| `headline_comparison_alpaca_plot.png` | `plots.py --only headline` | `results/drive_all.json` |
| `tab:surface`, `tab:trees`, `tab:lsweep`, `tab:headline_*` | `gen_tables.py` | both |

## Requirements

**Drive / analysis (CPU).** Python 3.12. The drive path itself needs nothing beyond the
standard library. `plots.py` additionally needs `matplotlib` and `numpy`; `tests/` needs
`pytest`.

```bash
python3 -m venv .venv && .venv/bin/pip install matplotlib numpy pytest
.venv/bin/pytest tests -q
```

**Collection (GPU).** A CUDA GPU of compute capability ≥ 7.5 (a Kaggle/Colab T4 is
sufficient) and ~16 GB of VRAM at INT8; `PRECISION=int4` lowers that, `fp16` raises it.
Python dependencies are installed by the collection scripts themselves — see the pinning
note in step 1. The base model (`lmsys/vicuna-7b-v1.3`), the EAGLE draft model
(`yuhuili/EAGLE-Vicuna-7B-v1.3`) and the MEDUSA heads
(`FasterDecoding/medusa-vicuna-7b-v1.3`) download from Hugging Face on first run.

Vicuna-7B-v1.3 is the backbone throughout because it is the only 7B target with **both** an
official EAGLE draft model and official MEDUSA heads, letting CAPIM and the LP-Spec baseline
run on an identical backbone with author-trained heads.

## Building the report

```bash
./doc/build.sh          # -> doc/main.pdf
./doc/build.sh clean
```

The report uses `fontspec`, so it must be compiled with **XeLaTeX**, not pdfLaTeX. The
script runs the full `xelatex → bibtex → makeindex → xelatex → xelatex` sequence; the
`makeindex` step builds the `nomencl` List of Symbols, and skipping it compiles that list
empty. TinyTeX is expected at `~/.TinyTeX`.

## Acknowledgements and licence

CAPIM builds directly on prior work: **EAGLE-2** (Li et al.) for confidence-scored dynamic
draft trees, **MEDUSA** (Cai et al.) for the baseline draft stack, **LP-Spec** (Peking
University, 2025) as the state-of-the-art mobile PIM baseline this work is measured against,
and **PAPI/AttAcc** for the analytical layer-decomposition cost model and its published
energy constants. The EAGLE and Medusa repositories are vendored as submodules under their
own licences.

Licensed under the Apache License 2.0 — see `LICENSE`.
