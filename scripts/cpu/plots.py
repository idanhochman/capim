"""CAPIM thesis figures — committed, read-only, PNG output.

Single source of figures. Reads two artifacts and never re-drives, never parses
filenames (drive records self-describe via `collection_gate`):

  * results/drive_all.json                 — the canonical CPU drive (frontier / L-sweep / table)
  * traces/eagle_{alpaca,gsm8k}_s-inf.json  — full ungated trees (premise / calibration)

Design system: the dataviz skill's validated default palette, light print surface
(thesis PDF pages are single-surface light). Datasets carried as blue/aqua; the
sequential blue ramp encodes magnitude (acceptance, sigma). LP-Spec = red baseline,
AR = ink reference.

Usage:
    .venv/bin/python scripts/cpu/plots.py [--only premise|calibration|frontier]
                                          [--outdir results] [--sigma-op -1.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# repo root on sys.path so `common` / `capim_ctrl` import when run from anywhere
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from common.schema import Trace                       # noqa: E402
from capim_ctrl.driver import _effective_accept        # noqa: E402  (exact truncation logic)

# ---------------------------------------------------------------------------
# Design tokens (dataviz reference palette, light surface)
# ---------------------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"

BLUE      = "#2a78d6"   # categorical slot 1 — Alpaca / CAPIM
AQUA      = "#1baf7a"   # categorical slot 2 — GSM8K
RED       = "#d03b3b"   # baseline — LP-Spec (status: critical)
ORANGE    = "#eb6834"   # secondary accent

# sequential blue ramp (100 -> 700), light near-zero -> dark high
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE  = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)

DS_COLOR  = {"alpaca": BLUE, "gsm8k": AQUA}
DS_LABEL  = {"alpaca": "Alpaca", "gsm8k": "GSM8K"}
DATASETS  = ["alpaca", "gsm8k"]

# Threshold symbols, matching the report's text-mode `σ\textsubscript{th}` convention.
# Unicode subscripts (U+209C, U+2095), NOT mathtext: mathtext renders italic in its own
# font and ignores the surrounding weight, so `$\mu_{th}$` would come out italic and
# non-bold inside the bold axes titles.  These glyphs are present in DejaVu Sans and
# inherit family and weight from whatever draws them.
SIG_TH = "σₜₕ"
MU_TH  = "μₜₕ"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 130,
    })


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_trace(ds: str) -> Trace:
    return Trace.load(str(REPO / "traces" / f"eagle_{ds}_s-inf.json"))


def load_drive() -> list[dict]:
    with open(REPO / "results" / "drive_all.json") as fh:
        return json.load(fh)


def _trace_arrays(trace: Trace):
    """Flatten all draft nodes -> (cumulative_log_prob, accepted, depth)."""
    cum = np.array([n.cumulative_log_prob for s in trace.steps for n in s.nodes])
    acc = np.array([n.accepted for s in trace.steps for n in s.nodes], dtype=float)
    dep = np.array([n.depth for s in trace.steps for n in s.nodes], dtype=int)
    return cum, acc, dep


# ---------------------------------------------------------------------------
# Pure compute
# ---------------------------------------------------------------------------
def premise_marginal(trace: Trace, edges: np.ndarray):
    """Acceptance rate per cumulative-log-prob bin. Returns (centers, rate, n)."""
    cum, acc, _ = _trace_arrays(trace)
    centers, rate, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (cum >= lo) & (cum < hi)
        centers.append((lo + hi) / 2)
        n.append(int(m.sum()))
        rate.append(acc[m].mean() if m.any() else np.nan)
    return np.array(centers), np.array(rate), np.array(n)


def depth_confidence_grid(trace: Trace, edges: np.ndarray, min_n: int = 30):
    """acceptance[depth, cum-bin]; cells with < min_n samples are masked (NaN)."""
    cum, acc, dep = _trace_arrays(trace)
    depths = np.arange(0, dep.max() + 1)
    grid = np.full((len(depths), len(edges) - 1), np.nan)
    for i, d in enumerate(depths):
        dm = dep == d
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = dm & (cum >= lo) & (cum < hi)
            if m.sum() >= min_n:
                grid[i, j] = acc[m].mean()
    return depths, grid


def calibration_sweep(trace: Trace, sigmas: np.ndarray):
    """For each sigma_th: pruning yield (fraction of nodes gated out) and
    accepted-path truncation (fraction of accepted tokens lost), using the
    driver's exact _effective_accept.  Returns (yield, truncation) arrays."""
    cum, _, _ = _trace_arrays(trace)
    n_nodes = len(cum)
    total_accept = sum(s.accepted_length for s in trace.steps)
    yld, trunc = [], []
    for sig in sigmas:
        yld.append(float((cum < sig).mean()))            # fraction of nodes pruned
        kept = sum(_effective_accept(s, float(sig)) for s in trace.steps)
        lost = total_accept - kept
        trunc.append(lost / total_accept)
    return np.array(yld), np.array(trunc)


# ---------------------------------------------------------------------------
# Figure 1 — Premise
# ---------------------------------------------------------------------------
def fig_premise(outdir: Path) -> Path:
    edges = np.concatenate([[-14.0], np.arange(-8.0, 0.01, 0.75)])
    fig = plt.figure(figsize=(12.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32,
                          top=0.80, bottom=0.14)

    # (a) marginal acceptance vs confidence, both datasets on one axes
    ax0 = fig.add_subplot(gs[0, 0])
    for ds in DATASETS:
        c, r, _ = premise_marginal(load_trace(ds), edges)
        ax0.plot(c, r, "-o", color=DS_COLOR[ds], ms=5, lw=2, label=DS_LABEL[ds])
    ax0.set_xlabel("Cumulative log-prob (root → node)")
    ax0.set_ylabel("Acceptance rate")
    ax0.set_title("(a) Marginal", loc="left")
    ax0.set_ylim(bottom=0)
    ax0.legend(loc="upper left")
    _despine(ax0)

    # (b,c) depth × confidence heatmaps — the depth-confound control
    hm_edges = np.arange(-8.0, 0.01, 1.0)
    axes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    vmax = 0.0
    grids = {}
    for ds in DATASETS:
        depths, grid = depth_confidence_grid(load_trace(ds), hm_edges)
        grids[ds] = (depths, grid)
        vmax = max(vmax, np.nanmax(grid))
    for ax, ds, tag in zip(axes, DATASETS, ["(b)", "(c)"]):
        depths, grid = grids[ds]
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=SEQ_BLUE,
                       vmin=0, vmax=vmax)
        ax.set_title(f"{tag} {DS_LABEL[ds]} — depth × confidence", loc="left")
        ax.set_xlabel("Cumulative log-prob")
        centers = (hm_edges[:-1] + hm_edges[1:]) / 2
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f"{c:.1f}" for c in centers], fontsize=8)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels(depths)
        if tag == "(b)":
            ax.set_ylabel("Draft-tree depth")
        ax.grid(False)
        # annotate cells
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color=INK if v < vmax * 0.55 else SURFACE)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
    cbar.set_label("Acceptance rate", color=INK_2, fontsize=9)
    cbar.outline.set_edgecolor(AXIS)

    fig.suptitle("Premise — live confidence predicts acceptance, within each depth band",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    out = outdir / "fig1_premise.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 — Calibration
# ---------------------------------------------------------------------------
def fig_calibration(outdir: Path, sigma_op: float | None) -> Path:
    sigmas = np.arange(-8.0, -0.49, 0.1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, ds, tag in zip(axes, DATASETS, ["(a)", "(b)"]):
        yld, trunc = calibration_sweep(load_trace(ds), sigmas)
        ax.plot(sigmas, yld, color=BLUE, lw=2, label="Pruning yield (nodes gated)")
        ax.plot(sigmas, trunc, color=RED, lw=2, label="Accepted-path truncation")
        if sigma_op is not None:
            ax.axvline(sigma_op, color=MUTED, ls="--", lw=1)
            ax.text(sigma_op, 1.02, f"σ={sigma_op:g}", ha="center", va="bottom",
                    fontsize=8, color=INK_2)
        ax.set_title(f"{tag} {DS_LABEL[ds]}")
        ax.set_xlabel(f"{SIG_TH} (cumulative log-prob gate)")
        ax.set_ylim(0, 1.05)
        _despine(ax)
        if tag == "(a)":
            ax.set_ylabel("Fraction")
            ax.legend(loc="center left")
    fig.suptitle(f"Calibration — pruning yield vs accepted-path truncation across {SIG_TH}",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    out = outdir / "fig2_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 (Result) — the σ × μ_th matrix as an energy–throughput frontier
# ---------------------------------------------------------------------------
# Old-repo scheme: points colored by σ (colorbar); grey lines fix μ_th and sweep
# σ, labeled μN at the σ=−inf end.  So μ_th = which line, σ = point color.
# PIM draft only (NPU draft is dominated once gated — see fig_draft_receipt).
import matplotlib.cm as cm                              # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D                     # noqa: E402

# LP-Spec's published headline (arXiv 2508.07227v3 Tab. III): 73.4 token/s, 32.6 token/J.
# Measured on THEIR setup -- drawn on both panels as the reference our baseline driver is
# validated against (throughput reproduces to 1.02x; energy to 0.75x -- see §5.2).
LP_SPEC_PUBLISHED = (73.4, 32.6)

# The two operation modes.  mu_th is NOT a per-step scheduler -- measured against the
# traces the router never fires in the gated regime (0% of steps go to the NPU at every
# EDP-optimal mu_th) -- so it is presented as a two-position power dial and the constant
# itself is kept out of the figure:
#   Standard   (mu_th=1)   87-94% of steps take the concurrent NPU||PIM split
#   Low-power  (mu_th=64)  0% do -- verification stays entirely PIM-resident
MODES      = {1: "CAPIM — Standard", 64: "CAPIM — Low-power"}
MODE_COLOR = {1: BLUE, 64: AQUA}

SIGMA_ORDER = [-2.5, -2.0, -1.5, -1.0, -0.5]
_SIG_SWATCH = [cm.plasma(x) for x in np.linspace(0.08, 0.92, len(SIGMA_ORDER))]
SIGMA_COLOR = dict(zip(SIGMA_ORDER, _SIG_SWATCH))

# LP-Spec's L is the SAME KIND of knob as CAPIM's sigma -- how much draft budget the
# method spends per step -- so it gets the same encoding (point fill = knob value) and
# the L tick labels come off the plot.  Distinct ramp (viridis) so the two knobs are
# never confused with each other.
# L reuses the SIGMA ramp, so one colour semantic covers both methods:
#   yellow = SMALL draft budget (tight sigma / few kept tokens)
#   dark   = LARGE draft budget (loose sigma / many kept tokens)
# Hence L is ordered DESCENDING (64 at the dark end, 2 at the yellow end) to line up with
# sigma, where the tight gate -0.5 is yellow.  Marker shape (circle vs X) is what tells
# the two methods apart, so sharing the ramp costs nothing and removes 5 hues.
# L=12 is dropped -- it sits between L=8 and L=16 on the same curve and adds no information.
L_ORDER   = [64, 16, 8, 4, 2]
_L_SWATCH = _SIG_SWATCH                     # same 5 plasma steps, same order (dark -> light)
L_COLOR   = dict(zip(L_ORDER, _L_SWATCH))

DRAFT_MARK  = {"npu": "o", "pim": "s"}
DRAFT_COLOR = {"npu": ORANGE, "pim": BLUE}


def _is_topm(r) -> bool:
    """The ungated EAGLE run is NOT a sigma point -- it is the fixed-budget CONTROL.

    main.py:_collection_mode tags it "topm" with collection_gate = m = 59 (EAGLE-2's own
    rerank cap).  Same drafter, same ranking, cardinality fixed instead of context-chosen,
    so it is the arm the gate is measured AGAINST -- not the loose end of the sigma axis.
    """
    return r.get("collection_mode") == "topm"


def _sig_key(r):
    """Order records loose -> tight; the top-m control sorts to the loose end."""
    return -1e9 if _is_topm(r) else r["collection_gate"]


def _knob_colorbar(fig, cax, order, swatch, label):
    """Categorical colorbar for a 'draft budget' knob (CAPIM's sigma_th, LP-Spec's L).

    Takes an EXPLICIT cax: `colorbar(ax=...)` steals space by re-laying-out the parent
    axes, which a later `subplots_adjust` then silently undoes -> bars overlap the panel.
    """
    cmap = ListedColormap(swatch)
    norm = BoundaryNorm(range(len(order) + 1), len(order))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, ticks=[i + 0.5 for i in range(len(order))])
    cb.ax.set_yticklabels([f"{v:g}" for v in order], fontsize=8)
    cb.set_label(label, color=INK_2, fontsize=9)
    cb.outline.set_edgecolor(AXIS)
    return cb


def fig_frontier(outdir: Path) -> Path:
    drive = load_drive()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, ds, tag in zip(axes, DATASETS, ["(a)", "(b)"]):
        cap = [r for r in drive
               if r["driver"] == "capim" and r["dataset"] == ds
               and r["config"]["draft_device"] == "pim"]
        gated = [r for r in cap if not _is_topm(r)]
        # Only the two OPERATION MODES are shown.  The intermediate mu_th families are
        # dropped: measured against the traces, the router never fires in the gated
        # regime (at every sigma the EDP-optimal mu_th sends 0% of steps to the NPU --
        # gated trees max out at 12-29 nodes), so mu_th is not a per-step scheduler but
        # a two-position power dial:
        #   mu_th=1   "Standard"   -> 87-94% of steps take the concurrent NPU||PIM split
        #   mu_th=64  "Low-power"  -> 0% do; verification is entirely PIM-resident
        for mu in MODES:
            fam = sorted((r for r in gated if r["config"]["mu_th"] == mu), key=_sig_key)
            ax.plot([r["token_per_s_mean"] for r in fam],
                    [r["token_per_j_mean"] for r in fam],
                    "-", color=MODE_COLOR[mu], lw=2.0, alpha=0.9, zorder=2)
            # points stay sigma-colored (colorbar): the LINE says which mode, the FILL
            # says which gate, so one plot carries both axes of the sweep.
            for r in fam:
                ax.scatter(r["token_per_s_mean"], r["token_per_j_mean"], s=52, marker="o",
                           facecolor=SIGMA_COLOR[r["collection_gate"]],
                           edgecolor=MODE_COLOR[mu], linewidth=1.2, zorder=3)
        # baseline: the WHOLE LP-Spec L sweep as its own curve, so CAPIM is compared
        # against a band, not a single hand-picked point.  L=8 is the config that
        # reproduces LP-Spec's published throughput (1.02x Tab. III); L=4 is the point
        # our own cost model says is its best (it is the largest keep-count that still
        # fits one N_ALU=4 pass -- L>=5 pays a second pass for ~4% more accepted tokens).
        lp = sorted((r for r in drive if r["driver"] == "lp_spec" and r["dataset"] == ds
                     and r["config"]["L_spec"] in L_ORDER),
                    key=lambda r: r["config"]["L_spec"])
        ax.plot([r["token_per_s_mean"] for r in lp], [r["token_per_j_mean"] for r in lp],
                "-", color=MUTED, lw=1.4, alpha=0.6, zorder=3)
        # X fill = L, read off the second colorbar.  This retires the L= text labels,
        # which could not be placed cleanly: L=2/L=4 coincide (both fit one N_ALU pass,
        # so they differ only by tau) and L=12/L=16 are ~5 token/s apart on GSM8K.
        for r in lp:
            ax.scatter(r["token_per_s_mean"], r["token_per_j_mean"], s=95, marker="X",
                       color=L_COLOR[r["config"]["L_spec"]], edgecolor=SURFACE,
                       linewidth=0.7, zorder=4)
        # LP-Spec's OWN PUBLISHED figure (Tab. III).  It does not sit on our L curve:
        # our L=8 reproduces its throughput (1.02x) but not its energy (0.75x), and no
        # L reproduces both.  Plotting it separately is the honest way to show that
        # residual -- the published pair sits ABOVE our whole baseline curve.
        ax.scatter([LP_SPEC_PUBLISHED[0]], [LP_SPEC_PUBLISHED[1]], s=120, marker="X",
                   color=INK, edgecolor=SURFACE, linewidth=0.9, zorder=5)
        ar = next(r for r in drive if r["driver"] == "ar" and r["dataset"] == ds)
        ax.scatter([ar["token_per_s_mean"]], [ar["token_per_j_mean"]], s=95, marker="^",
                   color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=4)
        ax.set_title(f"{tag} {DS_LABEL[ds]}", loc="center")
        ax.set_xlabel("Throughput (token/s)")
        if tag == "(a)":
            ax.set_ylabel("Energy efficiency (token/J)")
        ax.set_xlim(left=0)
        # headroom so the top CAPIM points (GSM8K reaches ~68 token/J) and their labels
        # are not clipped -- sharey=True, so this must cover BOTH panels, not just this one.
        ymax = max([r["token_per_j_mean"] for r in gated] +
                   [r["token_per_j_mean"] for r in lp] + [LP_SPEC_PUBLISHED[1]])
        ax.set_ylim(0, max(ax.get_ylim()[1], ymax * 1.12))
        _despine(ax)
    # legend lives OUTSIDE the axes (left gutter) so nothing overprints the data
    handles = [
        Line2D([], [], color=MODE_COLOR[1], lw=2.0, marker="o", ms=7,
               markerfacecolor=MUTED, markeredgecolor=MODE_COLOR[1], markeredgewidth=1.2,
               label=MODES[1]),
        Line2D([], [], color=MODE_COLOR[64], lw=2.0, marker="o", ms=7,
               markerfacecolor=MUTED, markeredgecolor=MODE_COLOR[64], markeredgewidth=1.2,
               label=MODES[64]),
        Line2D([], [], color=MUTED, lw=1.4, alpha=0.7, marker="X", ms=9,
               markerfacecolor=MUTED, markeredgecolor=SURFACE,
               label="LP-Spec (simulated)"),
        Line2D([], [], color="none", marker="X", ms=11, markerfacecolor=INK,
               markeredgecolor=SURFACE, label="LP-Spec (published)"),
        Line2D([], [], color="none", marker="^", ms=9, markerfacecolor=INK,
               markeredgecolor=SURFACE, label="Autoregressive"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.005, 0.5),
               frameon=False, fontsize=9, labelspacing=1.0,
               handletextpad=0.8, labelcolor=INK_2)
    # no suptitle -- the figure carries its own caption in the report
    # positions FIRST, then the colorbars into explicit gutter axes on the right
    fig.subplots_adjust(left=0.225, right=0.80, top=0.94, bottom=0.11, wspace=0.08)
    _knob_colorbar(fig, fig.add_axes([0.825, 0.11, 0.013, 0.83]),
                   SIGMA_ORDER, _SIG_SWATCH, f"CAPIM {SIG_TH} gate (cumulative log-prob)")
    _knob_colorbar(fig, fig.add_axes([0.915, 0.11, 0.013, 0.83]),
                   L_ORDER, _L_SWATCH, "LP-Spec L_spec")
    out = outdir / "fig2_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_frontier_full(outdir: Path) -> list[Path]:
    """The whole (sigma_th x mu_th) operating surface -- ONE FIGURE PER DATASET.

    Split from a single two-panel figure so each dataset can be placed next to the
    prose that discusses it; the surface is dense enough that a reader works through
    one panel at a time regardless.  The two figures are drawn by the same code on
    LOCKED, IDENTICAL axis limits: side by side the eye corrects for a shifted axis,
    but across two floats several pages apart it cannot, so the limits are the only
    thing keeping the datasets comparable.  Both carry the same annotated bead.

    Shows the gated sweep (PIM draft; the top-m control is dropped) as a controllable
    REGION rather than a ranked set of points with one winner.  What the figure has
    to carry is that BOTH thresholds are live controls over the energy-latency
    exchange, and that neither has a workload-independent optimum:

      * each sigma line traverses a real range (on Alpaca, sigma=-1.0 runs from
        152.9 token/s @ 46.1 token/J at mu_th=1 to 138.8 @ 63.8 at mu_th=64);
      * the achievable boundary of the surface is drawn from THREE different sigma
        families on Alpaca (-0.5/-1.0/-1.5) and two on GSM8K (-0.5/-1.0), so no
        single sigma dominates.  Any single "optimal" sigma is an artifact of
        whichever scalar (e.g. EDP) was asked to rank the surface, not a property
        of the architecture.

    Encoding inverts the earlier draft, which coloured by mu_th and left sigma as
    anonymous position along a line.  Here sigma is the series identity -- it selects
    WHICH exchange curve you are on -- and mu_th is travel along it, lower-right
    (mu_th=1, fast and thirsty) -> upper-left (mu_th=64, slow and lean).

    No baseline is drawn.  This section asks what the design space looks like; the
    comparison against LP-Spec and AR belongs to the evaluation, and putting the
    baseline here only forced the axes wide enough to smear the surface into an
    unreadable band.

    The annotated bead is a READING KEY, not a claim: it names one point in full --
    both knobs, both values -- so the two colorbars have somewhere to land, and from
    that one bead the rest follows off the ramps.  It sits on the sigma=-2.0 family's
    mu_th=64 end, the plateau corner, so both ramps are anchored at their high end.
    The SAME point is annotated in both figures, so the key reads identically twice.

    Palette: BOTH knobs are ordered magnitudes, so both get a sequential ramp --
    distinct/categorical hues would throw away the ordering, and the high-mu_th
    collapse is only legible if adjacent mu_th steps are adjacent in colour.  Two
    sequential contexts appear at once, so the second takes its own one-hue ramp:
    sigma = blue (line), mu_th = orange (dot fill); blue/orange are far enough apart
    that a bead is never mistaken for the thread it sits on.  Both were validated
    with the ordinal checks against this light print surface (monotone OKLCH
    dL >= 0.06 per step; lightest step clears 2:1).  Neither is the plasma ramp
    fig_frontier shares between sigma and LP-Spec's L: plasma's light end measures
    1.38:1 here, which survives as a marker fill ringed by a coloured edge but is
    invisible as a LINE colour.

    Mechanically, any mu_th above the largest tree seen at a given sigma fires for
    no step, so those families are IDENTICAL (all-PIM).  Gated mu maxes at ~12
    (Alpaca) / ~29 (GSM8K), so the top mu_th steps sit exactly on top of one another
    -- each sigma line therefore FANS OUT at low mu_th and COLLAPSES onto a plateau
    at high mu_th.  That collapse is why the shipped presets are two corners rather
    than a continuum.
    """
    drive = load_drive()
    # mu_th 2 and 16 are dropped: 2 sits between 1 and 4 on the same line and 16 is
    # inside the collapsed plateau (identical to 12/64 at every gate), so neither
    # adds a distinguishable point.  Five steps also let both ramps use the same
    # step count, so the two colorbars read at the same rate.
    mu_order = [1, 4, 8, 12, 64]
    # Both ramps are ORDINAL and validated (see docstring), and both now run the
    # same way -- low value = light, high value = dark, ascending upward on the bar.
    # NOTE this inverts fig_frontier's semantic, where dark = LOOSE gate (large
    # draft budget).  Here dark = HIGH sigma_th = the tightest gate / smallest tree.
    # The two figures therefore disagree on what dark means; they already use
    # different ramps (plasma vs blue), so nothing is silently reinterpreted, but
    # if fig_frontier is ever brought onto this ramp the direction must be settled.
    sig_swatch = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    mu_swatch = ["#fd9a4e", "#f26b15", "#d54601", "#a83703", "#7f2704"]
    sig_color = dict(zip(SIGMA_ORDER, sig_swatch))
    mu_color = dict(zip(mu_order, mu_swatch))

    out: list[Path] = []
    for ds in DATASETS:
        cap = [r for r in drive
               if r["driver"] == "capim" and r["dataset"] == ds
               and r["config"]["draft_device"] == "pim" and not _is_topm(r)
               and r["config"]["mu_th"] in mu_order]

        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        for sig in SIGMA_ORDER:
            fam = sorted((r for r in cap if r["collection_gate"] == sig),
                         key=lambda r: r["config"]["mu_th"])
            if not fam:
                continue
            xs = [r["token_per_s_mean"] for r in fam]
            ys = [r["token_per_j_mean"] for r in fam]
            mus = [r["config"]["mu_th"] for r in fam]
            # line = which gate (sigma); dots along it = which route threshold (mu_th)
            ax.plot(xs, ys, "-", color=sig_color[sig], lw=2.2, alpha=0.95, zorder=3)
            ax.scatter(xs, ys, s=78, marker="o",
                       color=[mu_color[m] for m in mus],
                       edgecolor=SURFACE, linewidth=1.0, zorder=5)

        # No title -- the LaTeX caption names the dataset; a title would say it twice.
        ax.set_xlabel("Throughput (token/s)")
        ax.set_ylabel("Energy efficiency (token/J)")

        # The reading key (see docstring) -- same bead, same wording, in both figures.
        fam1 = sorted((r for r in cap if r["collection_gate"] == -2.0),
                      key=lambda r: r["config"]["mu_th"])
        hi = (fam1[-1]["token_per_s_mean"], fam1[-1]["token_per_j_mean"])
        arrow = dict(arrowstyle="-", color=MUTED, lw=0.8,
                     shrinkA=2, shrinkB=8, connectionstyle="arc3,rad=0")
        ax.annotate(f"({MU_TH} = 64, {SIG_TH} = −2.0)", hi, textcoords="offset points",
                    xytext=(0, 34), ha="center", fontsize=8.5, color=INK_2,
                    zorder=6, arrowprops=arrow)

        # Framed to the sweep, and LOCKED ACROSS BOTH DATASETS -- these are now separate
        # floats, so shared limits are what make them comparable (see docstring).  A
        # tradeoff scatter has no zero baseline to truncate (nothing here is a length),
        # and a full-range view spends most of the canvas on empty space.
        ax.set_xlim(78, 168)
        ax.set_ylim(35, 72)
        _despine(ax)

        # No legend.  The two colorbars already name both knobs, and the annotated bead
        # ties them to a concrete point; a legend restating "line = sigma, fill = mu_th"
        # only bought back the same fact for a fifth of the canvas.  The caption carries it.
        fig.subplots_adjust(left=0.095, right=0.735, top=0.975, bottom=0.115)
        _knob_colorbar(fig, fig.add_axes([0.760, 0.115, 0.022, 0.860]),
                       SIGMA_ORDER, sig_swatch, f"{SIG_TH} (confidence-based pruning threshold)")
        _knob_colorbar(fig, fig.add_axes([0.890, 0.115, 0.022, 0.860]),
                       mu_order, mu_swatch, f"{MU_TH} (tree size-based routing threshold)")
        path = outdir / f"fig_frontier_full_{ds}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
    return out


def fig_edp_mu(outdir: Path) -> Path:
    """The decisive check for the 'EDP-optimum is at all-PIM mu_th' claim.

    EDP (s*mJ, lower is better) against mu_th, one line per gated sigma, both
    datasets.  A marker rings the per-sigma minimum.  If every ringed minimum sits
    at the high-mu_th (all-PIM) end, the claim 'the EDP-optimal mu_th sends 0% of
    steps to the NPU at every gated sigma' holds; if a minimum sits mid-axis, it
    does not.  The frontier cannot show this -- EDP ~ 1/(token_s * token_j) is a
    product the eye reads badly.
    """
    drive = load_drive()
    mu_order = [1, 2, 4, 8, 12, 16, 64]
    sig_swatch = dict(zip(SIGMA_ORDER, _SIG_SWATCH))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=False)
    for ax, ds, tag in zip(axes, DATASETS, ["(a)", "(b)"]):
        cap = [r for r in drive
               if r["driver"] == "capim" and r["dataset"] == ds
               and r["config"]["draft_device"] == "pim" and not _is_topm(r)]
        for sig in SIGMA_ORDER:
            fam = {r["config"]["mu_th"]: r for r in cap if r["collection_gate"] == sig}
            xs = [mu for mu in mu_order if mu in fam]
            ys = [fam[mu]["edp_mean"] for mu in xs]
            if not xs:
                continue
            ax.plot(xs, ys, "-o", color=sig_swatch[sig], lw=1.7, ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=3,
                    label=f"σ={sig:g}")
            jmin = int(np.argmin(ys))
            ax.scatter([xs[jmin]], [ys[jmin]], s=150, facecolor="none",
                       edgecolor=INK, linewidth=1.6, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mu_order)
        ax.set_xticklabels([str(m) for m in mu_order])
        ax.set_title(f"{tag} {DS_LABEL[ds]}", loc="center")
        ax.set_xlabel(f"{MU_TH} (verify-split threshold)")
        if tag == "(a)":
            ax.set_ylabel("EDP (s·mJ per token, lower = better)")
        _despine(ax)
    axes[0].legend(loc="best", fontsize=8, labelcolor=INK_2, title=f"gate {SIG_TH}",
                   title_fontsize=8)
    fig.text(0.5, 0.005, "ringed = per-σ EDP minimum", ha="center",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.13, wspace=0.20)
    out = outdir / "fig_edp_mu.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_validation(outdir: Path) -> Path:
    """§5.3 — validation of the reconstructed LP-Spec baseline (Alpaca only).

    The Alpaca LP-Spec L-sweep lifted straight out of fig2_frontier: the simulated
    baseline as a band over draft tree size (grey line, L-labelled X markers), with
    LP-Spec's OWN published pair (73.4 token/s, 32.6 token/J, Tab. III) overlaid as the
    black X the sweep is checked against.  At L=4 the sweep sits at (126.9, 31.9) --
    energy reproduces to 0.98x -- and at L=8 at (74.8, 24.5) -- throughput to 1.02x; no
    single L matches both, because the two straddle an N_ALU=4 ALU-pass boundary in
    LP-Spec's own latency model.  AR is the ink reference.
    """
    drive = load_drive()
    lp = sorted((r for r in drive if r["driver"] == "lp_spec" and r["dataset"] == "alpaca"
                 and r["config"]["L_spec"] in L_ORDER),
                key=lambda r: r["config"]["L_spec"])

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    # simulated baseline: the whole L sweep as one curve.  X fill = L, read off the
    # colorbar (same encoding as fig2_frontier): dark = large draft budget, light = small.
    ax.plot([r["token_per_s_mean"] for r in lp], [r["token_per_j_mean"] for r in lp],
            "-", color=MUTED, lw=1.6, alpha=0.7, zorder=2)
    for r in lp:
        ax.scatter(r["token_per_s_mean"], r["token_per_j_mean"], s=120, marker="X",
                   color=L_COLOR[r["config"]["L_spec"]], edgecolor=SURFACE,
                   linewidth=0.8, zorder=4)

    # LP-Spec's own published figure -- the point the reconstruction is validated against
    ax.scatter([LP_SPEC_PUBLISHED[0]], [LP_SPEC_PUBLISHED[1]], s=150, marker="X",
               color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=5,
               label="LP-Spec (published)")

    ar = next(r for r in drive if r["driver"] == "ar" and r["dataset"] == "alpaca")
    ax.scatter([ar["token_per_s_mean"]], [ar["token_per_j_mean"]], s=95, marker="^",
               color=INK, edgecolor=SURFACE, linewidth=0.8, zorder=4,
               label="Autoregressive")

    ax.set_xlabel("Throughput (token/s)")
    ax.set_ylabel("Energy efficiency (token/J)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, max([r["token_per_j_mean"] for r in lp] + [LP_SPEC_PUBLISHED[1]]) * 1.15)
    # explicit handles so "simulated" shows the grey line WITH its X marker
    handles = [
        Line2D([], [], color=MUTED, lw=1.6, alpha=0.7, marker="X", ms=10,
               markerfacecolor=MUTED, markeredgecolor=SURFACE, markeredgewidth=0.8,
               label="LP-Spec (simulated)"),
        Line2D([], [], color="none", marker="X", ms=11, markerfacecolor=INK,
               markeredgecolor=SURFACE, markeredgewidth=1.0, label="LP-Spec (published)"),
        Line2D([], [], color="none", marker="^", ms=9, markerfacecolor=INK,
               markeredgecolor=SURFACE, markeredgewidth=0.8, label="Autoregressive"),
    ]
    ax.legend(handles=handles, loc="lower right")
    _despine(ax)
    # position first, then the L colorbar into an explicit gutter axes on the right
    # (colorbar(ax=...) would re-lay-out the panel and fight a later subplots_adjust)
    fig.subplots_adjust(left=0.09, right=0.86, top=0.96, bottom=0.11)
    _knob_colorbar(fig, fig.add_axes([0.885, 0.11, 0.02, 0.85]),
                   L_ORDER, _L_SWATCH, "LP-Spec L_spec")
    out = outdir / "fig_validation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_headline(outdir: Path) -> Path:
    """5.5 -- the two shipped operating points against both LP-Spec references and AR.

    Five points, identified by a side legend -- no colorbar, no L-sweep curve, no
    on-plot text, unlike fig_frontier/fig_validation.  CAPIM's two operation modes
    (sigma_th=-1.0, the gate fig5_4's bound and dominance check settle on; mu_th=1
    "Standard" / mu_th=64 "Low-power" -- see fig_frontier for why the sweep collapses
    to just these two), LP-Spec's L=4 point (our own cost model's best-case
    reconstruction -- the largest keep-count that still fits one N_ALU=4 pass),
    LP-Spec's own published pair (Tab. III), and autoregressive decoding as the floor.

    Alpaca only.  LP-Spec has no published GSM8K run, so a second panel would be
    comparing CAPIM against a baseline that was never evaluated on that dataset --
    the same reason fig_validation is Alpaca-only.
    """
    drive = load_drive()
    cap = [r for r in drive if r["driver"] == "capim" and r["dataset"] == "alpaca"
           and r["config"]["draft_device"] == "pim" and not _is_topm(r)
           and r["collection_gate"] == -1.0]
    standard = next(r for r in cap if r["config"]["mu_th"] == 1)
    lowpower = next(r for r in cap if r["config"]["mu_th"] == 64)
    lp_best = next(r for r in drive if r["driver"] == "lp_spec" and r["dataset"] == "alpaca"
                   and r["config"]["L_spec"] == 4)
    ar = next(r for r in drive if r["driver"] == "ar" and r["dataset"] == "alpaca")

    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    ax.scatter(standard["token_per_s_mean"], standard["token_per_j_mean"], s=150,
               marker="o", color=MODE_COLOR[1], edgecolor=SURFACE, linewidth=1.0, zorder=4)
    ax.scatter(lowpower["token_per_s_mean"], lowpower["token_per_j_mean"], s=150,
               marker="o", color=MODE_COLOR[64], edgecolor=SURFACE, linewidth=1.0, zorder=4)
    ax.scatter(lp_best["token_per_s_mean"], lp_best["token_per_j_mean"], s=150,
               marker="X", color=RED, edgecolor=SURFACE, linewidth=1.0, zorder=4)
    ax.scatter([LP_SPEC_PUBLISHED[0]], [LP_SPEC_PUBLISHED[1]], s=150, marker="X",
               color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=4)
    ax.scatter([ar["token_per_s_mean"]], [ar["token_per_j_mean"]], s=115, marker="^",
               color=INK, edgecolor=SURFACE, linewidth=0.9, zorder=4)

    ax.set_xlabel("Throughput (token/s)")
    ax.set_ylabel("Energy efficiency (token/J)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, max(standard["token_per_j_mean"], lowpower["token_per_j_mean"],
                        lp_best["token_per_j_mean"], LP_SPEC_PUBLISHED[1]) * 1.15)
    _despine(ax)

    handles = [
        Line2D([], [], color="none", marker="o", ms=10, markerfacecolor=MODE_COLOR[1],
               markeredgecolor=SURFACE, label="CAPIM - Standard"),
        Line2D([], [], color="none", marker="o", ms=10, markerfacecolor=MODE_COLOR[64],
               markeredgecolor=SURFACE, label="CAPIM - Low-power"),
        Line2D([], [], color="none", marker="X", ms=11, markerfacecolor=RED,
               markeredgecolor=SURFACE, label="LP-Spec (simulated)"),
        Line2D([], [], color="none", marker="X", ms=11, markerfacecolor=INK,
               markeredgecolor=SURFACE, label="LP-Spec (published)"),
        Line2D([], [], color="none", marker="^", ms=9, markerfacecolor=INK,
               markeredgecolor=SURFACE, label="Autoregressive"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.775, 0.11),
               frameon=False, fontsize=9.5, labelspacing=1.3,
               handletextpad=0.8, labelcolor=INK_2)
    fig.subplots_adjust(left=0.125, right=0.78, top=0.97, bottom=0.11)
    out = outdir / "fig_headline_alpaca.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_draft_receipt(outdir: Path) -> Path:
    """Receipt: NPU-vs-PIM draft throughput collapses as σ tightens (μ_th=1)."""
    drive = load_drive()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, ds, tag in zip(axes, DATASETS, ["(a)", "(b)"]):
        cap = [r for r in drive if r["driver"] == "capim" and r["dataset"] == ds
               and r["config"]["mu_th"] == 1]
        for draft in ("npu", "pim"):
            fam = sorted((r for r in cap if r["config"]["draft_device"] == draft),
                         key=_sig_key)
            xs = [i for i, _ in enumerate(fam)]
            ax.plot(xs, [r["token_per_s_mean"] for r in fam], "-o", color=DRAFT_COLOR[draft],
                    lw=2, ms=5, label=f"draft={draft.upper()}")
        ordered = sorted({(_is_topm(r), r["collection_gate"]) for r in cap},
                         key=lambda t: -1e9 if t[0] else t[1])
        labels = [("m=59\n(ungated)" if is_topm else f"{g:g}") for is_topm, g in ordered]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"{tag} {DS_LABEL[ds]} ({MU_TH}=1)", loc="left")
        ax.set_xlabel(f"{SIG_TH} gate (loose → tight)")
        if tag == "(a)":
            ax.set_ylabel("Throughput (token/s)")
            ax.legend(loc="lower right")
        _despine(ax)
    fig.suptitle("Draft placement is a speed lever only ungated — NPU vs PIM draft collapse once gated",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.88, wspace=0.08)
    out = outdir / "fig2b_draft_receipt.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Section 5.2 — draft-tree behaviour under a confidence gate
# Data: results/pruning_analysis.json (scripts/cpu/pruning_analysis.py).
# Hardware-free by construction: work is counted in NODES, never in joules.
# ---------------------------------------------------------------------------
def load_pruning() -> dict:
    path = REPO / "results" / "pruning_analysis.json"
    if not path.exists():
        raise SystemExit("run scripts/cpu/pruning_analysis.py first (writes results/pruning_analysis.json)")
    with open(path) as fh:
        return json.load(fh)


def fig_yield(outdir: Path) -> Path:
    """5.2.1 — the draft budget saturates: the marginal node stops paying almost at once.

    Curve  = sigma swept over the ungated trace (dense, one instrument, counterfactual).
    Points = the real gated runs (ground truth), so the reader sees the curve land on them.
    AR at (0, 1.0) is exact: it drafts nothing and emits the one verification token.
    """
    pa = load_pruning()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    # ONE instrument: the ungated trace, sigma swept over it.  y = tau (accepted DRAFT
    # tokens); the iteration also emits a bonus token, so throughput is tau + 1.  The +1
    # is a constant, so it shifts the curve without changing any gradient on it.
    for ds in DATASETS:
        curve = sorted(pa[ds]["static_curve"], key=lambda r: r["mean_tree_size"])
        # anchored at the origin: no draft tree -> no accepted draft tokens, exactly
        ax.plot([0.0] + [r["mean_tree_size"] for r in curve],
                [0.0] + [r["tokens_per_iter"] - 1.0 for r in curve],
                "-", color=DS_COLOR[ds], lw=2.2, zorder=3, label=DS_LABEL[ds])

    ax.plot([0], [0.0], "o", color=INK, ms=6.5, zorder=6)
    ax.annotate("autoregressive", (0, 0), textcoords="offset points",
                xytext=(11, 1), fontsize=9, color=INK_2)

    ax.set_xlabel("Draft tree size")
    ax.set_ylabel("Mean accepted tokens per iteration")
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 4.05)
    ax.legend(loc="lower right")
    _despine(ax)
    out = outdir / "fig3_yield.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_prune_cost(outdir: Path, sigma_op: float | None) -> Path:
    """5.2.2 — work removed vs tokens kept.  Both axes are percentages: ONE y-scale."""
    pa = load_pruning()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    for ds in DATASETS:
        rows = sorted(pa[ds]["static_sweep"], key=lambda r: r["sigma"])
        sig = [r["sigma"] for r in rows]
        removed = [r["nodes_removed_frac"] * 100 for r in rows]
        kept_tok = [r["tokens_per_iter_retained_frac"] * 100 for r in rows]
        # colour = dataset (document-wide), linestyle = measure -> one y-scale, no dual axis
        ax.plot(sig, removed, "-", color=DS_COLOR[ds], lw=2.2, zorder=3, label=DS_LABEL[ds])
        ax.plot(sig, kept_tok, "--", color=DS_COLOR[ds], lw=2.2, zorder=3)

    # label each MEASURE once, away from the crowded right edge where datasets converge
    ax.annotate("nodes removed\n(solid)", (-4.35, 17), fontsize=9.5, color=INK_2,
                ha="center", va="center")
    ax.annotate("tokens/iteration kept\n(dashed)", (-4.9, 88), fontsize=9.5, color=INK_2,
                ha="center", va="center")

    if sigma_op is not None:
        ax.axvline(sigma_op, color=INK, ls=":", lw=1.3, zorder=2)
        ax.annotate(f"σ = {sigma_op:g}", (sigma_op, 6), textcoords="offset points",
                    xytext=(5, 0), fontsize=9.5, color=INK, fontweight="bold")

    ax.set_xlabel("Confidence gate σ  (loose ← → tight)")
    ax.set_ylabel("Percent of the ungated run")
    ax.set_ylim(0, 105)
    ax.set_xlim(-6.3, 0.3)
    ax.legend(loc="center left")
    ax.set_title("Most of the tree can go; most of the tokens stay", loc="left",
                 fontsize=12.5)
    _despine(ax)
    fig.text(0.02, -0.04,
             "Work is counted in nodes verified, not in joules — per-iteration energy is dominated by weight "
             "streaming, which is flat in tree size.",
             fontsize=8.5, color=MUTED, ha="left")
    out = outdir / "fig4_prune_cost.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_gate_signal(outdir: Path) -> Path:
    """5.2.3 — the score is not free (it needs a softmax, hence a bus crossing).
    Depth truncation is the free gate, so it is the bar the score must clear."""
    pa = load_pruning()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, ds in zip(axes, DATASETS):
        a = pa[ds]
        rows = sorted(a["static_sweep"], key=lambda r: r["nodes_kept_frac"])
        conf_x = [r["nodes_kept_frac"] * 100 for r in rows]
        conf_y = [r["accepted_retained_frac"] * 100 for r in rows]
        mb = a["signal_value"]["matched_budgets"]
        dep_x = [r["budget_nodes_frac"] * 100 for r in mb]
        dep_y = [r["depth_retention"] * 100 for r in mb]

        ax.plot([0, 100], [0, 100], ":", color=MUTED, lw=1.3, zorder=1,
                label="random (no signal)")
        ax.plot(dep_x, dep_y, "--o", color=ORANGE, lw=2, ms=6, mec=SURFACE, mew=1.2,
                zorder=3, label="depth truncation (free)")
        ax.plot(conf_x, conf_y, "-", color=BLUE, lw=2.4, zorder=4,
                label="cumulative confidence")

        # the operating budget, and the gap the score buys there
        op = mb[0]
        ax.annotate("", xy=(op["budget_nodes_frac"] * 100, op["confidence_retention"] * 100),
                    xytext=(op["budget_nodes_frac"] * 100, op["depth_retention"] * 100),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3), zorder=5)
        ax.annotate(f"{op['conf_vs_depth_x']:.1f}×",
                    (op["budget_nodes_frac"] * 100, (op["confidence_retention"] + op["depth_retention"]) * 50),
                    textcoords="offset points", xytext=(8, -3), fontsize=10,
                    color=INK, fontweight="bold", zorder=5)
        ax.set_title(f"{DS_LABEL[ds]}", loc="left")
        ax.set_xlabel("Verification budget (% of draft nodes kept)")
        _despine(ax)

    axes[0].set_ylabel("Accepted tokens retained (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("At the budget CAPIM operates at, the confidence score returns ~2× the tokens a free depth cut does",
                 x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.86, wspace=0.08)
    out = outdir / "fig5_gate_signal.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FIGURES = {
    "yield": lambda o, a: fig_yield(o),
    "prune_cost": lambda o, a: fig_prune_cost(o, a.sigma_op),
    "gate_signal": lambda o, a: fig_gate_signal(o),
    "premise": lambda o, a: fig_premise(o),
    "calibration": lambda o, a: fig_calibration(o, a.sigma_op),
    "frontier": lambda o, a: fig_frontier(o),
    "frontier_full": lambda o, a: fig_frontier_full(o),
    "edp_mu": lambda o, a: fig_edp_mu(o),
    "validation": lambda o, a: fig_validation(o),
    "headline": lambda o, a: fig_headline(o),
    "draft_receipt": lambda o, a: fig_draft_receipt(o),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPIM thesis figures")
    ap.add_argument("--only", choices=list(FIGURES) + ["all"], default="all")
    ap.add_argument("--outdir", default=str(REPO / "results"))
    ap.add_argument("--sigma-op", type=float, default=-1.5,
                    help="operating σ to annotate (None to omit)")
    args = ap.parse_args()

    apply_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(FIGURES) if args.only == "all" else [args.only]
    for name in names:
        made = FIGURES[name](outdir, args)
        for path in (made if isinstance(made, list) else [made]):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
