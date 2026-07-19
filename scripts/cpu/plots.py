"""CAPIM thesis figures — committed, read-only, PNG output.

Produces exactly the five figures Chapter 5 of main.tex embeds, no more:

    draft_tree_saturation_plot.png     fig:saturation   (sec:pruning)
    cost_model_validation_plot.png     fig:validation   (sec:validation)
    threshold_tradeoff_alpaca_plot.png fig:surface_alpaca   \\
    threshold_tradeoff_gsm8k_plot.png  fig:surface_gsm8k     } one function, two datasets
    headline_comparison_alpaca_plot.png fig:headline    (sec:headline)

Filenames describe content, not figure number or chapter position -- LaTeX figure
numbers renumber as the chapter is edited, so baking one into a filename just goes
stale.

Reads two artifacts and never re-drives, never parses filenames (drive records
self-describe via `collection_gate`):

  * results/drive_all.json         — the canonical CPU drive (validation / surface / headline)
  * results/pruning_analysis.json  — hardware-free draft-tree stats (saturation)

Output filenames match main.tex's \\includegraphics targets directly — no manual
rename-on-copy step. Write straight into doc/Figures/ to update the report:

    python3 scripts/cpu/plots.py --outdir ../doc/Figures
    python3 scripts/cpu/plots.py --only headline --outdir ../doc/Figures

Design system: the dataviz skill's validated default palette, light print surface
(thesis PDF pages are single-surface light). Datasets carried as blue/aqua; LP-Spec
= red baseline, AR = ink reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Design tokens (dataviz reference palette, light surface)
# ---------------------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"

BLUE      = "#2a78d6"   # categorical slot 1 — Alpaca / CAPIM Standard
AQUA      = "#1baf7a"   # categorical slot 2 — GSM8K / CAPIM Low-power
RED       = "#d03b3b"   # baseline — LP-Spec (status: critical)

DS_COLOR  = {"alpaca": BLUE, "gsm8k": AQUA}
DS_LABEL  = {"alpaca": "Alpaca", "gsm8k": "GSM8K"}
DATASETS  = ["alpaca", "gsm8k"]

# Threshold symbols, matching the report's text-mode `σ\textsubscript{th}` convention.
# Unicode subscripts (U+209C, U+2095), NOT mathtext: mathtext renders italic in its own
# font and ignores the surrounding weight, so `$\mu_{th}$` would come out italic and
# non-bold inside the bold axes titles. These glyphs are present in DejaVu Sans and
# inherit family and weight from whatever draws them.
SIG_TH = "σₜₕ"
MU_TH  = "μₜₕ"

# CAPIM's two shipped operating points. mu_th is NOT a per-step scheduler — measured
# against the traces, the router never fires in the gated regime (0% of steps go to
# the NPU at every EDP-optimal mu_th) — so it is presented as a two-position power
# dial: Standard (mu_th=1) takes the concurrent NPU||PIM split on most steps,
# Low-power (mu_th=64) stays entirely PIM-resident.
MODE_COLOR = {1: BLUE, 64: AQUA}

# CAPIM's sigma_th and LP-Spec's L are the same kind of knob (how much draft budget
# the method spends per step), so they share one ordinal ramp (plasma): dark = large
# draft budget / loose gate, light = small / tight.
SIGMA_ORDER = [-2.5, -2.0, -1.5, -1.0, -0.5]
_SIG_SWATCH = [cm.plasma(x) for x in np.linspace(0.08, 0.92, len(SIGMA_ORDER))]
L_ORDER     = [64, 16, 8, 4, 2]
_L_SWATCH   = _SIG_SWATCH                       # same 5 plasma steps, same order
L_COLOR     = dict(zip(L_ORDER, _L_SWATCH))

# LP-Spec's own published headline (arXiv 2508.07227v3 Tab. III): 73.4 token/s,
# 32.6 token/J. Measured on their setup — drawn as the point our reconstruction is
# validated against (fig:validation) and compared to directly (fig:headline).
LP_SPEC_PUBLISHED = (73.4, 32.6)


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
def load_drive() -> list[dict]:
    with open(REPO / "results" / "drive_all.json") as fh:
        return json.load(fh)


def load_pruning() -> dict:
    path = REPO / "results" / "pruning_analysis.json"
    if not path.exists():
        raise SystemExit("run scripts/cpu/pruning_analysis.py first (writes results/pruning_analysis.json)")
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _is_topm(r) -> bool:
    """The ungated EAGLE run is NOT a sigma point -- it is the fixed-budget CONTROL.

    main.py:_collection_mode tags it "topm" with collection_gate = m = 59 (EAGLE-2's own
    rerank cap). Same drafter, same ranking, cardinality fixed instead of context-chosen,
    so it is the arm the gate is measured AGAINST -- not the loose end of the sigma axis.
    """
    return r.get("collection_mode") == "topm"


def _best_lp_spec(drive: list[dict], ds: str) -> dict:
    """LP-Spec's own best simulated operating point: max throughput over its L sweep.

    Not hardcoded to a specific L -- re-derived from the data every run, so a cost-model
    change that shifts which L wins (as the 2026-07-12 PIM_INT8_GOPS fix did not, but a
    future one could) is picked up automatically. Mirrors gen_tables.py's identical
    selection, so the table and the figure can never disagree on which point is "best".
    """
    lp = [r for r in drive if r["driver"] == "lp_spec" and r["dataset"] == ds]
    return max(lp, key=lambda r: r["token_per_s_mean"])


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


# ---------------------------------------------------------------------------
# fig:saturation (sec:pruning) — the draft budget saturates: the marginal node
# stops paying almost at once. Hardware-free by construction: work is counted in
# nodes, never in joules. Data: results/pruning_analysis.json.
# ---------------------------------------------------------------------------
def fig_saturation(outdir: Path) -> Path:
    """Curve = sigma swept over the ungated trace (dense, one instrument, counterfactual).
    AR at (0, 1.0) is exact: it drafts nothing and emits the one verification token."""
    pa = load_pruning()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    # y = accepted DRAFT tokens (tau); the iteration also emits a bonus token, so
    # throughput is tau + 1 -- a constant shift that changes no gradient on the curve.
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
    out = outdir / "draft_tree_saturation_plot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig:validation (sec:validation) — the reconstructed LP-Spec baseline, swept
# across draft tree size, against LP-Spec's own published pair. Alpaca only:
# LP-Spec has no published GSM8K result to validate against.
# ---------------------------------------------------------------------------
def fig_validation(outdir: Path) -> Path:
    drive = load_drive()
    lp = sorted((r for r in drive if r["driver"] == "lp_spec" and r["dataset"] == "alpaca"
                 and r["config"]["L_spec"] in L_ORDER),
                key=lambda r: r["config"]["L_spec"])

    fig, ax = plt.subplots(figsize=(8.6, 5.4))

    # simulated baseline: the whole L sweep as one curve. X fill = L, read off the
    # colorbar: dark = large draft budget, light = small.
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
    out = outdir / "cost_model_validation_plot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# fig:surface_alpaca / fig:surface_gsm8k (sec:tradeoff) — the whole
# (sigma_th x mu_th) operating surface, ONE FIGURE PER DATASET.
# ---------------------------------------------------------------------------
def fig_surface(outdir: Path) -> list[Path]:
    """Split from a single two-panel figure so each dataset can be placed next to the
    prose that discusses it. The two figures are drawn by the same code on LOCKED,
    IDENTICAL axis limits: side by side the eye corrects for a shifted axis, but across
    two floats several pages apart it cannot, so the limits are the only thing keeping
    the datasets comparable. Both carry the same annotated bead.

    Shows the gated sweep (PIM draft; the top-m control dropped) as a controllable
    REGION rather than a ranked set of points with one winner: sigma is the series
    identity (line = which gate), mu_th is travel along it, lower-right (mu_th=1, fast
    and thirsty) -> upper-left (mu_th=64, slow and lean). No baseline is drawn -- this
    section asks what the design space looks like; the comparison against LP-Spec and
    AR belongs to fig:headline.

    Mechanically, any mu_th above the largest tree seen at a given sigma fires for no
    step, so those families are IDENTICAL (all-PIM): each sigma line FANS OUT at low
    mu_th and COLLAPSES onto a plateau at high mu_th.
    """
    drive = load_drive()
    # mu_th 2 and 16 dropped: 2 sits between 1 and 4 on the same line and 16 is inside
    # the collapsed plateau (identical to 12/64 at every gate), so neither adds a
    # distinguishable point. Five steps also let both ramps use the same step count.
    mu_order = [1, 4, 8, 12, 64]
    # Both ramps are ORDINAL: low value = light, high value = dark, ascending upward.
    sig_swatch = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    mu_swatch = ["#fd9a4e", "#f26b15", "#d54601", "#a83703", "#7f2704"]
    sig_color = dict(zip(SIGMA_ORDER, sig_swatch))
    mu_color = dict(zip(mu_order, mu_swatch))

    filenames = {"alpaca": "threshold_tradeoff_alpaca_plot.png",
                 "gsm8k": "threshold_tradeoff_gsm8k_plot.png"}
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

        # The reading key -- same bead, same wording, in both figures.
        fam1 = sorted((r for r in cap if r["collection_gate"] == -2.0),
                      key=lambda r: r["config"]["mu_th"])
        hi = (fam1[-1]["token_per_s_mean"], fam1[-1]["token_per_j_mean"])
        arrow = dict(arrowstyle="-", color=MUTED, lw=0.8,
                     shrinkA=2, shrinkB=8, connectionstyle="arc3,rad=0")
        ax.annotate(f"({MU_TH} = 64, {SIG_TH} = −2.0)", hi, textcoords="offset points",
                    xytext=(0, 34), ha="center", fontsize=8.5, color=INK_2,
                    zorder=6, arrowprops=arrow)

        # Framed to the sweep, and LOCKED ACROSS BOTH DATASETS -- these are separate
        # floats, so shared limits are what make them comparable. A tradeoff scatter
        # has no zero baseline to truncate, and a full-range view wastes the canvas.
        ax.set_xlim(78, 168)
        ax.set_ylim(35, 72)
        _despine(ax)

        # No legend. The two colorbars already name both knobs, and the annotated bead
        # ties them to a concrete point.
        fig.subplots_adjust(left=0.095, right=0.735, top=0.975, bottom=0.115)
        _knob_colorbar(fig, fig.add_axes([0.760, 0.115, 0.022, 0.860]),
                       SIGMA_ORDER, sig_swatch, f"{SIG_TH} (confidence-based pruning threshold)")
        _knob_colorbar(fig, fig.add_axes([0.890, 0.115, 0.022, 0.860]),
                       mu_order, mu_swatch, f"{MU_TH} (tree size-based routing threshold)")
        path = outdir / filenames[ds]
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# fig:headline (sec:headline) — CAPIM's two shipped operating points against
# both LP-Spec references and AR. Alpaca only: LP-Spec has no published GSM8K
# run, so a second panel would compare CAPIM against a baseline never
# evaluated on that dataset -- the same reason fig:validation is Alpaca-only.
# ---------------------------------------------------------------------------
def fig_headline(outdir: Path) -> Path:
    """Five points, identified by a side legend -- no colorbar, no L-sweep curve,
    no on-plot text, unlike fig:surface/fig:validation. CAPIM's two operation modes
    (sigma_th=-1.0, mu_th=1 "Standard" / mu_th=64 "Low-power" -- see fig:surface for
    why the sweep collapses to just these two), LP-Spec's best simulated point
    (_best_lp_spec: the L that maximises its own throughput), LP-Spec's own
    published pair (Tab. III), and autoregressive decoding as the floor.
    """
    drive = load_drive()
    cap = [r for r in drive if r["driver"] == "capim" and r["dataset"] == "alpaca"
           and r["config"]["draft_device"] == "pim" and not _is_topm(r)
           and r["collection_gate"] == -1.0]
    standard = next(r for r in cap if r["config"]["mu_th"] == 1)
    lowpower = next(r for r in cap if r["config"]["mu_th"] == 64)
    lp_best = _best_lp_spec(drive, "alpaca")
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
    out = outdir / "headline_comparison_alpaca_plot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
FIGURES = {
    "saturation": lambda o: fig_saturation(o),
    "validation": lambda o: fig_validation(o),
    "surface": lambda o: fig_surface(o),
    "headline": lambda o: fig_headline(o),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="CAPIM thesis figures (Chapter 5)")
    ap.add_argument("--only", choices=list(FIGURES) + ["all"], default="all")
    ap.add_argument("--outdir", default=str(REPO / "results"))
    args = ap.parse_args()

    apply_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(FIGURES) if args.only == "all" else [args.only]
    for name in names:
        made = FIGURES[name](outdir)
        for path in (made if isinstance(made, list) else [made]):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
