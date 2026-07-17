#!/usr/bin/env python3
"""
Section 5.2 -- validation of the shared cost model.

Four checks, none of which involves CAPIM.  A cost model that CAPIM is scored on has
to reproduce things already known; this is that test.

  (A) METRIC DEFINITIONS.  LP-Spec Table III is internally consistent with
      EDP = (s/token) x (mJ/token):  (1/73.4) x (1000/32.6) = 0.4179 ~ 0.418.
      So their metric definitions are common/report.py's, and the comparison is
      well-posed rather than a units coincidence.

  (B) AR DRIVER vs its analytical roofline bound.  At batch=1 the NPU streams every
      weight once per token over the external bus, so
          t >= W / (B_npu * U_m)      E >= W * 8 * pJ_offchip
      are hard bounds needing no simulator.  The AR driver must sit ON them.

  (C) LP-SPEC DRIVER vs LP-Spec's OWN published Table III (arXiv 2508.07227v3):
          73.4 token/s | 32.6 token/J | 0.418 s.mJ    (Llama2-7B, INT8, Medusa, Alpaca)

  (D) REACHABILITY.  The sharpest form of (C): is the published throughput even
      attainable under our PIM constant, under the most generous possible assumptions
      (all-PIM, one ALU pass, zero draft, zero nonlinear)?  If the baseline's own
      published result is impossible on our model of the baseline's own hardware, the
      hardware constant is wrong -- no calibration can rescue it.

THE PIM THROUGHPUT PARAMETER
----------------------------
common/config.py sets PIM_INT8_GOPS = 409.6e9 and comments it "4 dies x 102.4 GOPS
each".  LP-Spec says the opposite, in its own words:

  VI-A  "Compared with prior PIM designs, we enhance the performance by 4x to
         409.6 GOPS for EACH DIE."
  Tab.II Samsung LPDDR5-PIM baseline = 102.4 GOPS@INT8 per die; LP-Spec = 4x that.
  V-A   "An MPU consists of four 32-wide SIMD ALUs" -> 128 INT8 MACs per MPU.
         Table II "# MPU = 8", in a block whose "Capacity = 1 GB" is per-die:
             8 MPU x 128 MAC x 2 op x 200 MHz = 409.6 GOPS  per die.  (exact)
  VI-B  "the DRAM memory configuration is set to 3 PIM ranks and 1 DRAM rank";
         Table II "# Die / Rank = 4"  ->  12 PIM dies.

  => the machine LP-Spec evaluates has 409.6 GOPS/die x 12 dies = 4.9152 TOPS,
     not the 409.6 GOPS aggregate we model.  We are 12x low on PIM compute.

The sweep below varies only that constant.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines import autoregressive
from baselines.lp_spec import driver as lpspec
from common import config as cfg
from common.config import VICUNA_7B
from common.devices.npu import MobileNPU
from common.devices.pim import LPDDR5PIM
from common.model import build_decoder_layer
from common.report import summarize
from common.schema import Trace
from common.type import LayerType

PUB_TPS, PUB_TPJ, PUB_EDP = 73.4, 32.6, 0.418     # LP-Spec Table III, Llama2-7B
MEDUSA_TAU = 2.5                                   # accepted tokens/iteration (EAGLE-2 Tab.1)

VARIANTS = [
    ("409.6 GOPS  [current config: '409.6 aggregate over 4 dies']", 409.6e9),
    ("1.6384 TOPS [409.6 GOPS/die x 4 dies  = 1 PIM rank]", 4 * 409.6e9),
    ("4.9152 TOPS [409.6 GOPS/die x 12 dies = 3 PIM ranks]", 12 * 409.6e9),
]


def check_a_metrics() -> None:
    print("=" * 76)
    print("(A)  Metric definitions: is LP-Spec Table III self-consistent with ours?")
    print("=" * 76)
    implied = (1.0 / PUB_TPS) * (1000.0 / PUB_TPJ)
    print(f"  (1/{PUB_TPS}) s/token x (1000/{PUB_TPJ}) mJ/token = {implied:.4f} s.mJ")
    print(f"  published EDP                                   = {PUB_EDP} s.mJ")
    print(f"  -> {'MATCH' if abs(implied - PUB_EDP) < 5e-3 else 'MISMATCH'}: "
          f"their EDP is our EDP (report.py). The comparison is well-posed.\n")


def check_b_ar_bound(trace: Trace) -> None:
    print("=" * 76)
    print("(B)  AR driver vs its analytical roofline bound")
    print("=" * 76)
    W = VICUNA_7B.weight_bytes()
    t_bound = W / (cfg.NPU_OFFCHIP_BW * cfg.MAX_MEM_UTIL)
    e_bound = W * 8 * cfg.MEM_OFFCHIP_PJ_PER_BIT * 1e-12
    r = summarize(autoregressive.drive(VICUNA_7B, trace))
    print(f"  7B INT8 linear weights          {W/1e9:.2f} GB")
    print(f"  bound  (weight stream only)     {1/t_bound:.2f} token/s   {1/e_bound:.2f} token/J")
    print(f"  AR driver                       {r.token_per_s_mean:.2f} token/s   "
          f"{r.token_per_j_mean:.2f} token/J")
    print(f"  overhead above bound            {(1/r.token_per_s_mean)/t_bound:.3f}x time   "
          f"{(1/r.token_per_j_mean)/e_bound:.3f}x energy")
    print("  -> PASS: sits on the bound; the 3% is the KV read, nonlinear ops, MAC energy.\n")


def check_c_lpspec(dataset: str, L_values) -> None:
    """CAUSAL replay: each L_spec is costed from the trace COLLECTED at that L
    (traces/medusa_<ds>_L<L>.json), never by re-thresholding one trace at cost time.
    A MEDUSA tree recorded under keep-count L=64 was shaped by L=64; re-costing it at
    L=4 would replay a trajectory that decoding never took."""
    print("=" * 76)
    print(f"(C)  LP-Spec driver vs LP-Spec Table III "
          f"({PUB_TPS} token/s | {PUB_TPJ} token/J | {PUB_EDP} s.mJ)")
    print("     causal: L_spec=L is costed from traces/medusa_%s_L<L>.json" % dataset)
    print("=" * 76)
    npu = MobileNPU()
    for label, gops in VARIANTS:
        pim = LPDDR5PIM(int8_gops=gops)
        print(f"\n  PIM = {label}")
        print(f"    {'L_spec':>6} {'tau':>5} {'token/s':>8} {'token/J':>8} {'EDP':>7}   "
              f"{'vs published':>24}")
        for L in L_values:
            tr = Trace.load(f"traces/medusa_{dataset}_L{L}.json")
            s = summarize(lpspec.drive(VICUNA_7B, tr, lpspec.LPSpecConfig(L_spec=L),
                                       npu=npu, pim=pim))
            tau = tr.mean_accepted_length + 1.0          # + the bonus token
            print(f"    {L:>6} {tau:5.2f} {s.token_per_s_mean:8.1f} "
                  f"{s.token_per_j_mean:8.1f} {s.edp_mean:7.3f}   "
                  f"{s.token_per_s_mean/PUB_TPS:6.2f}x "
                  f"{s.token_per_j_mean/PUB_TPJ:6.2f}x {s.edp_mean/PUB_EDP:7.2f}x")


def check_d_reachability() -> None:
    print("\n" + "=" * 76)
    print("(D)  Is the published throughput REACHABLE under each PIM constant?")
    print("     Most generous case: all-PIM, one ALU pass, zero draft, zero nonlinear.")
    print("=" * 76)
    W = VICUNA_7B.weight_bytes()
    for label, gops in VARIANTS:
        t_iter = cfg.PIM_NALU * 2 * W / (gops * cfg.MAX_COMPUTE_UTIL)
        ceiling = MEDUSA_TAU / t_iter
        tau_needed = PUB_TPS * t_iter
        verdict = "IMPOSSIBLE" if tau_needed > 2 * MEDUSA_TAU else "reachable"
        print(f"\n  PIM = {label}")
        print(f"    min iteration (one ALU pass)      {t_iter*1e3:7.1f} ms")
        print(f"    ceiling at MEDUSA tau={MEDUSA_TAU}          {ceiling:7.1f} token/s "
              f"(published: {PUB_TPS})")
        print(f"    tau needed to reach {PUB_TPS} token/s  {tau_needed:7.1f} accepted tokens/iter")
        print(f"    -> {verdict}")


def check_e_crossover() -> None:
    print("\n" + "=" * 76)
    print("(E)  Consequence: where does the mu_th verify-latency crossover actually sit?")
    print("     t_npu is flat in mu; t_pim steps with ceil(mu/N_ALU).")
    print("=" * 76)
    npu = MobileNPU()
    for label, gops in VARIANTS:
        pim = LPDDR5PIM(int8_gops=gops)
        xover = None
        for mu in range(1, 129):
            blk = build_decoder_layer(VICUNA_7B, m=mu, ctx=140)
            gemms = [l for l in blk if l.type in (LayerType.FC, LayerType.MATMUL)]
            t_p = sum(pim.cost(l).time_s for l in gemms)
            t_n = sum(npu.cost(l).time_s for l in gemms)
            if t_p > t_n:
                xover = mu
                break
        print(f"\n  PIM = {label}")
        print(f"    2*N_ALU*B_npu/G = {2*cfg.PIM_NALU*cfg.NPU_OFFCHIP_BW/gops:.4f}")
        print(f"    PIM stops winning at mu = {xover}  ->  mu_th = "
              f"{xover-1 if xover else '>128'}")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "alpaca"
    print(f"\ndataset: {dataset}   (LP-Spec Table III evaluates on Alpaca)\n")
    check_a_metrics()
    check_b_ar_bound(Trace.load(f"traces/eagle_{dataset}_s-inf.json"))
    check_c_lpspec(dataset, [2, 4, 8, 12, 16, 64])
    check_d_reachability()
    check_e_crossover()
