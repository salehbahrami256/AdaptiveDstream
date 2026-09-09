"""Phase 1: single-cell equidistribution criteria comparison.

This is the evaluation the paper's Section on "criterion 1" (is this cell's
data locally equidistributed?) needed and did not have: it isolates the
uniformity *test* itself from the tree/split/merge machinery entirely.
Given a batch of points known to lie in a fixed box, two criteria are
compared:

  - ``moment``: the mean/variance-vs-uniform statistic already used in
    production (``AdaptiveDStream``'s ``refinement_score``), no known null
    distribution, decision threshold is a tuned constant.
  - ``weyl``: the Weyl-sum (finite-Fourier) statistic from
    ``adaptive_dstream.criteria``, asymptotically chi-square under H0,
    decision threshold is a chi-square quantile at a chosen significance
    level (see that module's docstring for the derivation).

  - ``randproj``: a random-projection test (``adaptive_dstream.criteria.
    random_projection_score``) with a finite-sample, distribution-free
    false-positive bound derived from Hoeffding's inequality plus a union
    bound over directions -- unlike Weyl's chi-square calibration, this
    bound is exact at *every* n, not just asymptotically. The trade-off,
    stated as a hypothesis to check empirically below rather than asserted:
    Hoeffding's inequality is loose (it uses only boundedness, not the true
    variance), so at matched alpha this test is expected to have less power
    than Weyl at any given finite n.

Three things are measured, all across dimension up to 20 (cheap here since
none of the three criteria enumerate 2^dim orthants -- see the criteria
module):

  1. Discriminative power on five synthetic single-cell shapes (the H0
     uniform baseline plus four adversarial alternatives, one of which is
     exactly the "uniform core, imbalanced opposite borders" shape flagged
     as the paper's headline stress case), summarized as ROC-AUC so the two
     criteria are compared without needing to match their very differently
     scaled thresholds.
  2. Threshold calibration: at a *fixed* sample size, does each criterion's
     empirical false-positive rate under true H0 match what its threshold
     nominally promises? (The Weyl test's threshold is chosen from its own
     chi-square null, so this doubles as a check that the derivation in
     ``criteria.py`` is not just correct asymptotically but well-calibrated
     at the sample sizes actually used elsewhere in this repo.)

Run: python examples/run_cell_equidistribution_sweep.py
Outputs: outputs/cell_equidistribution_auc.json,
outputs/cell_equidistribution_auc.png,
outputs/cell_equidistribution_calibration.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_dstream.cell_shapes import sample_shape
from adaptive_dstream.criteria import (
    moment_refinement_score,
    weyl_equidistribution_score,
    weyl_threshold,
    random_projection_score,
)

OUTPUT_DIR = Path("outputs")
SEED = 7

ALT_SHAPES = [
    "core_border_imbalance",
    "core_border_imbalance_diag",
    "diagonal_correlated",
    "symmetric_bimodal",
    "corner_spike",
]
DIMS = [2, 5, 10, 15, 20]
N_AUC = 400          # sample size used for the discriminative-power sweep
R_AUC = 60           # repeats per (shape, dim) for the ROC-AUC estimate
N_CALIBRATION = [50, 100, 200, 400, 800, 1600, 3200, 6400]
DIM_CALIBRATION = 10
R_CALIBRATION = 300
ALPHA = 0.05


def score_batch(kind: str, n: int, dim: int, seed: int) -> tuple[float, float, float]:
    """Return (moment_score, weyl_statistic, randproj_statistic) for one drawn sample."""
    X = sample_shape(kind, n, dim, seed)
    lower, upper = np.zeros(dim), np.ones(dim)
    moment = moment_refinement_score(X, lower, upper)
    weyl = weyl_equidistribution_score(X, lower, upper).statistic
    randproj = random_projection_score(X, lower, upper).statistic
    return moment, weyl, randproj


CRITERIA = ["moment", "weyl", "randproj"]
STYLE = {
    "moment": dict(color="tab:blue", marker="o", markersize=7, label="moment (mean/var)"),
    "weyl": dict(color="tab:red", marker="*", markersize=10, label="Weyl"),
    "randproj": dict(color="tab:green", marker="s", markersize=6, label="random projection (Hoeffding)"),
}


def run_auc_sweep() -> dict:
    print("Phase 1a: discriminative power (ROC-AUC) by shape and dimension")
    results = {shape: {"dims": DIMS, **{f"{c}_auc": [] for c in CRITERIA}} for shape in ALT_SHAPES}
    for dim in DIMS:
        null_scores = np.empty((R_AUC, 3))
        for r in range(R_AUC):
            null_scores[r] = score_batch("uniform", N_AUC, dim, seed=SEED * 100000 + dim * 1000 + r)
        for shape in ALT_SHAPES:
            alt_scores = np.empty((R_AUC, 3))
            for r in range(R_AUC):
                alt_scores[r] = score_batch(
                    shape, N_AUC, dim, seed=SEED * 200000 + dim * 1000 + r + hash(shape) % 1000)
            labels = np.concatenate([np.zeros(R_AUC), np.ones(R_AUC)])
            line = f"  dim={dim:>2}  {shape:<28}"
            for j, crit in enumerate(CRITERIA):
                scores = np.concatenate([null_scores[:, j], alt_scores[:, j]])
                auc = roc_auc_score(labels, scores)
                results[shape][f"{crit}_auc"].append(auc)
                line += f"  {crit} AUC={auc:.3f}"
            print(line)
    return results


def run_calibration_sweep() -> dict:
    print("\nPhase 1b: false-positive-rate calibration under true H0 (dim="
          f"{DIM_CALIBRATION}, batch/no-decay)")
    moment_thresh = 0.04  # AdaptiveDStream's production split_threshold default
    w_thresh = weyl_threshold(DIM_CALIBRATION, alpha=ALPHA)
    rp_thresh = -np.log(ALPHA)  # random_projection_score's statistic is -log(p_bound)
    out = {"n": N_CALIBRATION, "nominal_alpha": ALPHA,
           **{f"{c}_fpr": [] for c in CRITERIA}}
    for n in N_CALIBRATION:
        hits = {c: 0 for c in CRITERIA}
        for r in range(R_CALIBRATION):
            m, w, rp = score_batch("uniform", n, DIM_CALIBRATION, seed=90000 + n * 13 + r)
            hits["moment"] += m > moment_thresh
            hits["weyl"] += w > w_thresh
            hits["randproj"] += rp > rp_thresh
        line = f"  n={n:>6}"
        for c in CRITERIA:
            fpr = hits[c] / R_CALIBRATION
            out[f"{c}_fpr"].append(fpr)
            line += f"  {c} FPR={fpr:.3f}"
        print(line)
    return out


def plot_auc(results: dict) -> None:
    fig, axes = plt.subplots(1, len(ALT_SHAPES), figsize=(4 * len(ALT_SHAPES), 4), sharey=True)
    for ax, shape in zip(axes, ALT_SHAPES):
        r = results[shape]
        for crit in CRITERIA:
            ax.plot(r["dims"], r[f"{crit}_auc"], **STYLE[crit])
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(shape, fontsize=9)
        ax.set_xlabel("Dimension")
        ax.set_ylim(0.35, 1.02)
    axes[0].set_ylabel("ROC-AUC (uniform vs. shape)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Single-cell equidistribution test: discriminative power, n={N_AUC}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cell_equidistribution_auc.png", dpi=150)
    plt.close(fig)


def plot_calibration(cal: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = {"moment": "moment @ fixed threshold 0.04",
              "weyl": f"Weyl @ chi-square quantile (alpha={ALPHA})",
              "randproj": f"random projection @ Hoeffding bound (alpha={ALPHA})"}
    for crit in CRITERIA:
        style = dict(STYLE[crit])
        style["label"] = labels[crit]
        ax.plot(cal["n"], cal[f"{crit}_fpr"], **style)
    ax.axhline(cal["nominal_alpha"], color="gray", linestyle=":", linewidth=1, label="nominal alpha")
    ax.set_xscale("log")
    ax.set_xlabel("Sample size n (batch, no decay)")
    ax.set_ylabel("Empirical false-positive rate under true H0")
    ax.set_title(f"Threshold calibration under H0, dim={DIM_CALIBRATION}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cell_equidistribution_calibration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    auc_results = run_auc_sweep()
    calibration = run_calibration_sweep()

    with open(OUTPUT_DIR / "cell_equidistribution_auc.json", "w") as f:
        json.dump({"auc": auc_results, "calibration": calibration}, f, indent=2)

    plot_auc(auc_results)
    plot_calibration(calibration)
    print(f"\nWrote {OUTPUT_DIR}/cell_equidistribution_auc.{{json,png}} and "
          f"{OUTPUT_DIR}/cell_equidistribution_calibration.png")


if __name__ == "__main__":
    main()
