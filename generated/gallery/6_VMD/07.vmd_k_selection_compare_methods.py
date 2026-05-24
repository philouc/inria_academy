"""
VMD — Example 7 · Side-by-Side Comparison of K-Selection Methods
=================================================================

This script puts all seven K-selection criteria side by side on a
single test signal with a known number of components:

    -  Classical information criteria : AIC, BIC, HQ
    -  Modified BIC with quadratic penalty (BIC*)
    -  log-MSE elbow detector
    -  Kneedle algorithm (Satopaa et al. 2011)
    -  Permutation-entropy threshold rule (Bandt & Pompe 2002)

Multi-init averaging (``n_init = 3``) is enabled so that the residual
MSE per K is robust against ADMM local minima.

The figure has four panels:

    (a)  log MSE vs K, with elbow and Kneedle markers
    (b)  classical AIC / BIC / HQ + modified BIC*
    (c)  permutation entropy of every mode at every K
    (d)  a verdict panel showing the K proposed by each method,
         the practical consensus, and the ground-truth K

The take-home message of this comparison is consistent across signals:
the four data-driven criteria (elbow, Kneedle, PE threshold, BIC*)
agree on the true K, while the classical AIC / BIC / HQ over-select
by drifting towards K_max.

Usage:  python 07.vmd_k_selection_compare_methods.py            # interactive
        python 07.vmd_k_selection_compare_methods.py out.png    # save
"""
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from inria_academy.utils.k_selection import (
    plot_criteria,
    plot_pe,
    select_k_vmd,
)

warnings.filterwarnings("ignore")

NAVY   = "#0b1f3a"
RED    = "#c9191e"
GREEN  = "#1f8a4c"
AMBER  = "#e08a1f"
PURPLE = "#7e3a93"
GREY   = "#777777"
LIGHT  = "#f3f4f7"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


def make_signal(snr_db: float = 20.0, seed: int = 0):
    """Three well-separated tones in additive Gaussian noise."""
    fs, T = 500.0, 2.0
    t = np.arange(int(fs * T)) / fs
    clean = (np.cos(2 * np.pi * 10 * t)
             + 0.7 * np.cos(2 * np.pi * 40 * t)
             + 0.5 * np.cos(2 * np.pi * 80 * t))
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(clean ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(sig_pow / (snr_lin * np.mean(noise ** 2)))
    x = clean + noise
    return t, x, fs, [10.0, 40.0, 80.0]


def _verdict_panel(ax, methods, consensus, true_K, K_max):
    """Draw the verdict panel: one row per method, dot at proposed K*."""
    K_min = 2
    n = len(methods)
    ax.set_xlim(K_min - 0.5, K_max + 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(K_min, K_max + 1))
    ax.set_xlabel("proposed K*", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Method names as y-tick labels — matplotlib gives them their own space
    short_names = [m[0].split("(")[0].strip() for m in methods]
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=9.5, color=NAVY)
    ax.tick_params(axis="y", length=0)   # hide y-tick marks
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#888")
    ax.spines["bottom"].set_linewidth(0.5)

    # Subtle horizontal guides across the data area
    for i in range(n):
        ax.plot([K_min, K_max], [i, i],
                color="#dde0e5", lw=0.5, zorder=0)

    # Truth / consensus vertical guides
    if true_K is not None:
        ax.axvline(true_K, color=GREEN, ls="--", lw=1.5, alpha=0.7, zorder=1,
                   label=f"true K = {true_K}")
    if consensus is not None:
        ax.axvline(consensus, color=PURPLE, ls=":", lw=1.8, alpha=0.85, zorder=1,
                   label=f"consensus K* = {consensus}")

    # Dots for each method's proposal
    for i, (_name, k_val, family_color) in enumerate(methods):
        if k_val is None:
            ax.text(K_max, i, "no valid K", fontsize=9, color=GREY,
                    ha="right", va="center", style="italic")
        else:
            ax.scatter([k_val], [i], s=130, color=family_color,
                       edgecolor="white", lw=1.5, zorder=3)
            # Number annotation above the dot
            ax.text(k_val, i - 0.30, f"{k_val}", ha="center", va="bottom",
                    fontsize=9.5, color=family_color, fontweight="bold")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              frameon=False, fontsize=9, ncol=2)

    ax.set_title("(d)  Verdict by method",
                 loc="left", fontsize=10, color=NAVY,
                 fontweight="bold", pad=4)


def main(savepath=None):
    # Seed numpy's global RNG so VMD init=2 (random) is reproducible
    np.random.seed(0)

    # ─── Signal ────────────────────────────────────────────────────────────
    t, x, fs, true_omegas = make_signal(snr_db=20.0)
    true_K = len(true_omegas)
    print(f"Signal: N = {len(t)}, fs = {fs} Hz, true K = {true_K}")
    print(f"True centre frequencies: {true_omegas} Hz   (SNR = 20 dB)")
    print()

    # ─── Run select_k_vmd with all the bells and whistles ──────────────────
    print("Running select_k_vmd with n_init=3 and PE  (~20 s) ...")
    res = select_k_vmd(
        x, range(2, 9),
        alpha=2000.0,
        n_init=3,            # multi-init averaging (median MSE)
        pe_m=5, pe_tau=2,    # PE embedding adapted to fs=500 Hz / fmax=80 Hz
        pe_threshold=0.6,
        verbose=True,
    )
    print()

    # ─── Tabulate every method's proposal ──────────────────────────────────
    # (name, K*, colour) — colours match the families used elsewhere
    methods = [
        ("elbow            (log-MSE marginal drop)",  res["k_elbow"],   PURPLE),
        ("Kneedle          (max curvature)",          res["k_kneedle"], AMBER),
        ("PE threshold     (PE_max < 0.6)",           res["k_pe"],      GREEN),
        ("modified BIC*    (quadratic penalty)",      res["k_bic2"],    NAVY),
        ("classical AIC",                              res["k_aic"],     RED),
        ("classical BIC",                              res["k_bic"],     RED),
        ("classical HQ",                               res["k_hq"],      RED),
    ]
    print("\nProposals from each method:")
    for name, k_val, _ in methods:
        tag = f"K* = {k_val}" if k_val is not None else "K* = (no valid K)"
        print(f"  {name:55s} {tag}")

    # Consensus: median of the four data-driven methods (elbow, Kneedle, PE, BIC*)
    trustworthy = [res["k_elbow"], res["k_kneedle"], res["k_bic2"]]
    if res["k_pe"] is not None:
        trustworthy.append(res["k_pe"])
    consensus = int(np.median(trustworthy))
    print(f"\nConsensus (median of 4 robust methods): K* = {consensus}"
          f"    (true K = {true_K})")

    # ─── Figure 2×2 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 8.4), dpi=150)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0],
        hspace=0.45, wspace=0.28,
        left=0.05, right=0.98, top=0.91, bottom=0.12,
    )

    # ─── (a) MSE + elbow + Kneedle ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(res["K"], res["mse"], "o-", color=NAVY, lw=1.5, ms=6)
    ax.axvline(res["k_elbow"], color=PURPLE, ls="--", lw=1.4,
               label=f"elbow   (K* = {res['k_elbow']})")
    ax.axvline(res["k_kneedle"], color=AMBER, ls=":", lw=1.6,
               label=f"Kneedle (K* = {res['k_kneedle']})")
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("residual MSE  (log)", fontsize=9)
    ax.set_xticks(res["K"]); ax.tick_params(labelsize=8)
    ax.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.set_title("(a)  Residual variance  +  knee detectors",
                 loc="left", fontsize=10, color=NAVY,
                 fontweight="bold", pad=4)

    # ─── (b) Classical IC + modified BIC* ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for name, key, k_key, color, marker in (
        ("AIC",       "aic",  "k_aic",  RED,   "o"),
        ("BIC",       "bic",  "k_bic",  RED,   "s"),
        ("HQ",        "hq",   "k_hq",   RED,   "^"),
        ("BIC*",      "bic2", "k_bic2", NAVY,  "D"),
    ):
        alpha = 1.0 if name == "BIC*" else 0.55
        lw    = 1.8 if name == "BIC*" else 1.1
        ax.plot(res["K"], res[key], marker=marker, color=color,
                lw=lw, ms=5, alpha=alpha,
                label=f"{name}   (K* = {res[k_key]})")
        ax.axvline(res[k_key], color=color, ls=":", lw=0.6,
                   alpha=0.4 if name != "BIC*" else 0.8)
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("information criterion", fontsize=9)
    ax.set_xticks(res["K"]); ax.tick_params(labelsize=8)
    ax.grid(True, ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=8.5, loc="best", ncol=2)
    ax.set_title("(b)  Classical AIC / BIC / HQ  +  modified BIC*",
                 loc="left", fontsize=10, color=NAVY,
                 fontweight="bold", pad=4)

    # ─── (c) PE per mode at each K ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    plot_pe(res, ax=ax)

    # ─── (d) Verdict panel ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _verdict_panel(ax, methods, consensus, true_K, K_max=int(res["K"].max()))

    # ─── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"VMD K-selection · seven methods compared on 3 tones (10, 40, 80 Hz) "
        f"+ AWGN  ·  consensus K* = {consensus}   (true K = {true_K})",
        fontsize=11, color=NAVY, fontweight="bold", y=0.97,
    )

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"\nSaved: {savepath}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
