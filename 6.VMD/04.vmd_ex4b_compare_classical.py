"""
VMD — Example 4 (continued)  Classical methods on the same signal
=================================================================

Same synthetic input as Example 4:

    f(t) = 6t²
         + cos(10π t + 10π t²)                       (chirp 5→15 Hz)
         + {  cos(60π t)         if  t ≤ 1/2          (30 Hz)
              cos(80π t − 10π)   if  t > 1/2          (40 Hz)

How do the three classical decompositions handle it?

    - SST (synchrosqueezed CWT) — reassignment-based, no mode count
    - EMD-HHT  — Empirical Mode Decomposition + Hilbert-Huang spectrum
    - EEMD-HHT — Ensemble EMD averaging over noisy realisations

Expected outcome:
    SST  →  the chirp ridge is recovered (sloped); the piecewise-constant
            jump at t = 1/2 is visible but with reassignment artefacts.
    EMD  →  mode mixing — the chirp and the piecewise content end up
            entangled in the same IMF.
    EEMD →  reduces mode mixing somewhat, but the t = 1/2 transition
            still bleeds across modes.

Usage:  python vmd_ex4_compare_classical.py            # interactive
        python vmd_ex4_compare_classical.py out.png    # save figure
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from PyEMD import EMD, EEMD

from swt_emd_helpers import NAVY, LIME, DARK_BG, compute_sst, hilbert_spectrum

# --- Palette ---------------------------------------------------------------
RED    = "#c9191e"
GREEN  = "#1f8a4c"
AMBER  = "#e08a1f"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888", "axes.linewidth": 0.4,
    "xtick.color": "#444", "ytick.color": "#444",
})


def make_signal(fs=500.0, T=1.0):
    """The Dragomiretskiy Fig. 8 synthetic test signal."""
    t = np.arange(int(fs * T)) / fs
    trend  = 6.0 * t ** 2
    chirp  = np.cos(10 * np.pi * t + 10 * np.pi * t ** 2)
    piece1 = np.cos(60 * np.pi * t)
    piece2 = np.cos(80 * np.pi * t - 10 * np.pi)
    piece  = np.where(t <= 0.5, piece1, piece2)
    return t, trend + chirp + piece


def plot_tf_panel(ax, M, t, freqs, f_min, f_max,
                  title_left, status_text, status_color,
                  show_y_label=False):
    ax.set_facecolor(DARK_BG)
    freqs = np.asarray(freqs)
    order = np.argsort(freqs)
    M_sorted = M[order]
    f_sorted = freqs[order]
    mask = (f_sorted >= f_min) & (f_sorted <= f_max + 5)
    f_use = f_sorted[mask]
    M_use = M_sorted[mask]
    Mmax = M_use.max() if M_use.size else 1.0
    T_grid, F_grid = np.meshgrid(t, f_use)
    if Mmax > 0:
        floor = max(Mmax * 5e-3, 1e-12)
        ax.pcolormesh(T_grid, F_grid, np.maximum(M_use, floor),
                      cmap="magma", shading="auto",
                      norm=LogNorm(vmin=floor, vmax=Mmax),
                      rasterized=True)
    # --- Ground-truth overlays in lime ---
    # Chirp: instantaneous frequency varies linearly from 5 to 15 Hz
    chirp_t = np.linspace(t[0], t[-1], 200)
    chirp_f = 5.0 + 10.0 * chirp_t   # from f = (d/dt)(5t + 5t²)/2π · 2π = 5+10t
    ax.plot(chirp_t, chirp_f, color=LIME, lw=0.7, ls="--", alpha=0.85)
    # Piecewise: 30 Hz for t ≤ 0.5, 40 Hz for t > 0.5
    pre = chirp_t[chirp_t <= 0.5]
    post = chirp_t[chirp_t > 0.5]
    ax.plot(pre, np.full_like(pre, 30.0),
            color=LIME, lw=0.7, ls="--", alpha=0.85)
    ax.plot(post, np.full_like(post, 40.0),
            color=LIME, lw=0.7, ls="--", alpha=0.85)
    # Labels
    ax.text(0.04, 7, "chirp", color=LIME, fontsize=6,
            fontweight="bold", alpha=0.9)
    ax.text(0.85, 32, "30 Hz", color=LIME, fontsize=6,
            fontweight="bold", alpha=0.9)
    ax.text(0.85, 42, "40 Hz", color=LIME, fontsize=6,
            fontweight="bold", alpha=0.9)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f_min, f_max)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("time [s]", fontsize=6.5, labelpad=1)
    if show_y_label:
        ax.set_ylabel("freq [Hz]", fontsize=6.5, labelpad=1)
    ax.text(0.01, 1.05, title_left, transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=7,
            fontweight="bold")
    ax.text(0.99, 1.05, status_text, transform=ax.transAxes,
            ha="right", va="bottom", color=status_color, fontsize=6.5,
            fontweight="bold")


def main(savepath=None):
    fs = 500.0
    t, x = make_signal(fs=fs)
    f_max = 50.0

    # --- SST ---
    print("Running SST...")
    t0 = time.time()
    Tx, _, ssq_freqs, _ = compute_sst(x, fs)
    print(f"  SST: {time.time() - t0:.2f} s")
    sst_mag = np.abs(Tx)

    # --- EMD-HHT ---
    print("Running EMD...")
    t0 = time.time()
    imfs_emd = EMD()(x, t)
    f_grid_emd, H_emd = hilbert_spectrum(imfs_emd, fs, f_max=f_max)
    print(f"  EMD: {imfs_emd.shape[0]} IMFs, {time.time() - t0:.2f} s")

    # --- EEMD-HHT ---
    print("Running EEMD...")
    t0 = time.time()
    imfs_eemd = EEMD(trials=80, noise_width=0.2).eemd(x, t)
    f_grid_eemd, H_eemd = hilbert_spectrum(imfs_eemd, fs, f_max=f_max)
    print(f"  EEMD: {imfs_eemd.shape[0]} IMFs, {time.time() - t0:.2f} s")

    # --- Plot ---
    fig = plt.figure(figsize=(8.0, 4.5), dpi=200)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.45, 1.6],
        hspace=0.55, wspace=0.30,
        left=0.07, right=0.97, top=0.94, bottom=0.10,
    )

    # Row 0: signal
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, x, color=NAVY, lw=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(x) * 1.1, np.max(x) * 1.1)
    ax.tick_params(labelsize=6.5)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_title(r"(a)  Dragomiretskiy Fig. 8 signal:  $6t^2 + "
                 r"\cos(10\pi t + 10\pi t^2) + \mathrm{piecewise}\,"
                 r"(30\to 40\,\mathrm{Hz})$",
                 loc="left", color=NAVY, fontsize=8,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color("#888"); sp.set_linewidth(0.4)

    # Row 1: 3 TF panels
    f_min = 0.5
    ax = fig.add_subplot(gs[1, 0])
    plot_tf_panel(ax, sst_mag, t, np.asarray(ssq_freqs),
                  f_min, f_max,
                  "(b)  SST",
                  "~ smearing at jump",
                  AMBER, show_y_label=True)
    ax = fig.add_subplot(gs[1, 1])
    plot_tf_panel(ax, H_emd, t, f_grid_emd,
                  f_min, f_max,
                  "(c)  EMD-HHT",
                  "~ clean (3 IMFs)",
                  AMBER)
    ax = fig.add_subplot(gs[1, 2])
    plot_tf_panel(ax, H_eemd, t, f_grid_eemd,
                  f_min, f_max,
                  "(d)  EEMD-HHT",
                  "✗ noise dominates",
                  RED)

    fig.text(0.5, 0.02,
             "Frequencies are well-separated and the signal is noise-free — "
             "the harder regimes (close frequencies, additive noise) are "
             "covered in Examples 2 & 3.   Compare with VMD K=4 on the previous slide.",
             ha="center", va="bottom", fontsize=6.5,
             color=NAVY, style="italic")

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
