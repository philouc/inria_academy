"""
Close frequencies — SST vs the full EMD family
================================================

Two pure tones at 25 Hz and 32 Hz (ratio 0.78), continuous, no intermittence.
This is the regime where:
  - EMD merges the two tones into a single IMF (Rilling-Flandrin)
  - EEMD doesn't help (noise injection ≠ frequency resolution)
  - CEEMDAN doesn't help either (same root cause)
  - SST cleanly resolves them as two distinct ridges

Demonstrates that the EEMD/CEEMDAN ensemble cure addresses *mode mixing*
(intermittent signals) but NOT *frequency resolution* (close-frequency
continuous signals). For the latter, you need a frequency-aware method
like SST (or VMD).

Requires:  pip install ssqueezepy EMD-signal numpy scipy matplotlib
Usage:     python scenario2b_four_methods.py            # interactive
           python scenario2b_four_methods.py out.png    # save to file
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from PyEMD import EMD, EEMD, CEEMDAN

from swt_emd_helpers import NAVY, LIME, DARK_BG, compute_sst, hilbert_spectrum
from scenario2_close_frequencies import make_signal


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888", "axes.linewidth": 0.4,
    "xtick.color": "#444", "ytick.color": "#444",
})


# ============================================================
# Compute all four time-frequency representations
# ============================================================
def compute_all(x, t, fs, f_max=50.0,
                eemd_trials=100, eemd_noise=0.2,
                ceemdan_trials=100, ceemdan_eps=0.2):
    """Return dict of {method: (M, freqs, time_seconds, info_str)}."""
    out = {}

    # ----- SST -----
    print("  SST...")
    t0 = time.time()
    Tx, _, ssq_freqs, _ = compute_sst(x, fs)
    # |Tx| as TF map; rows indexed by ssq_freqs
    out["SST"] = (np.abs(Tx), np.asarray(ssq_freqs), time.time() - t0)
    print(f"    SST: {Tx.shape}, freq range {ssq_freqs.min():.1f}-{ssq_freqs.max():.1f} Hz")

    # ----- EMD HHT -----
    print("  EMD HHT...")
    t0 = time.time()
    imfs = EMD()(x, t)
    f_grid, H = hilbert_spectrum(imfs, fs, f_max=f_max)
    out["EMD"] = (H, f_grid, time.time() - t0, imfs.shape[0])

    # ----- EEMD HHT -----
    print("  EEMD HHT...")
    t0 = time.time()
    imfs = EEMD(trials=eemd_trials, noise_width=eemd_noise).eemd(x, t)
    f_grid, H = hilbert_spectrum(imfs, fs, f_max=f_max)
    out["EEMD"] = (H, f_grid, time.time() - t0, imfs.shape[0])

    # ----- CEEMDAN HHT -----
    print("  CEEMDAN HHT...")
    t0 = time.time()
    imfs = CEEMDAN(trials=ceemdan_trials, epsilon=ceemdan_eps).ceemdan(x, t)
    f_grid, H = hilbert_spectrum(imfs, fs, f_max=f_max)
    out["CEEMDAN"] = (H, f_grid, time.time() - t0, imfs.shape[0])

    return out


# ============================================================
# Plot
# ============================================================
def plot_tf_panel(ax, M, t, freqs, f_min, f_max, true_freqs,
                  title_left, status_text, status_color,
                  log_norm=True, floor_ratio=5e-3, show_y_label=False):
    """Plot one TF panel: pcolormesh with magma + true-freq overlays."""
    ax.set_facecolor(DARK_BG)
    freqs = np.asarray(freqs)
    order = np.argsort(freqs)
    M_sorted = M[order]
    f_sorted = freqs[order]
    mask = (f_sorted >= f_min - 5) & (f_sorted <= f_max + 5)
    f_use = f_sorted[mask]
    M_use = M_sorted[mask]
    Mmax = M_use.max() if M_use.size else 1.0
    T_grid, F_grid = np.meshgrid(t, f_use)
    if log_norm and Mmax > 0:
        floor = max(Mmax * floor_ratio, 1e-12)
        ax.pcolormesh(T_grid, F_grid, np.maximum(M_use, floor),
                      cmap="magma", shading="auto",
                      norm=LogNorm(vmin=floor, vmax=Mmax),
                      rasterized=True)
    # Overlay true frequencies as horizontal lime lines
    for f in true_freqs:
        ax.axhline(f, color=LIME, lw=0.65, ls="--", alpha=0.85)
        ax.text(t[0] + 0.06, f + 0.4, f"{f:g} Hz",
                color=LIME, fontsize=5.5, va="bottom",
                fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f_min, f_max)
    ax.set_yticks([20, 25, 30, 32, 40])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("time [s]", fontsize=6.5, labelpad=1)
    if show_y_label:
        ax.set_ylabel("freq [Hz]", fontsize=6.5, labelpad=1)

    # Two-line title: method/info on top, status verdict below in color
    ax.text(0.01, 1.05, title_left, transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=7,
            fontweight="bold")
    ax.text(0.99, 1.05, status_text, transform=ax.transAxes,
            ha="right", va="bottom", color=status_color, fontsize=7,
            fontweight="bold")


def plot_comparison(t, x, fs, results, savepath=None):
    fig = plt.figure(figsize=(6.5, 4.6), dpi=200)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.45, 1.5, 1.5],
        hspace=0.65, wspace=0.20,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    # ----- Row 0: signal -----
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, x, color=NAVY, lw=0.40)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(x) * 1.05, np.max(x) * 1.05)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6.5)
    ax.set_title("(a)  Signal:  25 Hz tone  +  32 Hz tone  "
                 "(ratio 0.78  ·  continuous, no intermittence)",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)
    for s in ax.spines.values():
        s.set_color("#888"); s.set_linewidth(0.4)

    # ----- 2×2 TF panels -----
    f_min, f_max = 15.0, 45.0
    GREEN_ = "#1f8a4c"
    RED_   = "#c9191e"
    panel_specs = [
        ((1, 0), "SST",     "(b)  SST",                "✓ TWO ridges resolved",   GREEN_, True),
        ((1, 1), "EMD",     "(c)  EMD-HHT",            "✗ tones merged",          RED_,   False),
        ((2, 0), "EEMD",    "(d)  EEMD-HHT",           "✗ tones still merged",    RED_,   True),
        ((2, 1), "CEEMDAN", "(e)  CEEMDAN-HHT",        "✗ tones still merged",    RED_,   False),
    ]
    for (r, c), key, lbl, status, color, show_y in panel_specs:
        ax = fig.add_subplot(gs[r, c])
        item = results[key]
        M, freqs = item[0], item[1]
        plot_tf_panel(ax, M, t, freqs, f_min, f_max,
                      true_freqs=[25, 32],
                      title_left=lbl,
                      status_text=status,
                      status_color=color,
                      show_y_label=show_y)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    t, x, fs, _ = make_signal()
    print("Running SST + EMD + EEMD + CEEMDAN on close-frequencies signal...")
    print("  Signal: 25 Hz + 32 Hz (ratio 0.78)")
    results = compute_all(x, t, fs)
    for k, v in results.items():
        print(f"  {k:<8s}: {v[2]:.2f}s")
    plot_comparison(t, x, fs, results, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
