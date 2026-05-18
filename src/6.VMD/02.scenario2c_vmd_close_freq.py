"""
Close frequencies — SST + EMD family + VMD  (the resolution)
=============================================================

Same signal as scenario 2 / 2b: 25 Hz + 32 Hz tones, continuous, no
intermittence. This is the close-frequency regime (ratio 0.78), where:
  - EMD merges the two tones (Rilling-Flandrin, 2008)
  - EEMD doesn't help (ensemble averaging cures mode-mixing on
    intermittent signals, not the frequency-resolution limit of sifting)
  - CEEMDAN doesn't help either (same root cause)
  - SST resolves them cleanly (reassignment in the freq direction)
  - VMD resolves them cleanly (bandwidth-constrained mode extraction)

This is the slide where the narrative bridge from SWT to VMD closes:
two structurally different roads to the same resolution.

Output: 6-panel figure
    (a) signal (full width)
    (b) SST          — winner #1     (✓ TWO ridges resolved)
    (c) VMD          — winner #2     (✓ TWO modes recovered)
    (d) EMD-HHT      — failure       (✗ tones merged)
    (e) EEMD-HHT     — failure       (✗ tones still merged)
    (f) CEEMDAN-HHT  — failure       (✗ tones still merged)

Requires:  pip install ssqueezepy EMD-signal vmdpy numpy scipy matplotlib
Usage:     python scenario2c_vmd_close_freq.py            # interactive
           python scenario2c_vmd_close_freq.py out.png    # save to file
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from PyEMD import EMD, EEMD, CEEMDAN
from vmdpy import VMD

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '..', '5.SWT'))

from swt_emd_helpers import NAVY, LIME, DARK_BG, compute_sst, hilbert_spectrum
from scenario2_close_frequencies import make_signal


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888", "axes.linewidth": 0.4,
    "xtick.color": "#444", "ytick.color": "#444",
})


# ============================================================
# Compute all five time-frequency representations
# ============================================================
def compute_all(x, t, fs, f_max=50.0,
                eemd_trials=100, eemd_noise=0.2,
                ceemdan_trials=100, ceemdan_eps=0.2,
                vmd_alpha=2000.0, vmd_K=2):
    """Return dict of {method: (M, freqs, time_seconds, extra_info)}.

    For SST, M is |Tx| with freqs = ssq_freqs.
    For all EMD-family + VMD, M is the Hilbert-Huang spectrum on a
    uniform f-grid (so it's directly comparable across methods).
    """
    out = {}

    # ----- SST -----
    print("  SST...")
    t0 = time.time()
    Tx, _, ssq_freqs, _ = compute_sst(x, fs)
    out["SST"] = (np.abs(Tx), np.asarray(ssq_freqs), time.time() - t0, None)

    # ----- VMD (the new addition) -----
    print("  VMD (K=2)...")
    t0 = time.time()
    # alpha = bandwidth penalty (larger -> tighter bands; 2000 standard)
    # tau = 0 -> noise-free reconstruction enforced
    # K = 2 modes (we know there are two tones)
    # DC = 0, init = 1 (uniform), tol = 1e-7
    u, _, omega = VMD(x, alpha=vmd_alpha, tau=0., K=vmd_K,
                      DC=0, init=1, tol=1e-7)
    f_grid, H_vmd = hilbert_spectrum(u, fs, f_max=f_max)
    # Final center frequencies (Hz) for annotation
    final_fc = omega[-1] * fs
    out["VMD"] = (H_vmd, f_grid, time.time() - t0,
                  {"fc": final_fc, "K": vmd_K, "alpha": vmd_alpha})
    print(f"    VMD final ω_k = {final_fc} Hz")

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
# Plot one TF panel
# ============================================================
def plot_tf_panel(ax, M, t, freqs, f_min, f_max, true_freqs,
                  title_left, status_text, status_color,
                  log_norm=True, floor_ratio=5e-3, show_y_label=False,
                  show_x_label=True):
    """One TF panel with magma cmap and lime true-frequency overlays."""
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
    # True-frequency overlays (lime dashed)
    for f in true_freqs:
        ax.axhline(f, color=LIME, lw=0.65, ls="--", alpha=0.85)
        ax.text(t[0] + 0.06, f + 0.4, f"{f:g} Hz",
                color=LIME, fontsize=5.5, va="bottom",
                fontweight="bold")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f_min, f_max)
    ax.set_yticks([20, 25, 30, 32, 40])
    ax.tick_params(labelsize=6.5)
    if show_x_label:
        ax.set_xlabel("time [s]", fontsize=6.5, labelpad=1)
    else:
        ax.set_xticklabels([])
    if show_y_label:
        ax.set_ylabel("freq [Hz]", fontsize=6.5, labelpad=1)

    # Two-line title
    ax.text(0.01, 1.05, title_left, transform=ax.transAxes,
            ha="left", va="bottom", color=NAVY, fontsize=7,
            fontweight="bold")
    ax.text(0.99, 1.05, status_text, transform=ax.transAxes,
            ha="right", va="bottom", color=status_color, fontsize=7,
            fontweight="bold")


# ============================================================
# Full 6-panel figure
# ============================================================
def plot_comparison(t, x, fs, results, savepath=None):
    fig = plt.figure(figsize=(6.5, 5.2), dpi=200)
    # Layout:
    #   row 0: signal (full width)
    #   row 1: SST | VMD                              (winners)
    #   row 2: EMD | EEMD | CEEMDAN                   (failures)
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        height_ratios=[0.35, 1.5, 1.5],
        hspace=0.65, wspace=0.55,
        left=0.07, right=0.97, top=0.94, bottom=0.10,
    )

    # ----- Row 0: signal (spans all 6 cols) -----
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

    # ----- Row 1: WINNERS  (SST | VMD), each spans 3 cols -----
    f_min, f_max = 15.0, 45.0
    GREEN_ = "#1f8a4c"
    RED_   = "#c9191e"
    AMBER_ = "#e08a1f"

    # SST
    ax = fig.add_subplot(gs[1, 0:3])
    plot_tf_panel(ax, results["SST"][0], t, results["SST"][1],
                  f_min, f_max, true_freqs=[25, 32],
                  title_left="(b)  SST  (reassignment)",
                  status_text="✓ TWO ridges resolved",
                  status_color=GREEN_,
                  show_y_label=True, show_x_label=False)

    # VMD
    ax = fig.add_subplot(gs[1, 3:6])
    vmd_info = results["VMD"][3]
    fc_str = ", ".join(f"{f:.2f}" for f in vmd_info["fc"])
    plot_tf_panel(ax, results["VMD"][0], t, results["VMD"][1],
                  f_min, f_max, true_freqs=[25, 32],
                  title_left=f"(c)  VMD  (K={vmd_info['K']}, "
                             f"α={int(vmd_info['alpha'])})",
                  status_text=f"✓ ω_k → {fc_str} Hz",
                  status_color=GREEN_,
                  show_y_label=False, show_x_label=False)

    # ----- Row 2: FAILURES (EMD | EEMD | CEEMDAN), each spans 2 cols -----
    # EMD
    ax = fig.add_subplot(gs[2, 0:2])
    plot_tf_panel(ax, results["EMD"][0], t, results["EMD"][1],
                  f_min, f_max, true_freqs=[25, 32],
                  title_left="(d)  EMD-HHT",
                  status_text="✗ tones merged",
                  status_color=RED_,
                  show_y_label=True)
    # EEMD
    ax = fig.add_subplot(gs[2, 2:4])
    plot_tf_panel(ax, results["EEMD"][0], t, results["EEMD"][1],
                  f_min, f_max, true_freqs=[25, 32],
                  title_left="(e)  EEMD-HHT",
                  status_text="✗ merged",
                  status_color=RED_)
    # CEEMDAN
    ax = fig.add_subplot(gs[2, 4:6])
    plot_tf_panel(ax, results["CEEMDAN"][0], t, results["CEEMDAN"][1],
                  f_min, f_max, true_freqs=[25, 32],
                  title_left="(f)  CEEMDAN-HHT",
                  status_text="✗ merged",
                  status_color=RED_)

    # Bottom annotation: two routes to the same resolution
    fig.text(0.5, 0.015,
             "Two structurally different roads to the same resolution:   "
             "SST sharpens in frequency by reassignment   ·   "
             "VMD extracts narrow bands directly by ADMM on the variational problem.",
             ha="center", va="bottom", fontsize=5.8,
             color=NAVY, style="italic")

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    t, x, fs, _ = make_signal()
    print("Running SST + VMD + EMD + EEMD + CEEMDAN on close-frequencies signal...")
    print("  Signal: 25 Hz + 32 Hz (ratio 0.78)")
    results = compute_all(x, t, fs)
    print("\nTiming summary:")
    for k, v in results.items():
        print(f"  {k:<8s}: {v[2]:6.2f} s")
    plot_comparison(t, x, fs, results, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
