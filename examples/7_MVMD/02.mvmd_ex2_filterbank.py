"""
MVMD — Example 2 · Filterbank for White Gaussian Noise
=======================================================

Apply MVMD to a multivariate white-noise input — show that the K modes
naturally tile the frequency axis like a filterbank.

    -  Input: N = 50 channels of i.i.d. 𝒩(0, 1), length T = 1 s, fs = 500 Hz
    -  Output: K = 5 modes; each mode's average power spectrum is a
       narrow band  →  the set of bands tiles [0, fs/2].

This reproduces the filterbank property reported in ur Rehman & Aftab 2019.

Usage:  python mvmd_ex2_filterbank.py            # interactive
        python mvmd_ex2_filterbank.py out.png    # save
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

from inria_academy.utils.mvmd import MVMD

NAVY   = "#0b1f3a"
RED    = "#c9191e"
GREEN  = "#1f8a4c"
AMBER  = "#e08a1f"
PURPLE = "#7e3a93"
GREY   = "#777777"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


def main(savepath=None):
    fs = 500.0
    T = 1.0
    N_channels = 50
    K = 5
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * T)) / fs
    # White Gaussian noise — N channels
    x = rng.standard_normal((N_channels, len(t)))
    print(f"Input: {N_channels} channels × {len(t)} samples, "
          f"fs = {fs} Hz")

    print(f"MVMD with K = {K}, α = 2000 ...")
    u, _, omega = MVMD(x, alpha=2000.0, tau=0., K=K,
                       DC=0, init=1, tol=1e-6)
    omegas_hz = omega[-1] * fs
    print(f"  ω_k = {omegas_hz} Hz")

    # Average power spectrum per mode (across channels)
    def avg_spectrum(u_mode):
        # u_mode: (C, T) — average |FFT|² across channels
        X = np.fft.rfft(u_mode, axis=1)
        return np.mean(np.abs(X) ** 2, axis=0)

    freqs = np.fft.rfftfreq(len(t), 1.0 / fs)
    spectra = [avg_spectrum(u[k]) for k in range(K)]

    # --- Plot ---
    fig = plt.figure(figsize=(8.0, 4.6), dpi=200)
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.4, 1.0],
        hspace=0.55,
        left=0.08, right=0.97, top=0.92, bottom=0.10,
    )

    # Row 0: a few sample input channels (top)
    ax = fig.add_subplot(gs[0])
    for c in range(3):
        ax.plot(t, x[c] + c * 6, color=NAVY, lw=0.35, alpha=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xticklabels([])
    ax.set_title(f"(a)  {N_channels}-channel white Gaussian noise  "
                 "(only 3 channels shown for clarity)",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    # Row 1: stacked mode spectra
    ax = fig.add_subplot(gs[1])
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, K))
    for k in range(K):
        spec_db = 10 * np.log10(spectra[k] / spectra[k].max() + 1e-9)
        ax.plot(freqs, spec_db, color=cmap[k], lw=1.2,
                label=f"Mode {k+1}  (ω = {omegas_hz[k]:.1f} Hz)")
        ax.axvline(omegas_hz[k], color=cmap[k], lw=0.5, ls=":", alpha=0.5)
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-30, 2)
    ax.set_xlabel("frequency  [Hz]", fontsize=9, color=NAVY, labelpad=2)
    ax.set_ylabel("average mode-power spectrum  [dB]",
                  fontsize=9, color=NAVY, labelpad=2)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    ax.set_title("(b)  MVMD modes form a filterbank — adjacent bands tile  [0, fs/2]",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5,
              handlelength=1.5, ncol=K)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
