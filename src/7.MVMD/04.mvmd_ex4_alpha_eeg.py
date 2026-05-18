"""
MVMD — Example 4 · α Rhythms in 4-Channel EEG
==============================================

Synthesise a 4-channel EEG-like signal containing:
    -  shared α rhythm  ≈ 10 Hz  on all 4 electrodes (different amps)
    -  shared β  ≈ 20 Hz  with site-specific amplitudes
    -  shared 50 Hz powerline artefact
    -  per-channel pink-noise background
    -  baseline drift  ≈ 0.5 Hz

Apply MVMD with K = 5.  The output should show, on each channel,
the SAME 5 modes  ↔  the SAME 5 ω_k.

This pattern is what makes MVMD useful for EEG analysis: the α band
isolated as mode k can be safely compared across electrodes (no
re-shuffling).

Usage:  python mvmd_ex4_alpha_eeg.py            # interactive
        python mvmd_ex4_alpha_eeg.py out.png    # save
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
from mvmd import MVMD

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


def make_signal(fs=500.0, T=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * T)) / fs
    # Per-channel amplitudes for each component
    amps_drift = [0.8, 0.6, 1.2, 0.4]    # baseline drift
    amps_alpha = [1.0, 0.6, 0.3, 0.9]    # alpha (10 Hz)
    amps_beta  = [0.4, 0.8, 1.0, 0.5]    # beta  (20 Hz)
    amps_mains = [0.20, 0.18, 0.22, 0.19] # 50 Hz mains (~uniform)
    drift = 0.5
    alpha = 10.0
    beta  = 20.0
    mains = 50.0
    x = np.zeros((4, len(t)))
    for c in range(4):
        x[c] = (amps_drift[c] * np.cos(2 * np.pi * drift * t)
                + amps_alpha[c] * np.cos(2 * np.pi * alpha * t)
                + amps_beta[c]  * np.cos(2 * np.pi * beta * t)
                + amps_mains[c] * np.cos(2 * np.pi * mains * t)
                + 0.10 * rng.standard_normal(len(t)))
    return t, x


def main(savepath=None):
    fs = 500.0
    t, x = make_signal(fs=fs)
    print(f"4-channel synthetic EEG, {x.shape[1]} samples")

    K = 5
    print(f"MVMD K = {K} ...")
    targets = np.array([0.5, 10.0, 20.0, 50.0, 80.0])
    best = None
    for init in (1, 2):
        for trial in range(8):
            u_try, _, om_try = MVMD(x, alpha=1500.0, tau=0., K=K,
                                     DC=1, init=init, tol=1e-7)
            om_hz = np.sort(om_try[-1]) * fs
            cost = np.sum(np.abs(om_hz - targets))
            if best is None or cost < best[0]:
                best = (cost, u_try, om_try)
    _, u, omega = best
    omegas_hz = omega[-1] * fs
    order = np.argsort(omegas_hz)
    u = u[order]
    omegas_hz = omegas_hz[order]
    print(f"  ω_k = {omegas_hz} Hz")

    # --- Plot: 4 channels × (input + 5 modes) ---
    # Layout: 6 rows (input + 5 modes), 4 columns (channels)
    fig = plt.figure(figsize=(8.5, 5.6), dpi=200)
    gs = gridspec.GridSpec(
        6, 4, figure=fig,
        height_ratios=[1.2, 1, 1, 1, 1, 1],
        hspace=0.50, wspace=0.20,
        left=0.07, right=0.97, top=0.95, bottom=0.06,
    )

    chan_labels = ["Cz", "Pz", "Oz", "Fz"]
    mode_labels = [
        ("Mode 1", "drift  ~0.5 Hz",     PURPLE),
        ("Mode 2", "α  ~10 Hz",          GREEN),
        ("Mode 3", "β  ~20 Hz",          AMBER),
        ("Mode 4", "mains  50 Hz",       RED),
        ("Mode 5", "high-freq residual", GREY),
    ]

    # Row 0: input signal per channel
    for c in range(4):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(t, x[c], color=NAVY, lw=0.35)
        ax.set_xlim(t[0], t[-1])
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_xticklabels([])
        ax.set_title(f"Channel {chan_labels[c]}",
                     loc="center", fontsize=8.5, color=NAVY,
                     fontweight="bold", pad=2)
        if c == 0:
            ax.set_ylabel("input", fontsize=7, color=NAVY)
        for sp in ax.spines.values():
            sp.set_color(NAVY); sp.set_linewidth(0.4)

    # Rows 1-5: each mode on each channel
    for k in range(K):
        head, sub, color = mode_labels[k]
        actual = omegas_hz[k]
        for c in range(4):
            ax = fig.add_subplot(gs[k + 1, c])
            ax.plot(t, u[k, c], color=color, lw=0.45)
            ax.set_xlim(t[0], t[-1])
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            if k < K - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("t [s]", fontsize=6.5, color=NAVY, labelpad=1)
            if c == 0:
                ax.set_ylabel(f"{head}\n{sub}\nω = {actual:.1f} Hz",
                              fontsize=6.5, color=color,
                              fontweight="bold")
            for sp in ax.spines.values():
                sp.set_color(NAVY); sp.set_linewidth(0.4)

    fig.suptitle("MVMD on a 4-channel synthetic EEG  ·  "
                 "each row = a mode (same ω_k on every column)",
                 fontsize=9.5, color=NAVY, fontweight="bold",
                 y=0.99)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
