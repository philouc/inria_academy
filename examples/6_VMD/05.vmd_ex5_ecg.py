"""
VMD — Example 5 · Real-world signal: ECG decomposition
=======================================================

A clean ECG (synthesised with neurokit2's ecg_simulate, 75 bpm,
fs = 500 Hz, duration 10 s) is contaminated with realistic artefacts:

    +  baseline drift     0.3 Hz sinusoidal + slow random walk
    +  mains interference 50 Hz sinusoid at small amplitude
    +  thermal noise      𝒩(0, 0.005²)

VMD with K = 10 sorts the components by their centre frequency:

    Mode 1     →  baseline drift               (~0.3 Hz)
    Mode 2     →  heart-rate fundamental       (~1.25 Hz at 75 bpm)
    Modes 3-9  →  QRS harmonics & ECG content  (a few to ~30 Hz)
    Mode 10    →  50 Hz mains interference

The clean ECG can be reconstructed as the sum of modes 2…9
(dropping the baseline drift and the mains noise).

Usage:  python vmd_ex5_ecg.py            # interactive
        python vmd_ex5_ecg.py out.png    # save figure
"""
import sys
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from vmdpy import VMD

# --- Palette --------------------------------------------------------------
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


def make_signal(fs=500.0, duration=10.0, heart_rate=75, seed=0):
    """Synthetic ECG + baseline drift + 50 Hz mains + white noise."""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Clean ECG
    ecg_clean = nk.ecg_simulate(duration=duration, sampling_rate=int(fs),
                                 heart_rate=heart_rate,
                                 noise=0, random_state=seed)
    ecg_clean = np.asarray(ecg_clean)[:n]

    # Baseline drift: slow sinusoid + slow random walk
    drift_sin  = 0.35 * np.sin(2 * np.pi * 0.3 * t)
    walk       = np.cumsum(rng.standard_normal(n)) * 0.005
    drift      = drift_sin + walk - np.mean(walk)

    # Mains interference (50 Hz)
    mains = 0.08 * np.cos(2 * np.pi * 50.0 * t + 1.0)

    # Thermal noise
    noise = 0.005 * rng.standard_normal(n)

    raw = ecg_clean + drift + mains + noise
    return t, raw, ecg_clean, drift, mains


def plot_decomp(t, raw, ecg_clean, drift, mains, u_sorted,
                final_omegas_hz, savepath=None):
    """
    Layout:
        Row 0: raw ECG (input)
        Row 1: 10 VMD modes stacked, coloured by frequency region
        Row 2: clean reconstruction (sum of modes 2..9) overlaid w/ truth
    """
    fig = plt.figure(figsize=(8.5, 5.4), dpi=200)
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 2.6, 1.0],
        hspace=0.40,
        left=0.07, right=0.98, top=0.95, bottom=0.07,
    )

    # Colour by frequency role
    K = u_sorted.shape[0]
    colors = []
    for k in range(K):
        f = final_omegas_hz[k]
        if f < 1.0:       colors.append(PURPLE)   # baseline drift
        elif f < 30.0:    colors.append(GREEN)    # heart-beat + QRS
        else:             colors.append(RED)      # mains noise

    # ----- Row 0: raw ECG -----
    ax = fig.add_subplot(gs[0])
    ax.plot(t, raw, color=NAVY, lw=0.4)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xticklabels([])
    ax.set_title("(a)  Raw ECG  =  clean ECG  +  baseline drift  "
                 "+  50 Hz mains  +  thermal noise",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- Row 1: stacked modes -----
    ax = fig.add_subplot(gs[1])
    # normalise each mode to unit max for visibility
    for k in range(K):
        amp = np.max(np.abs(u_sorted[k])) + 1e-9
        y = u_sorted[k] / amp
        offset = (K - 1 - k) * 2.2
        ax.plot(t, y + offset, color=colors[k], lw=0.45)
        # Combined label on right
        ax.text(t[-1] + 0.04, offset,
                f"Mode {k+1:2d}    ω = {final_omegas_hz[k]:6.2f} Hz",
                color=colors[k], fontsize=6.5, va="center",
                family="monospace", fontweight="bold")
    ax.set_xlim(t[0], t[-1] * 1.25)
    ax.set_ylim(-1.5, K * 2.2)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7)
    ax.set_title("(b)  VMD with K = 10  ·  modes sorted by centre frequency  "
                 "·  colour = frequency role  "
                 "(drift / ECG content / mains noise)",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- Row 2: clean reconstruction = sum of "ECG" modes -----
    ax = fig.add_subplot(gs[2])
    # Identify which modes are in the ECG band
    ecg_band = [k for k in range(K)
                if 0.5 < final_omegas_hz[k] < 30.0]
    if not ecg_band:
        ecg_band = list(range(1, K - 1))
    clean_recon = np.sum([u_sorted[k] for k in ecg_band], axis=0)
    # Show overlay with the original clean ECG (lime/green)
    ax.plot(t, ecg_clean, color=GREEN, lw=1.0, alpha=0.6,
            label="original clean ECG")
    ax.plot(t, clean_recon, color=NAVY, lw=0.4,
            label=f"VMD reconstruction = Σ modes "
                  f"{','.join(str(k+1) for k in ecg_band)}")
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("time [s]", fontsize=8, color=NAVY, labelpad=2)
    ax.set_title("(c)  Clean ECG reconstruction  =  Σ ECG-band modes  "
                 "(drift and mains discarded)",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5,
              handlelength=1.5)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    fs = 500.0
    print("Generating synthetic ECG signal with realistic artefacts...")
    t, raw, ecg_clean, drift, mains = make_signal(fs=fs)

    print("Running VMD K=10, α=2000, τ=0...")
    u, _, omega = VMD(raw, alpha=2000.0, tau=0., K=10,
                      DC=1, init=2, tol=1e-7)
    final_omegas_hz = omega[-1] * fs
    order = np.argsort(final_omegas_hz)
    u_sorted = u[order]
    omegas_sorted = final_omegas_hz[order]
    for k in range(10):
        print(f"  Mode {k+1:2d}:  ω → {omegas_sorted[k]:7.2f} Hz")

    plot_decomp(t, raw, ecg_clean, drift, mains, u_sorted,
                omegas_sorted, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
