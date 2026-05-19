"""
VMD — Example 3 · Noise robustness on a tri-harmonic
=====================================================

Signal:
    f_n(t) = cos(4π t) + (1/4) cos(48π t) + (1/16) cos(576π t) + η
    with η ∼ 𝒩(0, 0.1) and three known carriers at  ω ∈ {4π, 48π, 576π}
    (i.e.  2 Hz, 24 Hz, 288 Hz).

VMD with K = 3, λ disabled (τ = 0, denoising mode) should:
    - recover Mode 1 (ω ≈ 4π)   almost perfectly
    - recover Mode 2 (ω ≈ 48π)  with mild noise leakage
    - track Mode 3 (ω ≈ 576π)  with the right frequency but degraded
      amplitude — the amplitude is only 1/16 so noise dominates.

This reproduces Dragomiretskiy & Zosso 2014, Fig. 6.

Usage:  python vmd_ex3_noisy_triharmonic.py            # interactive
        python vmd_ex3_noisy_triharmonic.py out.png    # save figure
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from vmdpy import VMD

# --- Palette ---------------------------------------------------------------
NAVY  = "#0b1f3a"
RED   = "#c9191e"
GREEN = "#1f8a4c"
AMBER = "#e08a1f"
GREY  = "#777777"
LIME  = "#7CFC00"   # ground-truth overlay
DARK_BG = "#0a0a18"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


# --- Signal ---------------------------------------------------------------
def make_signal(fs=2000.0, T=1.0, noise_std=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * T)) / fs
    m1 = np.cos(4 * np.pi * t)
    m2 = (1.0 / 4.0) * np.cos(48 * np.pi * t)
    m3 = (1.0 / 16.0) * np.cos(576 * np.pi * t)
    noise = noise_std * rng.standard_normal(len(t))
    f_n = m1 + m2 + m3 + noise
    return t, f_n, (m1, m2, m3), noise


# --- Plot ------------------------------------------------------------------
def plot_decomp(t, f_n, modes_true, modes_vmd, omegas_final_hz, savepath=None):
    """
    Layout:
        Row 0: input signal (full width)
        Row 1: 3 columns - mode k recovered (red) overlaid with ground truth (lime)
        Row 2: 3 columns - power spectrum of mode k with ground-truth freq marker
    """
    fig = plt.figure(figsize=(8.0, 5.0), dpi=200)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        height_ratios=[0.7, 1.0, 0.7],
        hspace=0.65, wspace=0.30,
        left=0.07, right=0.97, top=0.95, bottom=0.07,
    )
    colors = [GREEN, AMBER, RED]
    omegas_true_hz = [2.0, 24.0, 288.0]
    sym = [r"$\omega \approx 4\pi$", r"$\omega \approx 48\pi$",
           r"$\omega \approx 576\pi$"]
    amps_true = [1.0, 0.25, 0.0625]

    # ----- Row 0: input -----
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, f_n, color=NAVY, lw=0.4)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.8, 1.8)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7)
    ax.set_title("(a)  Input signal:  "
                 r"$f_n(t) = \cos(4\pi t) + \frac{1}{4}\cos(48\pi t)"
                 r" + \frac{1}{16}\cos(576\pi t) + \mathcal{N}(0,0.1)$",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- Row 1: each mode (recovered + truth overlay) -----
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        mtrue = modes_true[k]
        mvmd  = modes_vmd[k]
        # Truth (lime, behind)
        ax.plot(t, mtrue, color=LIME, lw=0.95, alpha=0.85,
                label="ground truth")
        # VMD recovery (colored, in front)
        ax.plot(t, mvmd, color=colors[k], lw=0.55,
                label="VMD")
        ax.set_xlim(t[0], t[-1])
        # Y-limits keyed to the true amplitude
        ax.set_ylim(-amps_true[k] * 1.6, amps_true[k] * 1.6)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Mode {k+1}   {sym[k]}   "
                     f"(amp = {amps_true[k]:.4g})",
                     loc="left", fontsize=8, color=colors[k],
                     fontweight="bold", pad=2)
        if k == 0:
            ax.set_ylabel("amplitude", fontsize=7.5, color=NAVY)
            ax.legend(loc="upper right", frameon=False, fontsize=6.5,
                      handlelength=1.4)
        # ω_k convergence annotation
        ax.text(0.98, 0.05,
                f"VMD ω = {omegas_final_hz[k]:.2f} Hz\n"
                f"true    = {omegas_true_hz[k]:.2f} Hz",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=NAVY,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#ffffff", edgecolor="none",
                          alpha=0.75))
        ax.set_xlabel("time [s]", fontsize=7.5, color=NAVY, labelpad=1)
        for sp in ax.spines.values():
            sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- Row 2: spectra -----
    fs = 1.0 / (t[1] - t[0])
    win = np.hanning(len(t))
    for k in range(3):
        ax = fig.add_subplot(gs[2, k])
        mvmd = modes_vmd[k]
        X = np.fft.rfft(mvmd * win)
        freqs = np.fft.rfftfreq(len(t), 1.0 / fs)
        mag = np.abs(X) / np.max(np.abs(X) + 1e-12)
        ax.semilogy(freqs, np.maximum(mag, 1e-3),
                    color=colors[k], lw=0.7)
        ax.axvline(omegas_true_hz[k], color=LIME, lw=0.9, ls="--",
                   alpha=0.85)
        # Range zoomed around the carrier
        f_center = omegas_true_hz[k]
        f_min = max(f_center * 0.3, 0.5)
        f_max_x = f_center * 3.0
        ax.set_xlim(f_min, f_max_x)
        ax.set_ylim(1e-3, 1.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("freq [Hz]", fontsize=7.5, color=NAVY, labelpad=1)
        if k == 0:
            ax.set_ylabel("normalised |F|", fontsize=7.5, color=NAVY)
        for sp in ax.spines.values():
            sp.set_color(NAVY); sp.set_linewidth(0.4)
        ax.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    # Paper's parameters (Dragomiretskiy & Zosso 2014, Fig. 6)
    fs = 2000.0
    T = 1.0
    t, f_n, modes_true, _ = make_signal(fs=fs, T=T, noise_std=0.1)

    # Paper uses α=5000, K=3, τ=0, ε=1e-7
    # init=1 (uniform) sometimes misses middle frequencies; try a couple
    # of seeds and pick the run whose ω_k best span [2, 24, 288] Hz.
    print("Running VMD K=3, α=5000, τ=0 (denoising mode)...")
    best = None
    for init in (1, 2):
        for trial in range(3):
            u, _, omega = VMD(f_n, alpha=5000.0, tau=0., K=3,
                              DC=0, init=init, tol=1e-7)
            omegas_hz = np.sort(omega[-1]) * fs
            true_hz = np.array([2.0, 24.0, 288.0])
            cost = np.sum(np.abs(np.log(omegas_hz + 0.5)
                                 - np.log(true_hz + 0.5)))
            if best is None or cost < best[0]:
                best = (cost, u, omega, init, trial)
    cost, u, omega, init_used, trial_used = best
    print(f"  selected: init={init_used}, trial={trial_used}, "
          f"cost={cost:.3f}")
    omegas_final_hz = omega[-1] * fs
    # Sort modes by ascending centre frequency
    order = np.argsort(omegas_final_hz)
    omegas_sorted = omegas_final_hz[order]
    u_sorted = u[order]

    for k in range(3):
        amp_recov = np.std(u_sorted[k]) * np.sqrt(2)
        amp_true  = np.std(modes_true[k]) * np.sqrt(2)
        ratio = amp_recov / amp_true if amp_true > 0 else float('nan')
        print(f"  Mode {k+1}:  ω = {omegas_sorted[k]:7.2f} Hz  "
              f"(true {[2.0, 24.0, 288.0][k]:6.2f} Hz)   "
              f"amp ratio = {ratio:.3f}")

    plot_decomp(t, f_n, modes_true,
                [u_sorted[0], u_sorted[1], u_sorted[2]],
                omegas_sorted, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
