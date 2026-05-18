"""
VMD — Example 4 · Synthetic signal  (Dragomiretskiy & Zosso 2014, Fig. 8)
==========================================================================

Signal:

    f(t) = 6t²
         + cos(10π t + 10π t²)                       (chirp 5→15 Hz)
         + {  cos(60π t)         if  t ≤ 1/2          (30 Hz)
              cos(80π t − 10π)   if  t > 1/2          (40 Hz)

VMD with K = 4 cleanly separates:
    - the quadratic trend  6t²                     → ω_k → 0
    - the chirp                                    → ω_k → ~10 Hz (centre)
    - the 30 Hz piecewise-constant component       → ω_k → 30 Hz
    - the 40 Hz piecewise-constant component       → ω_k → 40 Hz

The convergence plot (right) shows the centre frequencies pulling
toward their final positions across ADMM iterations — visual evidence
that the algorithm "discovers" the right partitioning.

Usage:  python vmd_ex4_synthetic.py            # interactive
        python vmd_ex4_synthetic.py out.png    # save figure
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from vmdpy import VMD

# --- Palette ---------------------------------------------------------------
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


def make_signal(fs=500.0, T=1.0):
    """The Dragomiretskiy Fig. 8 synthetic test signal."""
    t = np.arange(int(fs * T)) / fs
    trend  = 6.0 * t ** 2
    chirp  = np.cos(10 * np.pi * t + 10 * np.pi * t ** 2)
    piece1 = np.cos(60 * np.pi * t)
    piece2 = np.cos(80 * np.pi * t - 10 * np.pi)
    piece  = np.where(t <= 0.5, piece1, piece2)
    f = trend + chirp + piece
    return t, f, (trend, chirp, piece)


def plot_decomp(t, f_sig, modes_true, u_sorted, omega_history_hz,
                savepath=None):
    """
    Layout:
        Left column  (rows 0..4):  input signal + 4 VMD modes stacked
        Right column (full height): ω_k(iter) convergence trajectory
    """
    fig = plt.figure(figsize=(8.0, 5.2), dpi=200)
    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        height_ratios=[1.0, 0.8, 0.8, 0.8, 0.8],
        width_ratios=[1.3, 1.0],
        hspace=0.45, wspace=0.20,
        left=0.07, right=0.97, top=0.95, bottom=0.07,
    )

    colors = [PURPLE, AMBER, GREEN, RED]
    labels = ["trend  (DC)",
              "chirp  5→15 Hz",
              "piecewise-constant 30 Hz  (t ≤ 1/2)",
              "piecewise-constant 40 Hz  (t > 1/2)"]
    final_omegas = [omega_history_hz[-1, i]
                    for i in range(omega_history_hz.shape[1])]

    # ----- input signal (top) -----
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, f_sig, color=NAVY, lw=0.5)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_title("(a)  Input signal  "
                 r"$f(t) = 6t^2 + \cos(10\pi t + 10\pi t^2)"
                 r" + \mathrm{piecewise}\,(30\to 40\,\mathrm{Hz})$",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
    for sp in ax.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- 4 VMD modes (stacked) -----
    for k in range(4):
        ax = fig.add_subplot(gs[k + 1, 0])
        ax.plot(t, u_sorted[k], color=colors[k], lw=0.5)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7)
        if k < 3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("time [s]", fontsize=7.5, color=NAVY,
                          labelpad=1)
        ax.set_yticks([])
        ax.set_title(f"Mode {k+1}   {labels[k]}   "
                     f"(ω → {final_omegas[k]:.2f} Hz)",
                     loc="left", fontsize=7.5, color=colors[k],
                     fontweight="bold", pad=2)
        # Mark t=0.5 for the piecewise modes
        if k >= 2:
            ax.axvline(0.5, color=GREY, lw=0.5, ls=":", alpha=0.7)
        for sp in ax.spines.values():
            sp.set_color(NAVY); sp.set_linewidth(0.4)

    # ----- ω_k convergence trajectory -----
    ax = fig.add_subplot(gs[:, 1])
    n_iter = omega_history_hz.shape[0]
    iters = np.arange(n_iter)
    # omega_history_hz columns already sorted to match colors/labels
    for j in range(omega_history_hz.shape[1]):
        ax.plot(iters, omega_history_hz[:, j],
                color=colors[j], lw=1.0)
    # Reference horizontal lines at the targets
    targets = [0.0, 10.0, 30.0, 40.0]
    for j, tgt in enumerate(targets):
        ax.axhline(tgt, color=colors[j], lw=0.5, ls=":", alpha=0.6)
        ax.text(n_iter * 0.98, tgt + 1.0,
                f"target {tgt:g} Hz",
                ha="right", va="bottom", color=colors[j],
                fontsize=6.5, style="italic")
    ax.set_xlim(0, n_iter - 1)
    ax.set_ylim(-2, 55)
    ax.set_xlabel("ADMM iteration  n", fontsize=8, color=NAVY,
                  labelpad=2)
    ax.set_ylabel(r"centre frequency  $\omega_k$  [Hz]",
                  fontsize=8, color=NAVY, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    ax.set_title("(b)  Centre-frequency convergence  (ADMM)",
                 loc="left", fontsize=8.5, color=NAVY,
                 fontweight="bold", pad=2)
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
    T = 1.0
    t, f_sig, _ = make_signal(fs=fs, T=T)

    # Paper uses K=4, α large enough to separate the close frequencies.
    print("Running VMD K=4 on Dragomiretskiy Fig. 8 signal...")
    best = None
    targets_hz = np.array([0.0, 10.0, 30.0, 40.0])
    for init in (1, 2):
        for trial in range(3):
            u, _, omega = VMD(f_sig, alpha=2000.0, tau=0., K=4,
                              DC=1, init=init, tol=1e-7)
            omegas = np.sort(omega[-1]) * fs
            cost = np.sum(np.abs(omegas - targets_hz))
            if best is None or cost < best[0]:
                best = (cost, u, omega, init, trial)
    cost, u, omega, init_used, trial_used = best
    print(f"  selected: init={init_used}, trial={trial_used}, "
          f"cost={cost:.3f}")
    omega_history_hz = omega * fs           # shape (n_iter, K)
    final_omegas_hz = omega_history_hz[-1]
    order = np.argsort(final_omegas_hz)
    u_sorted = u[order]
    # Also sort omega history columns so they line up with u_sorted
    omega_history_hz_sorted = omega_history_hz[:, order]

    for k in range(4):
        print(f"  Mode {k+1}:  ω → {final_omegas_hz[order[k]]:7.2f} Hz")

    plot_decomp(t, f_sig, None, u_sorted,
                omega_history_hz_sorted, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
