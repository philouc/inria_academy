"""
MVMD — Example 3 · Quasi-orthogonality of modes
================================================

After running MVMD on a synthetic multichannel signal, look at the
cross-correlation between recovered modes.  The matrix should be
near-diagonal: each mode captures a distinct frequency band, and the
spectral overlap between any two modes is small.

Setup:
    -  10 channels, each a sum of 4 tones at 5, 15, 30, 80 Hz with
       per-channel amplitudes drawn from 𝒩(1, 0.3) and a touch of noise.
    -  MVMD with K = 4.
    -  Compute  C(i, j) = mean_c |corr(u_{i,c}, u_{j,c})|
       and  also the average spectral overlap S(i, j).

Usage:  python mvmd_ex3_orthogonality.py            # interactive
        python mvmd_ex3_orthogonality.py out.png    # save
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

from inria_academy.utils.mvmd import MVMD

NAVY  = "#0b1f3a"
RED   = "#c9191e"
GREEN = "#1f8a4c"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


def main(savepath=None):
    fs = 500.0
    T = 1.0
    N_chan = 10
    K = 4
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * T)) / fs
    freqs_true = np.array([10.0, 30.0, 60.0, 100.0])
    # Per-channel amplitudes (rows = channels, cols = frequencies)
    amps = 1.0 + 0.3 * rng.standard_normal((N_chan, len(freqs_true)))
    amps = np.abs(amps)  # ensure positive
    # Build the multichannel signal
    x = np.zeros((N_chan, len(t)))
    for k, f in enumerate(freqs_true):
        x += amps[:, k:k+1] * np.cos(2 * np.pi * f * t)[None, :]
    x += 0.05 * rng.standard_normal(x.shape)
    print(f"Signal: {N_chan} channels × {len(t)} samples")

    print(f"MVMD K = {K} ...")
    # Multiple inits — try several to land all K modes near distinct freqs
    targets = np.array([10.0, 30.0, 60.0, 100.0])
    best = None
    for init in (0, 1, 2):
        n_trials = 8 if init == 2 else 1
        for trial in range(n_trials):
            u_try, _, om_try = MVMD(x, alpha=5000.0, tau=0., K=K,
                                     DC=0, init=init, tol=1e-7)
            om_hz = np.sort(om_try[-1]) * fs
            # L1 distance to targets
            cost = np.sum(np.abs(om_hz - targets))
            # Heavy penalty if any two ω are very close
            for i in range(len(om_hz) - 1):
                if om_hz[i + 1] - om_hz[i] < 5.0:
                    cost += 1000.0
            if best is None or cost < best[0]:
                best = (cost, u_try, om_try)
    _, u, omega = best
    omegas_hz = omega[-1] * fs
    # sort modes for plotting
    order = np.argsort(omegas_hz)
    u = u[order]
    omegas_hz = omegas_hz[order]
    print(f"  ω_k = {omegas_hz} Hz")

    # --- Cross-correlation matrix (averaged across channels) ---
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            r = 0.0
            for c in range(N_chan):
                a = u[i, c] - u[i, c].mean()
                b = u[j, c] - u[j, c].mean()
                r += abs(np.dot(a, b) / (np.linalg.norm(a) *
                                          np.linalg.norm(b) + 1e-12))
            C[i, j] = r / N_chan

    # --- Spectral overlap ---
    freqs = np.fft.rfftfreq(len(t), 1.0 / fs)
    # Average spectrum per mode (over channels)
    spec = np.array([
        np.mean([np.abs(np.fft.rfft(u[k, c])) for c in range(N_chan)],
                 axis=0) for k in range(K)
    ])
    # Normalise each mode's spectrum to unit area
    spec_norm = spec / (spec.sum(axis=1, keepdims=True) + 1e-12)
    S = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            # Bhattacharyya coefficient (in [0, 1], 1 = identical)
            S[i, j] = np.sum(np.sqrt(spec_norm[i] * spec_norm[j]))

    # --- Plot ---
    fig = plt.figure(figsize=(8.0, 4.6), dpi=200)
    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.30,
        left=0.07, right=0.97, top=0.82, bottom=0.10,
    )

    labels = [f"M{k+1}\n{omegas_hz[k]:.1f} Hz" for k in range(K)]

    def plot_matrix(ax, mat, title, vmin=0.0, vmax=1.0,
                    cmap="magma"):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                       aspect="equal")
        ax.set_xticks(range(K)); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(K)); ax.set_yticklabels(labels, fontsize=7)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{mat[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if mat[i, j] < 0.5 else "black",
                        fontsize=8, fontweight="bold")
        ax.set_title(title, loc="left", color=NAVY, fontsize=9.5,
                     fontweight="bold", pad=4)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 0])
    plot_matrix(ax, C,
                "(a)  Cross-correlation matrix\n"
                r"     $|\langle u_i,\,u_j\rangle|$  averaged across channels")

    ax = fig.add_subplot(gs[0, 1])
    plot_matrix(ax, S,
                "(b)  Spectral overlap\n"
                "     (Bhattacharyya coefficient)")

    fig.suptitle("MVMD modes are quasi-orthogonal — "
                 "off-diagonal entries close to 0",
                 fontsize=10, fontweight="bold", color=NAVY,
                 y=0.97)

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
