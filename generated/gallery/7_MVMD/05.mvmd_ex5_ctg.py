"""
MVMD — Example 5 · Cardiotocography (FHR + UC)
==============================================

Cardiotocography (CTG) records two biomedical signals simultaneously:

    -  FHR  (Fetal Heart Rate)         around 2 Hz fundamental (120 bpm)
                                        + 2nd / 3rd harmonics
                                        + beat-to-beat variability
    -  UC   (Uterine Contractions)     very slow envelope, < 0.2 Hz
                                        with active contraction periods

The two channels share the same overall recording but encode different
physiological processes.  MVMD with K = 4 extracts:

    -  Mode 1: UC slow rhythm (very low ω)
    -  Mode 2: FHR fundamental (~2 Hz)
    -  Mode 3: FHR 2nd harmonic
    -  Mode 4: high-frequency variability / artefacts

Usage:  python mvmd_ex5_ctg.py            # interactive
        python mvmd_ex5_ctg.py out.png    # save
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

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


def make_signal(fs=50.0, T=120.0, seed=0):
    """Synthetic CTG: FHR and UC channels, 2 minutes at 50 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * T)) / fs

    # ----- UC channel: slow contractions with envelope -----
    # baseline + 2-3 contractions per 2 minutes (~0.02 Hz)
    uc_baseline = 30.0  # mmHg
    # Each contraction: Gaussian bump around peaks
    uc_signal = np.full_like(t, uc_baseline)
    peak_times = [25, 65, 100]   # 3 contractions over 120 s
    for tp in peak_times:
        sigma = 8.0
        uc_signal += 40.0 * np.exp(-((t - tp) ** 2) / (2 * sigma ** 2))
    uc_signal += 0.5 * rng.standard_normal(len(t))

    # ----- FHR channel: ~2 Hz fundamental + 4 Hz harmonic + variability -----
    fhr_baseline = 140.0  # bpm
    # The FHR "signal" is a beating rhythm — we simulate it as a sum of
    # 2 Hz and 4 Hz sinusoids modulated by the UC envelope (loose coupling)
    fhr_fund = 8.0 * np.cos(2 * np.pi * 2.0 * t)
    fhr_h2   = 3.0 * np.cos(2 * np.pi * 4.0 * t)
    # beat-to-beat variability — slow random walk at high freq
    variability = 2.5 * np.cos(2 * np.pi * 8.0 * t
                                + np.cumsum(rng.standard_normal(len(t))) * 0.05)
    # Slight coupling: heart-rate dips during contractions (UC envelope effect)
    uc_envelope = (uc_signal - uc_baseline) / 40.0
    fhr_signal = (fhr_baseline + fhr_fund + fhr_h2 + variability
                   - 5.0 * uc_envelope
                   + 0.5 * rng.standard_normal(len(t)))

    # Stack 2 channels
    x = np.stack([fhr_signal, uc_signal])
    # Detrend each channel (subtract mean) — MVMD has DC=1 anyway
    x = x - x.mean(axis=1, keepdims=True)
    return t, x


def main(savepath=None):
    fs = 50.0
    T = 120.0
    t, x = make_signal(fs=fs, T=T)
    print(f"CTG signal: shape {x.shape}, duration {T:g} s")

    K = 4
    print(f"MVMD K = {K} ...")
    # UC is very slow → DC=1 to lock first mode at 0
    targets = np.array([0.0, 2.0, 4.0, 8.0])
    best = None
    for init in (1, 2):
        for trial in range(5):
            u_try, _, om_try = MVMD(x, alpha=1000.0, tau=0., K=K,
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

    # --- Plot ---
    # 5 rows × 2 columns: input, mode1, mode2, mode3, mode4 × (FHR, UC)
    fig = plt.figure(figsize=(8.5, 5.4), dpi=200)
    gs = gridspec.GridSpec(
        5, 2, figure=fig,
        height_ratios=[1.0, 0.85, 0.85, 0.85, 0.85],
        hspace=0.45, wspace=0.18,
        left=0.07, right=0.97, top=0.94, bottom=0.07,
    )

    chan_labels = ["FHR  (fetal heart rate)", "UC  (uterine contractions)"]
    mode_meta = [
        ("Mode 1", "UC slow rhythm", PURPLE),
        ("Mode 2", "FHR fundamental ≈ 2 Hz", GREEN),
        ("Mode 3", "FHR 2nd harmonic", AMBER),
        ("Mode 4", "high-freq variability", RED),
    ]

    # Row 0: input
    for c in range(2):
        ax = fig.add_subplot(gs[0, c])
        ax.plot(t, x[c], color=NAVY, lw=0.45)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelsize=7)
        ax.set_xticklabels([])
        ax.set_title(chan_labels[c], loc="left", fontsize=9, color=NAVY,
                     fontweight="bold", pad=2)
        if c == 0:
            ax.set_ylabel("input", fontsize=7.5, color=NAVY)
        for sp in ax.spines.values():
            sp.set_color(NAVY); sp.set_linewidth(0.4)

    # Rows 1-4: modes
    for k in range(K):
        head, sub, color = mode_meta[k]
        for c in range(2):
            ax = fig.add_subplot(gs[k + 1, c])
            ax.plot(t, u[k, c], color=color, lw=0.45)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=7)
            if k < K - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time [s]", fontsize=7.5, color=NAVY,
                              labelpad=1)
            if c == 0:
                ax.set_ylabel(f"{head}\n{sub}\nω = {omegas_hz[k]:.2f} Hz",
                              fontsize=6.5, color=color,
                              fontweight="bold")
            for sp in ax.spines.values():
                sp.set_color(NAVY); sp.set_linewidth(0.4)

    fig.suptitle("MVMD on a synthetic CTG  ·  bivariate FHR + UC  ·  "
                 "shared ω_k across both physiological signals",
                 fontsize=9.5, color=NAVY, fontweight="bold",
                 y=0.985)

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
