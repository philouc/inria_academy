"""
=======================================================================
EMD vs EEMD vs CEEMDAN: Mode-mixing demonstration (Wu & Huang canonical)
===========================================================================

Signal: continuous 4 Hz sine  +  intermittent 16 Hz burst (amplitude 0.5).
        fs = 500 Hz, T = 3 s. Bursts of 200 ms centred at t = 0.5, 1.5, 2.5 s.

Frequency ratio 4 : 1 — close enough to challenge EMD. Pure EMD's IMF1
captures the high-freq burst when present and the slow sine when the
burst is absent (mode mixing in time). The burst ends up *split* across
IMF1 and IMF2, and the slow sine bleeds into IMF1 between bursts.

EEMD and CEEMDAN inject noise that gives EMD a consistent high-frequency
reference everywhere, so the burst is captured cleanly in a single IMF
(typically IMF3 after noise residuals in IMF1-2), and the slow sine
occupies its own IMF without contamination.

Usage:     python 5.scenario4_mode_mixing.py            # interactive display
           python 5.scenario4_mode_mixing.py out.png    # save to file instead
=======================================================================
 Dependencies: numpy, scipy, matplotlib, EMD-signal.
 Requires:  pip install EMD-signal

 Author: Philippe Ciuciu
 Date: 05/07/2026
 Target: UnseenLabs / Inria Academy
=======================================================================
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter1d
from PyEMD import EMD, EEMD, CEEMDAN

NAVY     = "#0b1f3a"
LIME     = "#7CFC00"
RED      = "#e53935"
ORANGE   = "#d9820f"
GREEN    = "#1f8a4c"
GREY_MID = "#666"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888", "axes.linewidth": 0.4,
    "xtick.color": "#444", "ytick.color": "#444",
})


# ============================================================
# Signal
# ============================================================
def make_signal():
    """Wu & Huang canonical:  4 Hz sine  +  intermittent 16 Hz burst.

    Returns t, x, fs, true_sine, true_burst, envelope.
    """
    fs, T = 500.0, 3.0
    t = np.arange(int(fs * T)) / fs

    f_low, f_high = 4.0, 16.0
    sine = np.cos(2 * np.pi * f_low * t)

    # Three bursts at 0.5, 1.5, 2.5 s, 200 ms wide each, smoothed edges
    burst_centers = [0.5, 1.5, 2.5]
    burst_half = 0.10
    env = np.zeros_like(t)
    for c in burst_centers:
        env[(t > c - burst_half) & (t < c + burst_half)] = 1.0
    env = gaussian_filter1d(env, sigma=8)
    env = np.clip(env, 0.0, 1.0)
    burst = 0.5 * env * np.cos(2 * np.pi * f_high * t)

    x = sine + burst
    return t, x, fs, sine, burst, env


# ============================================================
# EMD-family decompositions
# ============================================================
def run_emd_family(x, t,
                   eemd_trials=100, eemd_noise=0.2,
                   ceemdan_trials=100, ceemdan_eps=0.2):
    out = {}
    t0 = time.time(); imfs = EMD()(x, t)
    out["EMD"]     = (imfs, time.time() - t0)
    t0 = time.time(); imfs = EEMD(trials=eemd_trials,
                                  noise_width=eemd_noise).eemd(x, t)
    out["EEMD"]    = (imfs, time.time() - t0)
    t0 = time.time(); imfs = CEEMDAN(trials=ceemdan_trials,
                                      epsilon=ceemdan_eps).ceemdan(x, t)
    out["CEEMDAN"] = (imfs, time.time() - t0)
    return out


def best_match_imf(imfs, target):
    """Return (idx, |correlation|) of the IMF best matching `target`."""
    tc = target - target.mean()
    tn = np.linalg.norm(tc) + 1e-12
    corrs = []
    for imf in imfs:
        ic = imf - imf.mean()
        corrs.append(np.dot(ic, tc) / (np.linalg.norm(ic) * tn + 1e-12))
    corrs = np.abs(np.array(corrs))
    return int(np.argmax(corrs)), float(corrs.max())


# ============================================================
# Plot
# ============================================================
def plot_mode_mixing_demo(t, x, sine, burst, env, results, savepath=None,
                          n_imfs_show=5):
    fig = plt.figure(figsize=(6.5, 4.6), dpi=200)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.48, 3.0],
        hspace=0.40, wspace=0.18,
        left=0.05, right=0.97, top=0.93, bottom=0.08,
    )

    # ----- Row 0: signal -----
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(t, -2.0, 2.0, where=env > 0.05,
                    color=ORANGE, alpha=0.12, lw=0)
    ax.plot(t, x, color=NAVY, lw=0.55)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-2.0, 2.0)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6.5)
    ax.set_title("(a)  Signal:  4 Hz sine  +  intermittent 16 Hz bursts   "
                 "(orange band = burst active)",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)
    for s in ax.spines.values():
        s.set_color("#888"); s.set_linewidth(0.4)

    methods = ["EMD", "EEMD", "CEEMDAN"]
    method_labels = ["(b)  EMD", "(c)  EEMD", "(d)  CEEMDAN"]

    # Pre-identify the burst-IMF and sine-IMF per method
    captures = {}
    for m in methods:
        imfs, _ = results[m]
        b_idx, b_corr = best_match_imf(imfs, burst)
        s_idx, s_corr = best_match_imf(imfs, sine)
        captures[m] = {"burst": (b_idx, b_corr), "sine": (s_idx, s_corr)}

    # ----- Row 1: three columns of IMFs -----
    for col, (m, lbl) in enumerate(zip(methods, method_labels)):
        ax = fig.add_subplot(gs[1, col])
        imfs, t_run = results[m]
        ax.fill_between(t, -100, 100, where=env > 0.05,
                        color=ORANGE, alpha=0.07, lw=0)

        burst_idx, burst_corr = captures[m]["burst"]
        sine_idx,  sine_corr  = captures[m]["sine"]

        n_avail = min(n_imfs_show, imfs.shape[0])
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_imfs_show))
        for k in range(n_avail):
            offset = (n_imfs_show - 1 - k) * 2.4
            amp = np.max(np.abs(imfs[k])) + 1e-9
            ax.plot(t, imfs[k] / amp + offset, color=colors[k], lw=0.45)
            tag = ""
            if k == burst_idx:
                tag = "  ← burst"
            elif k == sine_idx:
                tag = "  ← sine"
            ax.text(t[-1] + 0.04, offset, f"IMF{k+1}{tag}",
                    color=colors[k], fontsize=5.8, va="center",
                    fontweight="bold")

        # Overlay true burst aligned with the IMF that captures it
        if burst_idx < n_avail:
            offset = (n_imfs_show - 1 - burst_idx) * 2.4
            amp = np.max(np.abs(imfs[burst_idx])) + 1e-9
            ax.plot(t, burst / amp + offset, color=RED, lw=0.45, ls="--",
                    alpha=0.8)
        if sine_idx < n_avail:
            offset = (n_imfs_show - 1 - sine_idx) * 2.4
            amp = np.max(np.abs(imfs[sine_idx])) + 1e-9
            ax.plot(t, sine / amp + offset, color=LIME, lw=0.45, ls="--",
                    alpha=0.8)

        status = (f"  burst|IMF{burst_idx+1} r={burst_corr:.2f}  ·  "
                  f"sine|IMF{sine_idx+1} r={sine_corr:.2f}")
        ax.set_xlim(t[0], t[-1] * 1.32)
        ax.set_ylim(-1.7, n_imfs_show * 2.4)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("time [s]", fontsize=7, labelpad=1)
        ax.set_yticks([])
        ax.tick_params(labelsize=6.5)
        ax.set_title(f"{lbl}   ({imfs.shape[0]} IMFs, {t_run:.1f}s)\n"
                     f"{status}",
                     loc="left", color=NAVY, fontsize=6.5,
                     fontweight="bold", pad=2)

        if col == 0:
            ax.plot([], [], color=RED, lw=0.5, ls="--", label="true burst")
            ax.plot([], [], color=LIME, lw=0.5, ls="--", label="true sine")
            ax.legend(loc="lower right", frameon=False, fontsize=5.5,
                      handlelength=1.5, handletextpad=0.4,
                      labelcolor=GREY_MID)

        for s in ax.spines.values():
            s.set_color("#888"); s.set_linewidth(0.4)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    t, x, fs, sine, burst, env = make_signal()
    print("Running EMD / EEMD / CEEMDAN on Wu & Huang canonical signal...")
    print("  Signal: 4 Hz sine  +  intermittent 16 Hz burst (ratio 4:1)")
    results = run_emd_family(x, t)
    for m, (imfs, t_run) in results.items():
        b_idx, b_corr = best_match_imf(imfs, burst)
        s_idx, s_corr = best_match_imf(imfs, sine)
        print(f"  {m:<8s}: {imfs.shape[0]} IMFs ({t_run:.2f}s)  "
              f"burst best-match IMF{b_idx+1} (r={b_corr:.2f}),  "
              f"sine best-match IMF{s_idx+1} (r={s_corr:.2f})")
    plot_mode_mixing_demo(t, x, sine, burst, env, results, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
