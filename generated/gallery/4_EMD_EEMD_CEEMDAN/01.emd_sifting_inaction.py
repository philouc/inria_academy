"""
=======================================================================
 EMD Sifting Algorithm in action — AM-FM benchmark: First script
=======================================================================

This script implements the sifting algorithm from scratch (cubic-spline
envelopes through local extrema, Cauchy SD stopping criterion) and
visualizes each of the 6 algorithmic steps on the AM-FM benchmark
signal already used in the deck:

  x(t) = (1 + cos(2π fa t)) * cos(2π fc1 t)         (AM at fc1=80 Hz)
       + cos(2π fc2 t + β sin(2π fm t))             (FM at fc2=180 Hz)

The 6 panels match the 6 steps of the sifting algorithm:
  1. Identify local extrema of x(t)
  2. Cubic-spline interpolation through ±extrema -> upper & lower envelopes
  3. Compute mean of envelopes m(t) = (e_up + e_lo) / 2
  4. Subtract: h_1(t) = x(t) - m(t)
  5. Iterate; stop when the Cauchy SD criterion falls below ε
  6. The converged h is IMF_1; residue r_1 = x - IMF_1 (next sifting target)

=======================================================================
 Run:  python 01.emd_sifting_inaction.py
 Dependencies: numpy, scipy, matplotlib.

 Author: Philippe Ciuciu
 Date: 05/01/2026
 Target: UnseenLabs / Inria Academy
=======================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Signal generation (same as in amfm_tf_comparison.py for consistency)
# ---------------------------------------------------------------------------
def am_fm_signal(fs, T,
                 fc1=80.0, fa=4.0,
                 fc2=180.0, fm=4.0, beta=12.5):
    """
    Build x(t) = AM + FM with the convention of the lecture deck.
        AM:  (1 + cos(2π fa t)) * cos(2π fc1 t)        ∈ [-2, +2]
        FM:  cos(2π fc2 t + β sin(2π fm t))            ∈ [-1, +1]
    """
    N = int(round(fs * T))
    t = np.arange(N) / fs
    A = 1.0 + np.cos(2 * np.pi * fa * t)
    s_am = A * np.cos(2 * np.pi * fc1 * t)
    s_fm = np.cos(2 * np.pi * fc2 * t + beta * np.sin(2 * np.pi * fm * t))
    return t, s_am + s_fm


# ---------------------------------------------------------------------------
# Sifting primitives
# ---------------------------------------------------------------------------
def cubic_envelope(time, signal, extrema_idx):
    """
    Cubic-spline envelope passing through the values of `signal` at
    `extrema_idx`. The signal's first and last samples are added as
    boundary anchors to prevent the spline from going wild near the edges
    (Huang's classic "spline overshoot" issue; more elaborate boundary
    extensions exist — see e.g. mirror reflection — this minimal version
    is enough for the pedagogical example).
    """
    if len(extrema_idx) < 2:
        return np.zeros_like(time)
    t_pts = np.concatenate(([time[0]], time[extrema_idx], [time[-1]]))
    v_pts = np.concatenate(([signal[0]], signal[extrema_idx], [signal[-1]]))
    # remove duplicate t's at boundaries to keep monotonic input
    _, keep = np.unique(t_pts, return_index=True)
    keep = np.sort(keep)
    return CubicSpline(t_pts[keep], v_pts[keep])(time)


def sift_step(time, signal):
    """
    One sifting iteration. Returns
      h     : signal - mean-of-envelopes
      e_up  : upper cubic-spline envelope
      e_lo  : lower cubic-spline envelope
      m     : mean (e_up + e_lo) / 2
      max_i : indices of local maxima
      min_i : indices of local minima
    """
    max_i, _ = find_peaks(signal)
    min_i, _ = find_peaks(-signal)
    e_up = cubic_envelope(time, signal, max_i)
    e_lo = cubic_envelope(time, signal, min_i)
    m = 0.5 * (e_up + e_lo)
    h = signal - m
    return h, e_up, e_lo, m, max_i, min_i


def sift_imf(time, signal, sd_eps=0.05, max_iter=15):
    """
    Iterate sift_step until the Cauchy SD criterion falls below sd_eps.

    SD_n = sum_t |h_{n-1}(t) - h_n(t)|^2 / sum_t |h_{n-1}(t)|^2

    Returns the converged IMF and the SD history.
    """
    h = signal.copy()
    sd_hist = []
    for _ in range(max_iter):
        h_new, _, _, _, _, _ = sift_step(time, h)
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
        sd_hist.append(sd)
        h = h_new
        if sd < sd_eps:
            break
    return h, sd_hist


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sifting_panel(time, x,
                       t_view_end: float = 0.25,
                       sd_eps: float = 0.05,
                       savepath: str | None = None):
    """
    6-panel figure showing each step of sifting on `x` restricted to
    the view window [0, t_view_end].
    """
    NAVY      = "#0b1f3a"
    NAVY_DARK = "#0b1f3a"
    RED       = "#c9191e"
    BLUE      = "#1565c0"
    GREEN     = "#1f8a4c"
    GREY      = "#777777"
    LIGHT     = "#cccccc"

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.edgecolor": "#888", "axes.linewidth": 0.6,
        "xtick.color": "#444", "ytick.color": "#444",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    # run one iteration (for steps 1-4 of the figure)
    h_1, e_up, e_lo, m, max_idx, min_idx = sift_step(time, x)
    # run to convergence (for steps 5-6)
    imf1, sd_hist = sift_imf(time, x, sd_eps=sd_eps)
    residue = x - imf1

    mask = time <= t_view_end
    t_v = time[mask]

    fig = plt.figure(figsize=(13.5, 5.45), dpi=144)
    L, R, B, Top = 0.05, 0.985, 0.10, 0.88
    hgap, vgap = 0.04, 0.13
    panel_w = (R - L - 2 * hgap) / 3
    panel_h = (Top - B - vgap) / 2

    def add_panel(row, col):
        x0 = L + col * (panel_w + hgap)
        y0 = B + (1 - row) * (panel_h + vgap)
        return fig.add_axes([x0, y0, panel_w, panel_h])

    def style(ax, title):
        ax.set_title(title, loc="left", pad=4, color=NAVY_DARK, fontsize=10)
        ax.axhline(0, color=LIGHT, lw=0.4, zorder=0)
        ax.set_xlim(0, t_view_end)
        ax.tick_params(labelsize=8)

    # --- 1. extrema ---
    ax = add_panel(0, 0)
    ax.plot(t_v, x[mask], color=NAVY, lw=0.7, alpha=0.7)
    mi = max_idx[time[max_idx] <= t_view_end]
    ni = min_idx[time[min_idx] <= t_view_end]
    ax.scatter(time[mi], x[mi], s=10, c=RED, zorder=3, label="maxima")
    ax.scatter(time[ni], x[ni], s=10, c=BLUE, zorder=3, label="minima")
    style(ax, "1.  Identify extrema of  x(t)")
    ax.set_ylim(-3.2, 3.2); ax.set_ylabel("Amp.")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5,
              ncol=2, handletextpad=0.3, columnspacing=0.8)

    # --- 2. splines ---
    ax = add_panel(0, 1)
    ax.plot(t_v, x[mask], color=NAVY, lw=0.6, alpha=0.45)
    ax.plot(t_v, e_up[mask], color=RED,  lw=1.4, ls="--",
            label=r"$e_{\mathrm{up}}(t)$")
    ax.plot(t_v, e_lo[mask], color=BLUE, lw=1.4, ls="--",
            label=r"$e_{\mathrm{lo}}(t)$")
    style(ax, "2.  Cubic spline through extrema")
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="upper right", frameon=False, fontsize=8,
              ncol=2, handletextpad=0.3, columnspacing=0.8)

    # --- 3. mean ---
    ax = add_panel(0, 2)
    ax.plot(t_v, x[mask],   color=NAVY, lw=0.5, alpha=0.25)
    ax.plot(t_v, e_up[mask], color=RED,  lw=0.8, ls="--", alpha=0.5)
    ax.plot(t_v, e_lo[mask], color=BLUE, lw=0.8, ls="--", alpha=0.5)
    ax.plot(t_v, m[mask],   color=GREEN, lw=1.8,
            label=r"$m(t)$")
    style(ax, "3.  Mean of the envelopes")
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # --- 4. subtract: h_1 ---
    ax = add_panel(1, 0)
    ax.plot(t_v, h_1[mask], color=NAVY, lw=0.7)
    style(ax, r"4.  $h_{1}(t)\,=\,x(t)\,-\,m(t)$")
    ax.set_ylim(-2.5, 2.5); ax.set_ylabel("Amp.")
    ax.set_xlabel("Time  t  [s]")

    # --- 5. Cauchy convergence ---
    ax = add_panel(1, 1)
    iters = np.arange(1, len(sd_hist) + 1)
    ax.plot(iters, sd_hist, marker="o", color=NAVY, lw=1.4,
            markerfacecolor=RED, markeredgecolor="white", markersize=6)
    ax.axhline(sd_eps, color=GREEN, lw=1.2, ls="--",
               label=rf"$\epsilon={sd_eps}$")
    ax.set_yscale("log")
    ax.set_xticks(iters)
    ax.tick_params(labelsize=8)
    ax.set_title("5.  Cauchy SD criterion",
                 loc="left", pad=4, color=NAVY_DARK, fontsize=10)
    ax.set_xlabel("sifting iteration  n")
    ax.set_ylabel(r"$\mathrm{SD}_{n}$")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(True, color="#eee", lw=0.4, which="both")

    # --- 6. IMF + residue ---
    ax = add_panel(1, 2)
    ax.plot(t_v, imf1[mask],   color=RED,   lw=0.7, alpha=0.85,
            label=r"$\mathrm{IMF}_{1}(t)$")
    ax.plot(t_v, residue[mask], color=GREEN, lw=1.4,
            label=r"$r_{1}(t)$")
    style(ax, "6.  Extracted IMF and residue")
    ax.set_ylim(-2.5, 2.5); ax.set_xlabel("Time  t  [s]")
    ax.legend(loc="upper right", frameon=False, fontsize=8,
              ncol=2, handletextpad=0.3, columnspacing=0.8)

    # banner
    ax_t = fig.add_axes([L, 0.92, R - L, 0.06]); ax_t.axis("off")
    ax_t.text(0.0, 0.5,
              "Sifting in action  —  AM-FM benchmark  "
              "($f_{c_1}=80$ Hz AM, $f_{c_2}=180$ Hz FM)",
              ha="left", va="center", color=NAVY_DARK,
              fontsize=12, fontweight="bold")
    ax_t.text(1.0, 0.5,
              rf"shown on $t\in[0,\,{t_view_end:.2f}\,$s$]$",
              ha="right", va="center", color=GREY, fontsize=9, style="italic")

    if savepath:
        fig.savefig(savepath, dpi=144, bbox_inches="tight", facecolor="white")
    return fig, sd_hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs, T = 1000, 1.0
    t, x = am_fm_signal(fs, T)

    fig, sd_hist = plot_sifting_panel(
        t, x, t_view_end=0.25, sd_eps=0.05,
        savepath="emd_sifting_panel.png",
    )
    print(f"Sifting converged in {len(sd_hist)} iterations")
    print(f"Final SD = {sd_hist[-1]:.5f}")
    print("Figure saved to: emd_sifting_panel.png")
    plt.show()
