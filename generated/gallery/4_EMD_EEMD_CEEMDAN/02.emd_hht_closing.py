"""
=======================================================================
 EMD + HHT closing-the-loop on the AM-FM benchmark: Second Script
=======================================================================

Pipeline:
  1. Generate the AM-FM benchmark signal x(t) = AM + FM
  2. EMD: extract IMF_1, IMF_2 via sifting (Cauchy SD criterion)
  3. HHT: apply Hilbert transform to each IMF
        a_k(t) = |z_k(t)|                  (instantaneous amplitude)
        phi_k(t) = unwrap(arg(z_k(t)))     (instantaneous phase)
        f_k(t) = (1/2π) dphi_k/dt          (instantaneous frequency)
  4. Compare with ground-truth modulation parameters
  5. Plot (a) HHT spectrum, (b) a_k(t) vs truth, (c) f_k(t) vs truth

The HHT spectrum is drawn as colored line ridges in the time-frequency
plane, with line color encoding the instantaneous amplitude. The dashed
cyan lines show the ground-truth parameters.

=======================================================================
 Run:  python 02.emd_hht_closing.py
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
from matplotlib.collections import LineCollection
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def am_fm_signal(fs, T,
                 fc1=80.0, fa=4.0,
                 fc2=180.0, fm=4.0, beta=12.5):
    """AM-FM benchmark, same convention as the rest of the deck."""
    N = int(round(fs * T))
    t = np.arange(N) / fs
    A = 1.0 + np.cos(2 * np.pi * fa * t)
    s_am = A * np.cos(2 * np.pi * fc1 * t)
    s_fm = np.cos(2 * np.pi * fc2 * t + beta * np.sin(2 * np.pi * fm * t))
    return t, s_am + s_fm


# ---------------------------------------------------------------------------
# Sifting (reproduced from emd_sifting_inaction.py for self-containment)
# ---------------------------------------------------------------------------
def _cubic_envelope(time, signal, extrema_idx):
    if len(extrema_idx) < 2:
        return np.zeros_like(time)
    t_pts = np.concatenate(([time[0]], time[extrema_idx], [time[-1]]))
    v_pts = np.concatenate(([signal[0]], signal[extrema_idx], [signal[-1]]))
    _, keep = np.unique(t_pts, return_index=True)
    keep = np.sort(keep)
    return CubicSpline(t_pts[keep], v_pts[keep])(time)


def _sift_step(time, signal):
    max_i, _ = find_peaks(signal)
    min_i, _ = find_peaks(-signal)
    e_up = _cubic_envelope(time, signal, max_i)
    e_lo = _cubic_envelope(time, signal, min_i)
    return signal - 0.5 * (e_up + e_lo)


def sift_imf(time, signal, sd_eps=0.05, max_iter=20):
    """Sift one IMF using Cauchy-SD stopping criterion."""
    h = signal.copy()
    for _ in range(max_iter):
        h_new = _sift_step(time, h)
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
        h = h_new
        if sd < sd_eps:
            break
    return h


def emd_n_imfs(time, signal, n_imfs=2, sd_eps=0.05):
    """Extract `n_imfs` IMFs by recursive sifting. Returns (imfs, residue)."""
    imfs = []
    h = signal.copy()
    for _ in range(n_imfs):
        imf = sift_imf(time, h, sd_eps=sd_eps)
        imfs.append(imf)
        h = h - imf
    return imfs, h


# ---------------------------------------------------------------------------
# Hilbert-Huang parameters
# ---------------------------------------------------------------------------
def hht_params(time, imf, smooth_freq_samples=5):
    """
    From an IMF, return (a(t), phi(t), f(t)) via the analytic signal.

    smooth_freq_samples : moving-average width used to denoise the IF
                          estimate (numerical differentiation of unwrapped
                          phase is noisy). Set to 1 to disable.
    """
    z = hilbert(imf)
    a = np.abs(z)
    phi = np.unwrap(np.angle(z))
    f = np.gradient(phi, time) / (2 * np.pi)
    if smooth_freq_samples > 1:
        f = uniform_filter1d(f, size=smooth_freq_samples)
    return a, phi, f


def mask_low_amplitude(a, f, threshold=0.15, edge_samples=40):
    """
    Replace IF values with NaN where (i) the instantaneous amplitude is
    below `threshold` (phase undefined) or (ii) within `edge_samples` of
    a signal boundary (Hilbert edge effects).
    """
    fm = f.astype(float).copy()
    fm[:edge_samples] = np.nan
    fm[-edge_samples:] = np.nan
    fm[a < threshold] = np.nan
    return fm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _colored_ridge(ax, time, f, a, vmax=2.0, lw=2.5):
    mask = ~np.isnan(f)
    if mask.sum() < 2:
        return None
    points = np.array([time[mask], f[mask]]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap="magma",
                        norm=plt.Normalize(0, vmax))
    lc.set_array(a[mask][:-1])
    lc.set_linewidth(lw)
    return ax.add_collection(lc)


def plot_hht_closing(time, a1, f1_masked, a2, f2_masked,
                     a_AM_true, f_AM_true, a_FM_true, f_FM_true,
                     fmax: float = 260.0,
                     savepath: str | None = None):
    """Three-panel HHT closing-the-loop figure."""
    NAVY_DARK = "#0b1f3a"
    RED       = "#c9191e"
    GREEN     = "#1f8a4c"
    CYAN      = "#7FE0FF"
    DARK_BG   = "#0a0a18"

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#888", "axes.linewidth": 0.6,
        "xtick.color": "#444", "ytick.color": "#444",
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })

    fig = plt.figure(figsize=(13.5, 5.45), dpi=144)

    # === LEFT: HHT spectrum ===
    ax_h = fig.add_axes([0.045, 0.11, 0.51, 0.78])
    ax_h.set_facecolor(DARK_BG)
    _colored_ridge(ax_h, time, f2_masked, a2)
    lc_ref = _colored_ridge(ax_h, time, f1_masked, a1)
    ax_h.plot(time, f_FM_true, ls="--", color=CYAN, lw=0.8, alpha=0.6,
              label=r"ground truth  $f_i(t)$")
    ax_h.plot(time, f_AM_true, ls="--", color=CYAN, lw=0.8, alpha=0.6)
    ax_h.set_xlim(time[0], time[-1])
    ax_h.set_ylim(0, fmax)
    ax_h.set_xlabel("Time  t  [s]")
    ax_h.set_ylabel("Frequency  [Hz]")
    ax_h.set_title("(a)  Hilbert-Huang spectrum",
                   loc="left", pad=6, color=NAVY_DARK)
    ax_h.legend(loc="upper right", frameon=False, fontsize=8.5,
                labelcolor=CYAN)

    cax = fig.add_axes([0.56, 0.11, 0.010, 0.78])
    cb = fig.colorbar(lc_ref, cax=cax)
    cb.set_label("instantaneous amplitude  a(t)", fontsize=9)
    cb.ax.tick_params(labelsize=8, colors="#444")
    cb.outline.set_linewidth(0.4)

    # === RIGHT-TOP: amplitudes ===
    ax_a = fig.add_axes([0.66, 0.55, 0.32, 0.34])
    ax_a.plot(time, a_FM_true, ls="--", color=CYAN, lw=1.0, alpha=0.95)
    ax_a.plot(time, a1, color=RED, lw=1.6,
              label=r"$a_{1}(t)\,\approx\,1$  (FM)")
    ax_a.plot(time, a_AM_true, ls="--", color=CYAN, lw=1.0, alpha=0.95)
    ax_a.plot(time, a2, color=GREEN, lw=1.6,
              label=r"$a_{2}(t)\,\approx\,1+\cos(2\pi f_{a} t)$  (AM)")
    ax_a.set_xlim(time[0], time[-1])
    ax_a.set_ylim(-0.1, 2.4)
    ax_a.set_xticklabels([])
    ax_a.set_ylabel("amplitude  a(t)")
    ax_a.set_title("(b)  Instantaneous amplitudes  vs  ground truth",
                   loc="left", pad=4, color=NAVY_DARK, fontsize=10)
    ax_a.legend(loc="upper right", frameon=False, fontsize=8)
    ax_a.grid(True, color="#eee", lw=0.4)

    # === RIGHT-BOTTOM: frequencies ===
    ax_f = fig.add_axes([0.66, 0.11, 0.32, 0.34])
    ax_f.plot(time, f_FM_true, ls="--", color=CYAN, lw=1.0, alpha=0.95,
              label="ground truth")
    ax_f.plot(time, f1_masked, color=RED, lw=1.6, label=r"$f_{1}(t)$  (FM)")
    ax_f.plot(time, f_AM_true, ls="--", color=CYAN, lw=1.0, alpha=0.95)
    ax_f.plot(time, f2_masked, color=GREEN, lw=1.6, label=r"$f_{2}(t)$  (AM)")
    ax_f.set_xlim(time[0], time[-1]); ax_f.set_ylim(0, fmax)
    ax_f.set_xlabel("Time  t  [s]"); ax_f.set_ylabel("frequency  [Hz]")
    ax_f.set_title("(c)  Instantaneous frequencies  vs  ground truth",
                   loc="left", pad=4, color=NAVY_DARK, fontsize=10)
    ax_f.legend(loc="upper right", frameon=False, fontsize=8)
    ax_f.grid(True, color="#eee", lw=0.4)

    if savepath:
        fig.savefig(savepath, dpi=144, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs, T = 1000, 1.0
    fc1, fa            = 80.0,  4.0
    fc2, fm, beta      = 180.0, 4.0, 12.5

    # 1. signal
    t, x = am_fm_signal(fs, T, fc1=fc1, fa=fa, fc2=fc2, fm=fm, beta=beta)

    # 2. EMD: get IMF_1 (FM) and IMF_2 (AM)
    imfs, residue = emd_n_imfs(t, x, n_imfs=2, sd_eps=0.05)
    imf1, imf2 = imfs

    # 3. HHT: instantaneous params on each IMF
    a1, _, f1 = hht_params(t, imf1)
    a2, _, f2 = hht_params(t, imf2)
    f1_masked = mask_low_amplitude(a1, f1, threshold=0.15, edge_samples=int(0.04*fs))
    f2_masked = mask_low_amplitude(a2, f2, threshold=0.15, edge_samples=int(0.04*fs))

    # 4. ground truth
    a_AM_true = 1.0 + np.cos(2 * np.pi * fa * t)
    f_AM_true = fc1 * np.ones_like(t)
    a_FM_true = np.ones_like(t)
    f_FM_true = fc2 + beta * fm * np.cos(2 * np.pi * fm * t)

    # 5. plot
    fig = plot_hht_closing(
        t, a1, f1_masked, a2, f2_masked,
        a_AM_true, f_AM_true, a_FM_true, f_FM_true,
        fmax=260.0, savepath="emd_hht_closing.png",
    )

    # diagnostics
    print("Recovery diagnostics:")
    edge = int(0.05 * fs)
    print(f"  a1 mean              = {np.mean(a1[edge:-edge]):.3f}   "
          f"(ground truth = 1.000)")
    print(f"  a2 range             = [{np.nanmin(a2[edge:-edge]):.3f}, "
          f"{np.nanmax(a2[edge:-edge]):.3f}]   "
          f"(ground truth = [0.000, 2.000])")
    mid = slice(int(0.1 * fs), int(0.9 * fs))
    print(f"  f1 range (FM)        = [{np.nanmin(f1_masked[mid]):.1f}, "
          f"{np.nanmax(f1_masked[mid]):.1f}] Hz   "
          f"(ground truth = [130.0, 230.0])")
    print(f"  f2 mean (AM carrier) = {np.nanmean(f2_masked[mid]):.1f} Hz   "
          f"(ground truth = 80.0)")
    print("Figure saved to: emd_hht_closing.png")
    plt.show()
