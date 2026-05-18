"""
=============================================================================
 SPWVD vs HHT — side-by-side comparison on the AM-FM benchmark: Third script
=============================================================================

Two paradigms, same signal:

  (a) SPWVD : Wigner-Ville smoothed by a separable Cohen-class kernel.
              Quadratic in the signal, resolution-limited by the kernel
              widths (σ_t, σ_f). Cross-terms appear as oscillating
              ripples that the kernel suppresses at the cost of ridge
              concentration.

  (b) HHT   : EMD decomposes x into IMFs; Hilbert is then applied
              linearly to each IMF, returning instantaneous amplitude
              and frequency. Plotted as colored line ridges in the TF
              plane, with line color = a(t). No kernel, no smoothing —
              the ridges collapse to a single line per IMF.

The two panels share axes and a colormap on purpose: the visual
difference is information content, not rendering choices.

=======================================================================
 Run:  python 03.spwvd_vs_hht.py
 Dependencies: numpy, scipy, matplotlib.

 Author: Philippe Ciuciu
 Date: 05/02/2026
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
from scipy.ndimage import gaussian_filter, uniform_filter1d


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def am_fm_signal(fs, T,
                 fc1=80.0, fa=4.0,
                 fc2=180.0, fm=4.0, beta=12.5):
    N = int(round(fs * T))
    t = np.arange(N) / fs
    A = 1.0 + np.cos(2 * np.pi * fa * t)
    s_am = A * np.cos(2 * np.pi * fc1 * t)
    s_fm = np.cos(2 * np.pi * fc2 * t + beta * np.sin(2 * np.pi * fm * t))
    return t, s_am + s_fm


# ---------------------------------------------------------------------------
# WVD / SPWVD
# ---------------------------------------------------------------------------
def wigner_ville(x, fs):
    """
    Discrete Wigner-Ville distribution on the analytic signal.
    FFT bin k → true frequency k * fs / (2N); full bins span [0, fs/2].
    """
    z = hilbert(x)
    N = len(z)
    tfr = np.zeros((N, N), dtype=complex)
    for n in range(N):
        taumax = min(n, N - 1 - n, N // 2 - 1)
        if taumax > 0:
            tau = np.arange(-taumax, taumax + 1)
            tfr[tau % N, n] = z[n + tau] * np.conj(z[n - tau])
    tfr = np.fft.fft(tfr, axis=0)
    W = np.real(tfr)
    freq = np.arange(N) * fs / (2 * N)
    time = np.arange(N) / fs
    return freq, time, W


def smoothed_pseudo_wvd(x, fs, sigma_freq_bins=2.5, sigma_time_samples=12.0):
    f, t, W = wigner_ville(x, fs)
    return f, t, gaussian_filter(W, sigma=(sigma_freq_bins, sigma_time_samples))


# ---------------------------------------------------------------------------
# EMD sifting
# ---------------------------------------------------------------------------
def _cubic_envelope(time, signal, idx):
    if len(idx) < 2:
        return np.zeros_like(time)
    t_pts = np.concatenate(([time[0]], time[idx], [time[-1]]))
    v_pts = np.concatenate(([signal[0]], signal[idx], [signal[-1]]))
    _, keep = np.unique(t_pts, return_index=True); keep = np.sort(keep)
    return CubicSpline(t_pts[keep], v_pts[keep])(time)


def _sift_step(time, signal):
    mi, _ = find_peaks(signal); ni, _ = find_peaks(-signal)
    return signal - 0.5 * (_cubic_envelope(time, signal, mi)
                            + _cubic_envelope(time, signal, ni))


def sift_imf(time, signal, sd_eps=0.05, max_iter=20):
    h = signal.copy()
    for _ in range(max_iter):
        h_new = _sift_step(time, h)
        sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
        h = h_new
        if sd < sd_eps:
            break
    return h


# ---------------------------------------------------------------------------
# HHT params
# ---------------------------------------------------------------------------
def hht_params(time, imf, smooth=5):
    z = hilbert(imf)
    a = np.abs(z)
    f = np.gradient(np.unwrap(np.angle(z)), time) / (2 * np.pi)
    if smooth > 1:
        f = uniform_filter1d(f, size=smooth)
    return a, f


def mask_low_amplitude(a, f, threshold=0.15, edge_samples=40):
    fm_ = f.astype(float).copy()
    fm_[:edge_samples] = np.nan; fm_[-edge_samples:] = np.nan
    fm_[a < threshold] = np.nan
    return fm_


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_spwvd_vs_hht(time, spwvd_f, spwvd_W,
                     imf_a, imf_f_masked,
                     f_AM_true, f_FM_true,
                     fmax: float = 260.0,
                     spwvd_vmin: float = 0.0,
                     spwvd_vmax: float = 0.20,
                     savepath: str | None = None):
    """Side-by-side SPWVD (left) vs HHT spectrum (right)."""
    NAVY_DARK = "#0b1f3a"; GREY = "#777777"
    CYAN = "#7FE0FF"; DARK_BG = "#0a0a18"

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#888", "axes.linewidth": 0.6,
        "xtick.color": "#444", "ytick.color": "#444",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(13.5, 5.45), dpi=144)

    # === (a) SPWVD ===
    ax_l = fig.add_axes([0.05, 0.13, 0.42, 0.76])
    Z = spwvd_W / max(spwvd_W.max(), 1e-20)
    ax_l.imshow(Z, aspect="auto", origin="lower",
                extent=[time[0], time[-1], spwvd_f[0], spwvd_f[-1]],
                cmap="magma", vmin=spwvd_vmin, vmax=spwvd_vmax,
                interpolation="bilinear")
    ax_l.plot(time, f_AM_true, ls="--", color=CYAN, lw=0.9, alpha=0.8)
    ax_l.plot(time, f_FM_true, ls="--", color=CYAN, lw=0.9, alpha=0.8,
              label=r"ground truth  $f_i(t)$")
    ax_l.set_xlim(time[0], time[-1]); ax_l.set_ylim(0, fmax)
    ax_l.set_xlabel("Time  t  [s]"); ax_l.set_ylabel("Frequency  [Hz]")
    ax_l.set_title("(a)  SPWVD  —  Cohen-class kernel smoothing",
                   loc="left", pad=6, color=NAVY_DARK)
    ax_l.legend(loc="upper right", frameon=False, fontsize=8.5,
                labelcolor=CYAN)
    ax_l.text(0.01, -0.18,
              "ridges visible but smeared in t and f — limited by kernel σ",
              transform=ax_l.transAxes,
              color=GREY, fontsize=9, ha="left")

    # === (b) HHT spectrum ===
    ax_r = fig.add_axes([0.55, 0.13, 0.42, 0.76])
    ax_r.set_facecolor(DARK_BG)

    def colored_ridge(ax, time, f, a, vmax=2.0, lw=2.5):
        mask = ~np.isnan(f)
        if mask.sum() < 2:
            return None
        pts = np.array([time[mask], f[mask]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="magma",
                            norm=plt.Normalize(0, vmax))
        lc.set_array(a[mask][:-1])
        lc.set_linewidth(lw)
        return ax.add_collection(lc)

    lc_ref = None
    for a, fmask in zip(imf_a, imf_f_masked):
        lc_here = colored_ridge(ax_r, time, fmask, a)
        if lc_here is not None:
            lc_ref = lc_here

    ax_r.plot(time, f_AM_true, ls="--", color=CYAN, lw=0.9, alpha=0.8)
    ax_r.plot(time, f_FM_true, ls="--", color=CYAN, lw=0.9, alpha=0.8,
              label=r"ground truth  $f_i(t)$")
    ax_r.set_xlim(time[0], time[-1]); ax_r.set_ylim(0, fmax)
    ax_r.set_xlabel("Time  t  [s]"); ax_r.set_yticklabels([])
    ax_r.set_title("(b)  HHT spectrum  —  EMD + Hilbert on each IMF",
                   loc="left", pad=6, color=NAVY_DARK)
    ax_r.legend(loc="upper right", frameon=False, fontsize=8.5,
                labelcolor=CYAN)
    ax_r.text(0.01, -0.18,
              "ridges concentrated to a single line — adaptive, no kernel",
              transform=ax_r.transAxes,
              color=GREY, fontsize=9, ha="left")

    cax = fig.add_axes([0.985, 0.13, 0.010, 0.76])
    if lc_ref is not None:
        cb = fig.colorbar(lc_ref, cax=cax)
        cb.set_label("amplitude / normalized power", fontsize=9)
        cb.ax.tick_params(labelsize=8, colors="#444")
        cb.outline.set_linewidth(0.4)

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

    # signal
    t, x = am_fm_signal(fs, T, fc1=fc1, fa=fa, fc2=fc2, fm=fm, beta=beta)

    # ground truth
    f_AM_true = fc1 * np.ones_like(t)
    f_FM_true = fc2 + beta * fm * np.cos(2 * np.pi * fm * t)

    # SPWVD
    spwvd_f, _, spwvd_W = smoothed_pseudo_wvd(
        x, fs, sigma_freq_bins=2.5, sigma_time_samples=12.0,
    )

    # EMD → IMFs
    imf1 = sift_imf(t, x)
    imf2 = sift_imf(t, x - imf1)

    # HHT params on each IMF
    a1, f1 = hht_params(t, imf1)
    a2, f2 = hht_params(t, imf2)

    f1_v = mask_low_amplitude(a1, f1, threshold=0.15,
                              edge_samples=int(0.04 * fs))
    f2_v = mask_low_amplitude(a2, f2, threshold=0.15,
                              edge_samples=int(0.04 * fs))

    # plot
    fig = plot_spwvd_vs_hht(
        t, spwvd_f, spwvd_W,
        imf_a=[a2, a1],
        imf_f_masked=[f2_v, f1_v],
        f_AM_true=f_AM_true, f_FM_true=f_FM_true,
        fmax=260.0,
        savepath="spwvd_vs_hht.png",
    )

    # diagnostics: ridge "thickness" comparison
    # For SPWVD, the AM ridge thickness is roughly 2*sigma_f_eff
    # For HHT, the ridge spans a single line.
    print("Figure saved to: spwvd_vs_hht.png")
    print()
    print("Ridge thickness comparison (a.u.):")
    sigma_freq_bins = 2.5
    bin_width = fs / (2 * len(t))
    sigma_f_hz = sigma_freq_bins * bin_width  # in Hz
    print(f"  SPWVD AM-ridge half-width ≈ {sigma_f_hz:.2f} Hz "
          f"(set by σ_freq_bins · Δf)")
    f2_clean = f2_v[~np.isnan(f2_v)]
    hht_am_std = np.std(f2_clean) if len(f2_clean) > 1 else float("nan")
    print(f"  HHT  AM-ridge std-dev    ≈ {hht_am_std:.2f} Hz "
          f"(IF estimate noise)")
    plt.show()
