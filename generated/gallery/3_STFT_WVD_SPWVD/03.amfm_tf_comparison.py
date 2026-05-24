"""
=======================================================================
 AM-FM benchmark — STFT vs WVD vs SPWVD: Third script
=======================================================================

Signal: x(t) = s_AM(t) + s_FM(t)
  s_AM(t) = (1 + cos(2π fa t)) * cos(2π fc1 t)        (AM at fc1=80 Hz, fa=4 Hz)
  s_FM(t) = cos(2π fc2 t + β sin(2π fm t))            (FM at fc2=180 Hz, fm=4 Hz, β=12.5)

Three panels:
  (a) STFT          -> two ridges, no cross-terms, slightly blurred
  (b) WVD           -> sharpest ridges + wavy cross-term in chevrons
  (c) SPWVD         -> cross-term suppressed, auto-terms enlarged

The cross-term sits at the midline (fc1 + f2(t)) / 2 and inherits the
modulation of the FM component, so its fringe density changes with time.

=======================================================================
 Run:  python 03.amfm_tf_comparison.py
 Dependencies: numpy, scipy, matplotlib.

 Author: Philippe Ciuciu
 Date: 04/27/2026
 Target: UnseenLabs
=======================================================================
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import stft, hilbert
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def am_component(t, fc, fa, A0=1.0, A1=1.0, phi=0.0):
    """
    Sinusoidal AM at carrier fc, modulator fa.
        s(t) = (A0 + A1*cos(2π fa t)) * cos(2π fc t + phi)
    Default (A0=A1=1) gives an envelope swinging between 0 and 2.
    """
    A = A0 + A1 * np.cos(2 * np.pi * fa * t)
    return A * np.cos(2 * np.pi * fc * t + phi)


def fm_component(t, fc, fm, beta, phi=0.0):
    """
    Sinusoidal FM at carrier fc, modulator fm, modulation index beta.
        s(t)   = cos(2π fc t + β sin(2π fm t) + phi)
        f_i(t) = fc + β * fm * cos(2π fm t)
    Peak frequency deviation is beta * fm.
    """
    return np.cos(2 * np.pi * fc * t + beta * np.sin(2 * np.pi * fm * t) + phi)


def amfm_signal(fs, T,
                fc1=80.0,  fa=4.0,
                fc2=180.0, fm=4.0, beta=12.5):
    """
    Build the two-component AM + FM benchmark.

    Returns
    -------
    t        : time vector
    x        : x(t) = s_AM + s_FM
    f_AM     : instantaneous frequency of the AM component (constant, fc1)
    f_FM     : instantaneous frequency of the FM component (cos-modulated)
    f_cross  : WVD cross-term midline (fc1 + f_FM) / 2
    """
    N = int(round(fs * T))
    t = np.arange(N) / fs
    s_am = am_component(t, fc=fc1, fa=fa)
    s_fm = fm_component(t, fc=fc2, fm=fm, beta=beta)
    x = s_am + s_fm
    f_AM = np.full_like(t, fc1)
    f_FM = fc2 + beta * fm * np.cos(2 * np.pi * fm * t)
    f_cross = 0.5 * (f_AM + f_FM)
    return t, x, f_AM, f_FM, f_cross


# ---------------------------------------------------------------------------
# Time-frequency representations
# ---------------------------------------------------------------------------
def stft_power(x, fs, nperseg, overlap=0.9, nfft=None):
    """STFT magnitude squared (Hann window)."""
    if nfft is None:
        nfft = max(nperseg, 1024)
    f, t, Z = stft(x, fs=fs, window="hann",
                   nperseg=nperseg, noverlap=int(round(nperseg * overlap)),
                   nfft=nfft, boundary="zeros", padded=True)
    return f, t, np.abs(Z) ** 2


def wigner_ville(x, fs):
    """
    Discrete WVD on the analytic signal.

    FFT bin k maps to true frequency k * fs / (2N) — see the docstring
    of the calibration-tested version in stft_vs_wvd_chirp.py.
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


def smoothed_pseudo_wvd(x, fs,
                        sigma_freq_bins=2.5,
                        sigma_time_samples=12.0):
    """
    2-D Gaussian smoothing of the WVD.

    For the AM-FM benchmark, the smallest meaningful component spacing
    is the carrier gap fc2 - fc1 ≈ 100 Hz, but the cross-term fringe
    period (controlled by the *instantaneous* spacing) goes down to
    1/(fc2 - fc1 - β*fm) ≈ 1/50 s ≈ 20 samples at fs=1000 — set
    sigma_time around half that.
    """
    freq, time, W = wigner_ville(x, fs)
    W_smooth = gaussian_filter(W, sigma=(sigma_freq_bins, sigma_time_samples))
    return freq, time, W_smooth


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _to_db(S, floor=-50):
    Sn = S / max(S.max(), 1e-20)
    return 10 * np.log10(np.maximum(Sn, 10 ** (floor / 10)))


def plot_amfm_comparison(t_sig, f_AM, f_FM, f_cross,
                         stft_panel, wvd_panel, spwvd_panel,
                         fmax: float = 260.0,
                         savepath: str | None = None):
    """Side-by-side comparison panel: STFT | WVD | SPWVD."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#888", "axes.linewidth": 0.6,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.color": "#444", "ytick.color": "#444",
    })
    NAVY_DARK = "#0b1f3a"
    GREY = "#777777"
    IF_CYAN = "#7FE0FF"
    CROSS_YELLOW = "#FFD43B"

    fig = plt.figure(figsize=(13.5, 5.45), dpi=144)
    left, bottom, width, height, gap = 0.045, 0.13, 0.295, 0.74, 0.025

    panels = [
        (stft_panel,  "(a) STFT  ·  Hann window",
         "two ridges, no cross-terms, blurred",      "db"),
        (wvd_panel,   "(b) Wigner-Ville distribution",
         "sharpest ridges + wavy cross-term",        "wvd"),
        (spwvd_panel, "(c) Smoothed pseudo-WVD",
         "cross-term suppressed, ridges enlarged",   "spwvd"),
    ]

    last_im = None
    for i, ((freq, time, M), title, subtitle, kind) in enumerate(panels):
        ax = fig.add_axes([left + i * (width + gap), bottom, width, height])

        if kind == "db":
            Z, vmin, vmax = _to_db(M), -50, 0
        elif kind == "wvd":
            Z = M / np.abs(M).max()
            vmin, vmax = -0.05, 0.10
        else:
            Z = M / max(M.max(), 1e-20)
            vmin, vmax = 0.0, 0.20

        last_im = ax.imshow(Z, aspect="auto", origin="lower",
                            extent=[time[0], time[-1], freq[0], freq[-1]],
                            cmap="magma", vmin=vmin, vmax=vmax,
                            interpolation="bilinear")

        # theoretical IFs
        ax.plot(t_sig, f_AM, ls="--", color=IF_CYAN, lw=1.0, alpha=0.9)
        ax.plot(t_sig, f_FM, ls="--", color=IF_CYAN, lw=1.0, alpha=0.9,
                label=r"theoretical $f_i(t)$")
        if kind == "wvd":
            ax.plot(t_sig, f_cross, ls=":", color=CROSS_YELLOW, lw=1.0,
                    alpha=0.9, label="cross-term midline")

        ax.set_xlim(t_sig[0], t_sig[-1])
        ax.set_ylim(0, fmax)
        ax.set_title(title, loc="left", pad=6, color=NAVY_DARK)
        ax.text(0.01, -0.18, subtitle, transform=ax.transAxes,
                color=GREY, fontsize=9, ha="left")
        ax.set_xlabel("Time  t  [s]")
        if i == 0:
            ax.set_ylabel("Frequency  [Hz]")
        else:
            ax.set_yticklabels([])

        if i in (1, 2):
            ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                      labelcolor=IF_CYAN if i == 2 else None)

    cax = fig.add_axes([0.97, bottom, 0.012, height])
    cb = fig.colorbar(last_im, cax=cax)
    cb.ax.tick_params(labelsize=8, colors="#444")
    cb.outline.set_linewidth(0.4)

    if savepath:
        fig.savefig(savepath, dpi=144, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- experimental parameters ----------------------------------------
    fs   = 1000     # sampling rate, Hz
    T    = 1.0      # duration, s
    fc1, fa            = 80.0,  4.0          # AM carrier and modulator
    fc2, fm, beta      = 180.0, 4.0, 12.5    # FM carrier, modulator, index

    # --- build the signal -----------------------------------------------
    t, x, f_AM, f_FM, f_cross = amfm_signal(
        fs, T, fc1=fc1, fa=fa, fc2=fc2, fm=fm, beta=beta,
    )

    # --- TF representations ---------------------------------------------
    stft_panel  = stft_power(x, fs, nperseg=128)
    wvd_panel   = wigner_ville(x, fs)
    spwvd_panel = smoothed_pseudo_wvd(x, fs,
                                      sigma_freq_bins=2.5,
                                      sigma_time_samples=12.0)

    # --- plot ------------------------------------------------------------
    fig = plot_amfm_comparison(t, f_AM, f_FM, f_cross,
                               stft_panel, wvd_panel, spwvd_panel,
                               savepath="amfm_tf_comparison.png")
    print("Figure saved to: amfm_tf_comparison.png")
    print(f"FM peak frequency deviation: ±{beta * fm:.1f} Hz")
    print(f"FM range: [{fc2 - beta*fm:.1f}, {fc2 + beta*fm:.1f}] Hz")
    plt.show()
