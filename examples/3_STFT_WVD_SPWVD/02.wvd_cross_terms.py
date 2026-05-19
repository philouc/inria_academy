"""
=======================================================================
 Multi-component WVD cross-terms — Two parallel chirps: Second script
=======================================================================

Demonstrates the central limitation of the Wigner-Ville distribution:
when the signal contains several components, every pair produces an
oscillating cross-term that sits halfway between the two auto-terms
in the time-frequency plane. The smoothed pseudo Wigner-Ville
distribution (SPWVD) suppresses these cross-terms at the cost of
spatial concentration.

Three panels produced:
  (a) STFT          -> two clean (blurred) auto-terms, NO cross-term
  (b) WVD           -> two sharp auto-terms PLUS striped cross-term
  (c) SPWVD         -> cross-term suppressed, auto-terms wider

For two complex exponentials at f1 and f2, the WVD cross-term sits at
the mean frequency (f1+f2)/2 and oscillates in time at the difference
frequency (f1-f2). For two parallel chirps with rates k1=k2=k, the
cross-term mid-line is also a parallel chirp, with a constant fringe
spacing equal to that difference frequency.

=======================================================================
 Run:  python 02.wvd_cross_terms.py
 Dependencies: numpy, scipy, matplotlib.

 Author: Philippe Ciuciu
 Date: 04/25/2026
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
def linear_chirp(fs: float, T: float,
                 f0: float, k: float,
                 phi0: float = 0.0, A: float = 1.0):
    """Real linear chirp x(t) = A*cos(2*pi*(f0*t + 0.5*k*t**2) + phi0)."""
    n = int(round(fs * T))
    t = np.arange(n) / fs
    x = A * np.cos(2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phi0)
    return t, x


def two_parallel_chirps(fs: float, T: float,
                        f0_1: float, f0_2: float,
                        k: float):
    """
    Two parallel chirps (same rate k, different start frequencies).
    Returns (t, x, f_theo_1, f_theo_2, f_cross).
    """
    t, x1 = linear_chirp(fs, T, f0_1, k)
    _, x2 = linear_chirp(fs, T, f0_2, k)
    x = x1 + x2
    f1 = f0_1 + k * t
    f2 = f0_2 + k * t
    return t, x, f1, f2, 0.5 * (f1 + f2)


# ---------------------------------------------------------------------------
# Time-frequency representations
# ---------------------------------------------------------------------------
def stft_power(x, fs, nperseg, overlap=0.9, nfft=None):
    """STFT magnitude squared with a Hann window."""
    if nfft is None:
        nfft = max(nperseg, 512)
    f, t, Z = stft(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=int(round(nperseg * overlap)), nfft=nfft,
        boundary="zeros", padded=True,
    )
    return f, t, np.abs(Z) ** 2


def wigner_ville(x, fs):
    """
    Discrete Wigner-Ville distribution on the analytic signal.

    The kernel z[n+m]*conj(z[n-m]) advances in phase at twice the
    true signal rate, so FFT bin k maps to true frequency k*fs/(2N).
    The full N bins span [0, fs/2].
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


def smoothed_pseudo_wvd(x, fs, sigma_freq_bins=2.5, sigma_time_samples=14.0):
    """
    Smoothed pseudo Wigner-Ville distribution via 2-D Gaussian smoothing
    of the WVD.

    The two smoothing widths play distinct roles. The time-smoothing
    averages out cross-term oscillations along the time axis, and must
    span at least one period of the slowest cross-term oscillation
    (≈ fs / |f_i - f_j| samples for components at f_i, f_j). The
    frequency-smoothing kills any remaining ridges that oscillate along
    the frequency direction.

    Wider kernels suppress cross-terms more aggressively but enlarge
    the auto-terms. As a rule of thumb, pick
        sigma_time ≈ (fs / Δf_min) / 2
    where Δf_min is the smallest expected inter-component spacing — i.e.
    roughly half the period of the slowest cross-term oscillation.
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


def plot_cross_terms(t_sig, f_theo_1, f_theo_2, f_cross,
                     stft_panel, wvd_panel, spwvd_panel,
                     fmax: float = 260.0,
                     savepath: str | None = None):
    """
    Side-by-side panel: STFT | WVD | SPWVD. Theoretical IF lines for
    both chirps overlaid in cyan; cross-term midline overlaid on the
    WVD panel in yellow.
    """
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
         "two blurred auto-terms, no cross-term",       "db"),
        (wvd_panel,   "(b) Wigner-Ville distribution",
         "auto-terms + striped cross-term at midline",  "wvd"),
        (spwvd_panel, "(c) Smoothed pseudo-WVD",
         "cross-term suppressed by 2-D smoothing",      "spwvd"),
    ]

    last_im = None
    for i, ((freq, time, M), title, subtitle, kind) in enumerate(panels):
        ax = fig.add_axes([left + i * (width + gap), bottom, width, height])

        if kind == "db":
            Z, vmin, vmax = _to_db(M), -50, 0
        elif kind == "wvd":
            Z = M / np.abs(M).max()
            vmin, vmax = -0.10, 0.20
        else:
            Z = M / max(M.max(), 1e-20)
            vmin, vmax = 0.0, 0.20

        last_im = ax.imshow(Z, aspect="auto", origin="lower",
                            extent=[time[0], time[-1], freq[0], freq[-1]],
                            cmap="magma", vmin=vmin, vmax=vmax,
                            interpolation="bilinear")

        ax.plot(t_sig, f_theo_1, ls="--", color=IF_CYAN, lw=1.0, alpha=0.9)
        ax.plot(t_sig, f_theo_2, ls="--", color=IF_CYAN, lw=1.0, alpha=0.9,
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
    fs    = 1000        # sampling rate, Hz
    T     = 1.0         # duration, s
    f0_1  = 20.0        # chirp 1 start frequency, Hz
    f0_2  = 140.0       # chirp 2 start frequency, Hz
    k     = 80.0        # common chirp rate, Hz/s
    delta_f = abs(f0_2 - f0_1)            # constant inter-component gap

    # SPWVD smoothing: enough time-smoothing to cover ~2 cross-term periods
    # The cross-term oscillates at delta_f Hz → period 1/delta_f s.
    # In samples: fs/delta_f. Take sigma ≈ that / 4.
    cross_period_samples = fs / delta_f
    sigma_time = cross_period_samples * 0.5
    sigma_freq = 2.5

    # --- generate two parallel chirps -----------------------------------
    t, x, f1, f2, f_cross = two_parallel_chirps(fs, T, f0_1, f0_2, k)

    # --- TF representations ---------------------------------------------
    stft_panel  = stft_power(x, fs, nperseg=96)
    wvd_panel   = wigner_ville(x, fs)
    spwvd_panel = smoothed_pseudo_wvd(x, fs,
                                      sigma_freq_bins=sigma_freq,
                                      sigma_time_samples=sigma_time)

    # --- plot ------------------------------------------------------------
    fig = plot_cross_terms(t, f1, f2, f_cross,
                           stft_panel, wvd_panel, spwvd_panel,
                           savepath="wvd_cross_terms.png")
    print("Figure saved to: wvd_cross_terms.png")
    print(f"Cross-term spacing Δf = {delta_f:.1f} Hz "
          f"(period {cross_period_samples:.1f} samples)")
    plt.show()
