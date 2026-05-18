"""
=======================================================================
 STFT vs Wigner-Ville Distribution: Linear chirp example: First script
=======================================================================

Reproduces the time-frequency comparison panel:
  (a) STFT with a short Hann window  -> fine time, blurred frequency
  (b) STFT with a long  Hann window  -> fine frequency, blurred time
  (c) Wigner-Ville distribution      -> razor-thin ridge on f_i(t)

The theoretical instantaneous frequency f_i(t) = f0 + k*t is overlaid
on every panel as a dashed cyan line.

=======================================================================
 Run:  python 01.stft_vs_wvd_chirp.py
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


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def linear_chirp(fs: float, T: float,
                 f0: float, k: float,
                 phi0: float = 0.0, A: float = 1.0):
    """
    Generate a real linear chirp x(t) = A*cos(2*pi*(f0*t + 0.5*k*t**2) + phi0).

    Instantaneous frequency: f_i(t) = f0 + k*t.

    Parameters
    ----------
    fs   : sampling rate [Hz]
    T    : signal duration [s]
    f0   : start frequency [Hz]
    k    : chirp rate [Hz/s]  (k>0: up-chirp, k<0: down-chirp)
    phi0 : initial phase [rad]
    A    : amplitude

    Returns
    -------
    t : 1D array, time stamps [s]
    x : 1D array, signal samples
    """
    n_samples = int(round(fs * T))
    t = np.arange(n_samples) / fs
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2) + phi0
    x = A * np.cos(phase)
    return t, x


# ---------------------------------------------------------------------------
# STFT
# ---------------------------------------------------------------------------
def stft_power(x: np.ndarray, fs: float,
               nperseg: int, overlap: float = 0.9,
               nfft: int | None = None):
    """
    Short-Time Fourier Transform magnitude squared (Hann window).

    Parameters
    ----------
    nperseg : window length in samples
    overlap : fractional overlap between windows (0 to 1)
    nfft    : zero-padded FFT length (defaults to max(nperseg, 512))

    Returns
    -------
    f : frequency axis [Hz]
    t : time axis [s]
    S : 2D array |STFT|^2, shape (len(f), len(t))
    """
    noverlap = int(round(nperseg * overlap))
    if nfft is None:
        nfft = max(nperseg, 512)
    f, t, Z = stft(
        x, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        boundary="zeros", padded=True,
    )
    return f, t, np.abs(Z) ** 2


# ---------------------------------------------------------------------------
# Wigner-Ville Distribution
# ---------------------------------------------------------------------------
def wigner_ville(x: np.ndarray, fs: float):
    """
    Discrete Wigner-Ville distribution of a real signal, using the
    analytic signal to suppress cross-terms between positive and
    negative frequency components.

      W[n, k] = real( FFT_m { z[n+m] * conj(z[n-m]) } )

    Notes on the frequency axis
    ---------------------------
    The discrete kernel z[n+m]*conj(z[n-m]) advances in phase at
    twice the rate of the true signal frequency, because the
    "tau/2" of the continuous definition is replaced by an integer
    step m (so tau = 2m). Consequently, FFT bin k corresponds to
    *true* frequency k * fs / (2N), not k * fs / N, and the N bins
    of the FFT span [0, fs/2] -- not [0, fs] as a naive reading
    would suggest.

    Sanity-check this against a pure tone before trusting it on
    your own data.

    Parameters
    ----------
    x  : real-valued 1D array
    fs : sampling rate [Hz]

    Returns
    -------
    freq : frequency axis [Hz], length N, spanning [0, fs/2)
    time : time axis [s], length N
    W    : 2D array, real-valued WVD, shape (N, N) = (freq, time)
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


def _sanity_check_wvd(verbose: bool = False) -> None:
    """Verify the frequency calibration on a pure tone."""
    fs, T, f0 = 1000.0, 0.5, 75.0
    t, x = linear_chirp(fs, T, f0=f0, k=0.0)
    freq, _, W = wigner_ville(x, fs)
    k_peak = np.argmax(W[:, len(x) // 2])
    if verbose:
        print(f"[sanity] Pure tone {f0} Hz -> WVD peak at "
              f"bin {k_peak}, freq = {freq[k_peak]:.2f} Hz")
    assert abs(freq[k_peak] - f0) < 1.0, "WVD frequency axis miscalibrated"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _to_db(S: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    """Power spectrogram -> dB, normalised so 0 dB is the peak."""
    Snorm = S / np.maximum(S.max(), 1e-20)
    return 10 * np.log10(np.maximum(Snorm, 10 ** (floor_db / 10)))


def plot_stft_vs_wvd(t_signal, f_theo,
                     stft_short, stft_long, wvd,
                     fmax: float = 260.0,
                     savepath: str | None = None):
    """
    Side-by-side comparison panel.

    Each `stft_short`, `stft_long`, `wvd` is a (freq, time, matrix) tuple
    as returned by stft_power / wigner_ville.
    """
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#888",
        "axes.linewidth": 0.6,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.color": "#444", "ytick.color": "#444",
    })

    fig = plt.figure(figsize=(13.5, 5.45), dpi=144)
    left, bottom, width, height, gap = 0.045, 0.13, 0.295, 0.74, 0.025
    if_color = "#7FE0FF"  # cyan, visible on any colormap top

    panels = [
        (stft_short, "(a) STFT  ·  short window",
         "good time res / blurred frequency", "db"),
        (stft_long,  "(b) STFT  ·  long window",
         "good frequency res / blurred time", "db"),
        (wvd,        "(c) Wigner-Ville distribution",
         "optimal concentration on linear FM", "wvd"),
    ]

    last_im = None
    for i, ((freq, time, M), title, subtitle, kind) in enumerate(panels):
        ax = fig.add_axes([left + i * (width + gap), bottom, width, height])

        if kind == "db":
            Z, vmin, vmax = _to_db(M), -50, 0
        else:                              # wvd: saturate the thin ridge
            Z = M / np.abs(M).max()
            vmin, vmax = 0.0, 0.25

        last_im = ax.imshow(
            Z, aspect="auto", origin="lower",
            extent=[time[0], time[-1], freq[0], freq[-1]],
            cmap="magma", vmin=vmin, vmax=vmax, interpolation="bilinear",
        )
        ax.plot(t_signal, f_theo, ls="--", color=if_color, lw=1.1, alpha=0.95,
                label=r"theoretical $f_i(t)$")
        ax.set_xlim(t_signal[0], t_signal[-1])
        ax.set_ylim(0, fmax)
        ax.set_title(title, loc="left", pad=6, color="#0b1f3a")
        ax.text(0.01, -0.18, subtitle, transform=ax.transAxes,
                color="#777", fontsize=9, ha="left")
        ax.set_xlabel("Time  t  [s]")
        if i == 0:
            ax.set_ylabel("Frequency  [Hz]")
        else:
            ax.set_yticklabels([])
        if i == 2:
            ax.legend(loc="upper left", frameon=False,
                      fontsize=8.5, labelcolor=if_color)

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
    # --- experimental parameters -----------------------------------------
    fs    = 1000      # sampling rate, Hz
    T     = 1.0       # signal duration, s
    f0    = 20.0      # start frequency, Hz
    k     = 200.0     # chirp rate, Hz/s  (20 -> 220 Hz over 1 s)
    n_short = 32      # STFT short window, samples
    n_long  = 256     # STFT long  window, samples

    # --- generate the chirp ---------------------------------------------
    t, x = linear_chirp(fs, T, f0, k)
    f_theo = f0 + k * t                                  # ground-truth IF

    # --- TF representations ---------------------------------------------
    _sanity_check_wvd(verbose=True)
    stft_s = stft_power(x, fs, nperseg=n_short)
    stft_l = stft_power(x, fs, nperseg=n_long)
    wvd    = wigner_ville(x, fs)

    # --- plot ------------------------------------------------------------
    fig = plot_stft_vs_wvd(t, f_theo, stft_s, stft_l, wvd,
                           savepath="stft_vs_wvd_chirp.png")
    print("Figure saved to: stft_vs_wvd_chirp.png")
    plt.show()
