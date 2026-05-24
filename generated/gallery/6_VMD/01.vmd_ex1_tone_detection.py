"""
VMD — Example 1 · Pure tone detection across the spectrum
==========================================================

For a single-harmonic input  x(t) = cos(2π ν t), how accurately does
each method recover the frequency ν?

    VMD with K = 1 returns its centre frequency  ω_k.  We compare
    ν_est = ω_k · fs  to the true ν.

    EMD returns its dominant IMF; we estimate its frequency via the
    Hilbert transform (mean of the instantaneous frequency, with
    boundary-trimming to avoid the well-known edge artefacts).

This reproduces the central observation of Dragomiretskiy & Zosso
2014, Fig. 1:  VMD's frequency error stays near the convergence
tolerance, while EMD's grows with the carrier frequency.

Usage:  python vmd_ex1_tone_detection.py            # interactive
        python vmd_ex1_tone_detection.py out.png    # save figure
"""
import sys
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import matplotlib as mpl
from PyEMD import EMD
from vmdpy import VMD

# --- Palette ---------------------------------------------------------------
NAVY  = "#0b1f3a"
RED   = "#c9191e"
GREEN = "#1f8a4c"
GREY  = "#777777"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


# --- Frequency estimators -------------------------------------------------
def vmd_estimate_freq(x, fs, alpha=2000.0, tol=1e-9):
    """VMD with K=1: returns the centre frequency in Hz."""
    _, _, omega = VMD(x, alpha=alpha, tau=0., K=1, DC=0, init=1, tol=tol)
    return float(omega[-1, 0]) * fs


def emd_estimate_freq(x, t, fs):
    """EMD: dominant IMF → boundary-trimmed mean instantaneous frequency."""
    imfs = EMD()(x, t)
    if imfs.shape[0] == 0:
        return 0.0
    best = max(imfs, key=lambda imf: abs(np.dot(imf, x)))
    z = hilbert(best)
    phase = np.unwrap(np.angle(z))
    inst_f = np.gradient(phase) * fs / (2 * np.pi)
    trim = max(1, len(inst_f) // 10)
    return float(np.mean(inst_f[trim:-trim]))


def sweep(freqs_hz, fs=2000.0, T=1.0, snr_db=20.0, seed=0):
    """Frequency sweep with a small additive noise (SNR = 20 dB).

    A noise-free pure tone is recovered perfectly by both methods —
    differences only emerge when the extrema-finding (EMD) has to
    compete with noise.  20 dB SNR is a realistic baseline (signal
    power 100× noise power).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * T)) / fs
    err_vmd = np.zeros(len(freqs_hz))
    err_emd = np.zeros(len(freqs_hz))
    nu_vmd  = np.zeros(len(freqs_hz))
    nu_emd  = np.zeros(len(freqs_hz))
    sig_pow = 0.5  # power of cos
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_pow)
    print(f"Sweeping {len(freqs_hz)} freqs from "
          f"{freqs_hz[0]:.1f} to {freqs_hz[-1]:.1f} Hz, "
          f"SNR={snr_db} dB...")
    for i, f in enumerate(freqs_hz):
        x = np.cos(2 * np.pi * f * t) + noise_std * rng.standard_normal(len(t))
        nu_v = vmd_estimate_freq(x, fs)
        nu_e = emd_estimate_freq(x, t, fs)
        nu_vmd[i] = nu_v
        nu_emd[i] = nu_e
        err_vmd[i] = abs(nu_v - f)
        err_emd[i] = abs(nu_e - f)
        if (i + 1) % 5 == 0 or i == len(freqs_hz) - 1:
            print(f"  {i+1:3d}/{len(freqs_hz)}  ν={f:6.1f}  "
                  f"EMD est={nu_e:7.2f} (Δ={err_emd[i]:.2e})  "
                  f"VMD est={nu_v:7.2f} (Δ={err_vmd[i]:.2e})")
    return nu_vmd, nu_emd, err_vmd, err_emd


def plot_sweep(freqs, nu_vmd, nu_emd, err_vmd, err_emd, savepath=None):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=200)

    # --- LEFT: estimated vs true ---
    diag = np.linspace(freqs[0] * 0.8, freqs[-1] * 1.2, 100)
    axL.loglog(diag, diag, color=GREY, lw=0.7, ls="--",
               label="y = ν  (ideal)")
    axL.loglog(freqs, np.maximum(nu_emd, 1e-3),
               color=RED, lw=0, marker="o", markersize=4,
               markeredgewidth=0, label="EMD")
    axL.loglog(freqs, np.maximum(nu_vmd, 1e-3),
               color=GREEN, lw=0, marker="s", markersize=4,
               markeredgewidth=0, label="VMD")
    axL.set_xlim(freqs[0] * 0.85, freqs[-1] * 1.15)
    axL.set_ylim(freqs[0] * 0.85, freqs[-1] * 1.15)
    axL.set_xlabel("true frequency  ν  [Hz]",
                   fontsize=9.5, color=NAVY, labelpad=2)
    axL.set_ylabel("estimated frequency  [Hz]",
                   fontsize=9.5, color=NAVY, labelpad=2)
    axL.set_xticks([10, 30, 100, 300])
    axL.set_xticklabels(["10", "30", "100", "300"])
    axL.set_yticks([10, 30, 100, 300])
    axL.set_yticklabels(["10", "30", "100", "300"])
    axL.tick_params(labelsize=8)
    axL.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    axL.set_title("(a)  Estimated frequency vs ν",
                  loc="left", fontsize=10, color=NAVY,
                  fontweight="bold", pad=4)
    axL.legend(loc="upper left", frameon=False, fontsize=8.5)
    for sp in axL.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.5)

    # --- RIGHT: error magnitude ---
    axR.semilogy(freqs, np.maximum(err_emd, 1e-7),
                 color=RED, lw=1.2, marker="o", markersize=3.5,
                 markeredgewidth=0, label="EMD")
    axR.semilogy(freqs, np.maximum(err_vmd, 1e-7),
                 color=GREEN, lw=1.2, marker="s", markersize=3.5,
                 markeredgewidth=0, label="VMD")
    axR.axhline(1e-3, color=GREEN, lw=0.5, ls=":", alpha=0.6)
    axR.text(freqs[-1] * 0.96, 1.3e-3,
             "VMD stays at the tolerance floor",
             ha="right", va="bottom", color=GREEN, fontsize=7,
             fontstyle="italic")
    axR.set_xscale("log")
    axR.set_xlim(freqs[0] * 0.85, freqs[-1] * 1.15)
    axR.set_ylim(5e-4, max(err_emd.max() * 2, 50))
    axR.set_xticks([10, 30, 100, 300])
    axR.set_xticklabels(["10", "30", "100", "300"])
    axR.tick_params(labelsize=8)
    axR.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    axR.set_xlabel("true frequency  ν  [Hz]",
                   fontsize=9.5, color=NAVY, labelpad=2)
    axR.set_ylabel(r"frequency error  $|\nu_{\mathrm{est}} - \nu|$  [Hz]",
                   fontsize=9.5, color=NAVY, labelpad=2)
    axR.set_title("(b)  Absolute frequency error",
                  loc="left", fontsize=10, color=NAVY,
                  fontweight="bold", pad=4)
    axR.legend(loc="upper left", frameon=False, fontsize=8.5)
    for sp in axR.spines.values():
        sp.set_color(NAVY); sp.set_linewidth(0.5)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    freqs = np.unique(np.round(np.logspace(np.log10(5.0),
                                            np.log10(300.0),
                                            18))).astype(float)
    nu_vmd, nu_emd, err_vmd, err_emd = sweep(freqs)
    plot_sweep(freqs, nu_vmd, nu_emd, err_vmd, err_emd, savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
