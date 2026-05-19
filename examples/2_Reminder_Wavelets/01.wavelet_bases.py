"""
=======================================================================
 Wavelet Theory — Script 1: Mother Wavelets & Basis Functions
=======================================================================
 Topics covered:
   • Fourier sinusoids vs Morlet wavelets (time localization)
   • Scaling and translation of a mother wavelet
   • Admissibility condition (zero mean)
   • Continuous Wavelet Transform (CWT) via convolution
   • Scalogram visualisation
 Run:  python 01_wavelet_bases.py
 Requires: numpy, matplotlib, scipy

Author: Philippe Ciuciu
Date: 04/07/2026
Target: UnseenLabs
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 1. Mother wavelet: Morlet ──────────────────────────────────────────
def morlet(t, sigma=1.0, c=2.5, analytic=False):
    """
    Morlet mother wavelet: modulated Gaussian.
      ψ(t) = π^{-1/4} · e^{iω₀t} · e^{-t²/2}      with  ω₀ = 2c
    Returns the real part by default (for visualisation).
    Set analytic=True for the complex analytic form — required for a clean
    CWT envelope (|W| is otherwise modulated by the wavelet's oscillations).
      Admissibility:  ∫ψ(t) dt ≈ 0   (zero mean)
    """
    psi = np.pi**(-0.25) * np.exp(1j * 2.0 * c * t) * np.exp(-t**2 / (2.0 * sigma**2))
    return psi if analytic else psi.real


def morlet_scaled(t, a, b, sigma=1.0, analytic=False):
    """
    Scaled & translated Morlet wavelet:
      ψ_{a,b}(t) = 1/√a · ψ((t-b)/a)
    a > 1  →  stretched (lower frequency, wider)
    a < 1  →  compressed (higher frequency, narrower)
    """
    return (1.0 / np.sqrt(abs(a))) * morlet((t - b) / a, sigma=sigma, analytic=analytic)


# ── 2. Visualise scaling & translation ────────────────────────────────
t = np.linspace(-5, 15, 2000)

fig, axes = plt.subplots(3, 3, figsize=(14, 8))
fig.suptitle("Morlet Wavelet — Scaling (rows) & Translation (columns)",
             fontsize=13, fontweight='bold')

scales       = [0.5, 1.0, 2.5]   # a  (scale: low=compressed, high=stretched)
translations = [0.0, 4.0, 8.0]   # b  (center position)
colors       = ['steelblue', 'darkorange', 'seagreen']

for i, a in enumerate(scales):
    for j, b in enumerate(translations):
        ax   = axes[i][j]
        psi  = morlet_scaled(t, a, b)
        ax.plot(t, psi, color=colors[i], lw=1.5)
        ax.fill_between(t, psi, alpha=0.18, color=colors[i])
        ax.axhline(0, color='grey', lw=0.6, ls='--')
        ax.set_title(f'a={a}, b={b}', fontsize=9)
        ax.set_xlim(-1, 14)
        if j == 0: ax.set_ylabel(f'Scale {a}', fontsize=9)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('plot1a_scaling_translation.png', dpi=120)
plt.show()
print("Saved: plot1a_scaling_translation.png")


# ── 3. Fourier basis vs Wavelet basis (key difference) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
t2 = np.linspace(0, 4, 1000)

# Fourier: global, infinite support
ax = axes[0]
for f, col, lbl in [(2, 'steelblue', 'f=2 Hz'), (5, 'darkorange', 'f=5 Hz'),
                    (10, 'seagreen',  'f=10 Hz')]:
    ax.plot(t2, np.sin(2 * np.pi * f * t2), color=col, lw=1.5, label=lbl)
ax.set_title("Fourier Bases — Global Support\n(no time localization)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")

# Wavelet: local support
ax = axes[1]
t3 = np.linspace(-8, 8, 2000)
for a, b, col, lbl in [(1.0, -4, 'steelblue', 'a=1, b=−4'),
                        (0.5, 0,  'darkorange', 'a=0.5, b=0'),
                        (2.0, 4,  'seagreen',   'a=2, b=4')]:
    psi = morlet_scaled(t3, a, b)
    ax.plot(t3, psi, color=col, lw=1.5, label=lbl)
    ax.fill_between(t3, psi, alpha=0.12, color=col)
ax.set_title("Wavelet Bases — Compact Support\n(localized in time & frequency)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")
ax.axhline(0, color='grey', lw=0.5)

plt.tight_layout()
plt.savefig('plot1b_fourier_vs_wavelet.png', dpi=120)
plt.show()
print("Saved: plot1b_fourier_vs_wavelet.png")

