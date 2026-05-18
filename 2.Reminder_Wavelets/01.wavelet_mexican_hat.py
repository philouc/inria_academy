"""
=======================================================================
 Wavelet Theory — Exercise solutions for Script 1
=======================================================================
 1. Mexican hat (Ricker) wavelet  +  numerical admissibility check
 2. Sine signal with a frequency jump at t=0.5  →  scalogram
 3. Effect of additive Gaussian noise on the scalogram
=======================================================================
 Run:  python 01.wavelet_mexican_hat.py

Requires: numpy, matplotlib, scipy

Author: Philippe Ciuciu
Date: 04/07/2026
Target: UnseenLabs
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert


# ── Ricker / Mexican hat mother wavelet ───────────────────────────────
def ricker(t):
    """
    Ricker (Mexican hat) wavelet — 2nd derivative of a Gaussian:
        ψ(t) = (1 - t²) · exp(-t²/2)
    Admissibility:  ∫ψ(t) dt = 0
    """
    return (1.0 - t**2) * np.exp(-t**2 / 2.0)


def ricker_scaled(t, a, b):
    """ψ_{a,b}(t) = 1/√a · ψ((t - b) / a)"""
    return (1.0 / np.sqrt(abs(a))) * ricker((t - b) / a)


# ── Generic CWT (reused for parts 2 and 3) ────────────────────────────
def cwt(signal, t_sig, scales, wavelet=ricker_scaled, analytic=True):
    """
    Continuous Wavelet Transform via convolution.

    Returns the scalogram  |W_x(a,b)|²  (energy density convention).

    The Ricker is a real wavelet, so for an oscillatory input the raw
    convolution oscillates in b and the scalogram acquires spurious
    vertical stripes (one bright "blob" per oscillation of the signal).
    With analytic=True (default) we apply a Hilbert transform to the
    convolution to recover the smooth envelope — equivalent to using a
    complex Ricker  ψ̃(t) = ψ(t) + i·H{ψ}(t), since Hilbert is LTI and
    commutes with convolution by a real signal.

    Setting analytic=False reproduces the textbook real-CWT behaviour
    (useful pedagogically to *see* the striping).
    """
    dt = t_sig[1] - t_sig[0]
    scalogram = np.zeros((len(scales), len(signal)))
    for k, a in enumerate(scales):
        n_kern = max(int(8 * a / dt), 10)
        tau    = np.linspace(-4 * a, 4 * a, n_kern)
        kernel = wavelet(tau, a, 0)
        # Riemann-sum approximation of  ∫ x(t) ψ((t-b)/a) dt / √a   needs * dt
        conv   = np.convolve(signal, kernel[::-1], mode='same') * dt
        # mode='same' returns max(len(sig), len(kernel)) — trim centrally if needed
        if len(conv) > len(signal):
            s    = (len(conv) - len(signal)) // 2
            conv = conv[s:s + len(signal)]
        if analytic:
            conv = hilbert(conv)               # envelope via analytic signal
        scalogram[k] = np.abs(conv)**2         # |W_x(a,b)|²
    return scalogram


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 1 — Ricker wavelet & admissibility                            ║
# ╚═════════════════════════════════════════════════════════════════════╝
t = np.linspace(-10, 10, 5000)
psi = ricker(t)
integral = np.trapezoid(psi, t)        # numerical ∫ψ(t)dt  (np.trapz removed in NumPy 2.x)
print(f"Part 1 — ∫ψ(t)dt = {integral:.3e}   (≈ 0 ⇒ admissible)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# (a) mother wavelet
ax = axes[0]
ax.plot(t, psi, color='steelblue', lw=1.7)
ax.fill_between(t, psi, alpha=0.18, color='steelblue')
ax.axhline(0, color='grey', lw=0.6, ls='--')
ax.set_title(f"Ricker mother wavelet — ∫ψ dt = {integral:.2e}",
             fontsize=11, fontweight='bold')
ax.set_xlabel("t"); ax.set_ylabel("ψ(t)"); ax.set_xlim(-6, 6)

# (b) scaled & translated versions
ax = axes[1]
t2 = np.linspace(-8, 12, 2000)
for a, b, col, lbl in [(1.0, -4, 'steelblue',  'a=1, b=−4'),
                       (0.5,  0, 'darkorange', 'a=0.5, b=0'),
                       (2.0,  6, 'seagreen',   'a=2, b=6')]:
    ax.plot(t2, ricker_scaled(t2, a, b), color=col, lw=1.5, label=lbl)
    ax.fill_between(t2, ricker_scaled(t2, a, b), alpha=0.12, color=col)
ax.axhline(0, color='grey', lw=0.5)
ax.set_title("Scaled & translated Ricker wavelets",
             fontsize=11, fontweight='bold')
ax.set_xlabel("t"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot_ex1_ricker.png', dpi=120)
plt.show()
print("Saved: plot_ex1_ricker.png\n")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 2 — Frequency jump at t=0.5                                   ║
# ╚═════════════════════════════════════════════════════════════════════╝
t_sig  = np.linspace(0, 1, 1024)
f1, f2 = 8.0, 30.0
sig    = np.where(t_sig < 0.5,
                  np.sin(2 * np.pi * f1 * t_sig),
                  np.sin(2 * np.pi * f2 * t_sig))

scales    = np.logspace(np.log10(0.005), np.log10(0.15), 80)
scalogram = cwt(sig, t_sig, scales)

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                         gridspec_kw={'height_ratios': [1, 2]})
axes[0].plot(t_sig, sig, color='steelblue', lw=1.0)
axes[0].axvline(0.5, color='crimson', lw=1.2, ls='--', label='jump at t=0.5')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].set_title(f"Sine signal: {f1:.0f} Hz → {f2:.0f} Hz at t=0.5",
                  fontsize=11, fontweight='bold')
axes[0].set_ylabel("Amplitude")

im = axes[1].imshow(scalogram, aspect='auto', origin='lower',
                    extent=[t_sig[0], t_sig[-1], 0, len(scales)],
                    cmap='inferno')
axes[1].axvline(0.5, color='white', lw=1.0, ls='--', alpha=0.8)
axes[1].set_yticks(np.linspace(0, len(scales), 5))
axes[1].set_yticklabels([f'{s:.3f}' for s in
                         scales[np.linspace(0, len(scales)-1, 5, dtype=int)]])
axes[1].set_title("Scalogram |CWT(a,b)|² — Ricker wavelet (analytic envelope)",
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel("Time (b)"); axes[1].set_ylabel("Scale (a)")
plt.colorbar(im, ax=axes[1], label=r'$|W_x(a,b)|^2$')
plt.tight_layout()
plt.savefig('plot_ex2_frequency_jump.png', dpi=120)
plt.show()
print("Saved: plot_ex2_frequency_jump.png\n")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 3 — Noise robustness                                          ║
# ╚═════════════════════════════════════════════════════════════════════╝
noise_levels = [0.0, 0.3, 1.0, 3.0]      # std of additive Gaussian noise
rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, len(noise_levels), figsize=(16, 6.5),
                         sharex=True,
                         gridspec_kw={'height_ratios': [1, 2.2]})
for j, sigma_n in enumerate(noise_levels):
    noisy = sig + sigma_n * rng.standard_normal(len(sig))
    sc    = cwt(noisy, t_sig, scales)

    axes[0][j].plot(t_sig, noisy, color='steelblue', lw=0.6)
    axes[0][j].axvline(0.5, color='crimson', lw=0.8, ls='--')
    axes[0][j].set_title(f"σ_noise = {sigma_n}", fontsize=10, fontweight='bold')
    axes[0][j].set_ylim(-5, 5)
    if j == 0: axes[0][j].set_ylabel("Amplitude")

    axes[1][j].imshow(sc, aspect='auto', origin='lower',
                      extent=[t_sig[0], t_sig[-1], 0, len(scales)],
                      cmap='inferno')
    axes[1][j].axvline(0.5, color='white', lw=0.8, ls='--', alpha=0.7)
    axes[1][j].set_xlabel("Time (b)")
    if j == 0:
        axes[1][j].set_yticks(np.linspace(0, len(scales), 5))
        axes[1][j].set_yticklabels([f'{s:.3f}' for s in
            scales[np.linspace(0, len(scales)-1, 5, dtype=int)]])
        axes[1][j].set_ylabel("Scale (a)")
    else:
        axes[1][j].set_yticks([])

fig.suptitle("Effect of noise on the scalogram — small scales degrade first",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_ex3_noise.png', dpi=120)
plt.show()
print("Saved: plot_ex3_noise.png\n")


# ── Observations ──────────────────────────────────────────────────────
print("""
─── OBSERVATIONS ────────────────────────────────────────────────────
1. ∫ψ(t)dt ≈ 0: the Ricker is admissible (zero mean).
   It has no complex part, so it captures *symmetric* features
   (peaks, edges) rather than oscillations like the Morlet.

2. Two horizontal energy bands appear at the scales matching f₁
   and f₂. The transition is abrupt and tightly localised at
   t=0.5: wavelets reveal *when* a frequency change occurs,
   something the Fourier transform smears across the whole signal.

3. As noise grows, the small scales (top of the scalogram) wash
   out first: white noise has uniform power, but small-scale
   wavelets integrate over short windows, so they accumulate
   noise energy comparable to the signal. Large scales remain
   readable longer because they average over wider supports ->
   wavelet analysis is naturally more robust at low frequencies.
─────────────────────────────────────────────────────────────────────
""")
