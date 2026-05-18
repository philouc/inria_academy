"""
=======================================================================
 Script 3: Benchmark — CWT vs Fourier vs Short-Time Fourier (Gabor)
=======================================================================
 Topics covered:
   • Fourier sinusoids vs Morlet wavelets (time localization)
   • Scaling and translation of a mother wavelet
   • Admissibility condition (zero mean) and C_ψ
   • Continuous Wavelet Transform (CWT) via convolution + scalogram
   • Parseval check for the CWT (analytic Morlet, real signal)
   • Fourier analysis of a chirp + transient
   • Short-Time Fourier Transform (Gabor) with a Gaussian window
     — illustration of the Heisenberg trade-off σ_t · σ_f = 1/(4π)

=======================================================================
Run:  python 03.benchmark_CWT_vs_Fourier_STFTGabor.py
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



# ── 4. Continuous Wavelet Transform (CWT) via convolution ─────────────
np.random.seed(0)
t_sig = np.linspace(0, 1, 1024)
dt    = t_sig[1] - t_sig[0]
# Chirp signal: frequency increases over time  +  transient at t=0.7
sig = (np.sin(2 * np.pi * (5 + 30 * t_sig) * t_sig)
       + 1.5 * np.exp(-200 * (t_sig - 0.7)**2))

# --- Admissibility constant  C_ψ = ∫₀^∞ |ψ̂(ω)|² / ω  dω ---
# No closed form for the standard Morlet (only approximately admissible
# for ω₀ ≥ 5); evaluate numerically on a densely-sampled analytic mother.
t_psi   = np.linspace(-30, 30, 8192)
dt_psi  = t_psi[1] - t_psi[0]
psi     = morlet(t_psi, analytic=True)
psi_hat = np.fft.fft(psi) * dt_psi                      # continuous-time FT
omega   = 2 * np.pi * np.fft.fftfreq(len(t_psi), d=dt_psi)
pos     = omega > 0
order   = np.argsort(omega[pos])
om_p    = omega[pos][order]
pw_p    = np.abs(psi_hat[pos][order])**2
C_psi   = np.trapezoid(pw_p / om_p, om_p)
print(f"Admissibility constant  C_ψ ≈ {C_psi:.4f}")

# --- CWT with the analytic Morlet ---
scales_cwt = np.logspace(np.log10(0.005), np.log10(0.25), 60)
W_squared  = np.zeros((len(scales_cwt), len(t_sig)))     # |W_x(a,b)|²

for k, a in enumerate(scales_cwt):
    tau    = np.linspace(-4 * a, 4 * a, max(int(8 * a / dt), 10))
    kernel = morlet_scaled(tau, a, 0, analytic=True)
    # Cross-correlation  ∫ x(t) ψ*((t-b)/a) dt / √a  via convolution with
    # the conjugate-reversed kernel; multiply by dt for the Riemann sum.
    conv   = np.convolve(sig, np.conj(kernel)[::-1], mode='same') * dt
    # np.convolve(..., mode='same') returns length max(len(sig), len(kernel)).
    # At large scales the kernel exceeds len(sig); trim centrally to match.
    if conv.size != sig.size:
        start = (conv.size - sig.size) // 2
        conv  = conv[start:start + sig.size]
    W_squared[k] = np.abs(conv)**2

# --- Parseval check ---
# Energy density:  E(a,b) = |W_x(a,b)|² / (C_ψ · a²)
# Reconstruction measure on log-spaced scales:  da/a² = (Δ ln a) / a
# Factor 2: real signal × analytic wavelet sees only the positive-freq half.
log_spacing = np.diff(np.log(scales_cwt)).mean()
energy_cwt  = 2 * np.sum(W_squared * (log_spacing / scales_cwt)[:, None]) * dt / C_psi
energy_sig  = np.sum(sig**2) * dt
# Frequency band covered by the scale grid (Morlet:  f ≈ ω₀ / (2π a))
omega_0 = 2 * 2.5                                       # = 2c, with default c=2.5
f_min   = omega_0 / (2 * np.pi * scales_cwt.max())
f_max   = omega_0 / (2 * np.pi * scales_cwt.min())
print(f"∫|x(t)|² dt              = {energy_sig:.4f}")
print(f"(2/C_ψ)∬|W|²/a² da db    = {energy_cwt:.4f}")
print(f"Energy recovery ratio    = {energy_cwt/energy_sig:.4f}")
print(f"Scale grid covers f ∈ [{f_min:.2f}, {f_max:.1f}] Hz  "
      f"— gap to 1.0 = energy outside this band")

scalogram = W_squared                                    # keep variable name for plot

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(t_sig, sig, 'steelblue', lw=1.2)
axes[0].set_title("Signal: linear chirp + transient at t=0.7", fontsize=11)
axes[0].set_ylabel("Amplitude")

im = axes[1].imshow(
    scalogram, aspect='auto', origin='lower',
    extent=[t_sig[0], t_sig[-1], 0, len(scales_cwt)],
    cmap='inferno'
)
axes[1].set_yticks(np.linspace(0, len(scales_cwt), 5))
axes[1].set_yticklabels([f'{s:.3f}' for s in scales_cwt[np.linspace(0, len(scales_cwt)-1, 5, dtype=int)]])
axes[1].set_title("Scalogram  |CWT(a,b)|²  — Morlet wavelet", fontsize=11)
axes[1].set_xlabel("Time (b)"); axes[1].set_ylabel("Scale (a)")
plt.colorbar(im, ax=axes[1], label=r'$|W_x(a,b)|^2$')
plt.tight_layout()
plt.savefig('plot1c_scalogram.png', dpi=120)
plt.show()
print("Saved: plot1c_scalogram.png")


# ── 5. Fourier analysis of the same signal ─────────────────────────────
# The DFT gives a global view of the signal's frequency content but loses
# all temporal information: we see WHICH frequencies are present, not WHEN.
freqs_f = np.fft.rfftfreq(len(sig), d=dt)
X       = np.fft.rfft(sig) * dt                   # continuous-time FT approx
mag     = np.abs(X)

# Parseval sanity check (one-sided spectrum needs a factor of 2)
df       = freqs_f[1] - freqs_f[0]
energy_f = 2 * np.sum(mag**2) * df - mag[0]**2 * df            # DC counted once
print(f"\n[Fourier] ∫|x|² dt  = {np.sum(sig**2)*dt:.4f}   "
      f"2·∫|X(f)|² df = {energy_f:.4f}   (Parseval ✓)")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.4]})
axes[0].plot(t_sig, sig, 'steelblue', lw=1.2)
axes[0].set_title("Signal: linear chirp (5→65 Hz) + transient at t=0.7", fontsize=11)
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("x(t)")
axes[0].grid(alpha=0.3)

axes[1].plot(freqs_f, mag, 'crimson', lw=1.2)
axes[1].axvspan(5, 65, alpha=0.10, color='steelblue', label='chirp band  [5–65 Hz]')
axes[1].axvspan(0, 4.5, alpha=0.18, color='orange',    label='transient band  [0–4.5 Hz]')
axes[1].set_title(r"$|X(f)|$  —  global spectrum (no time localisation)", fontsize=11)
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("|X(f)|")
axes[1].set_xlim(0, 80)
axes[1].legend(fontsize=9, loc='upper right'); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot1d_fourier.png', dpi=120)
plt.show()
print("Saved: plot1d_fourier.png")


# ── 6. Short-Time Fourier Transform with Gaussian window (Gabor) ───────
# STFT:  S_x(t, f) = ∫ x(τ) g*(τ − t) e^{−i2πfτ} dτ
# With a Gaussian window the STFT becomes the Gabor transform, which
# saturates the Heisenberg uncertainty bound  σ_t · σ_f = 1/(4π).
# Window width is FIXED → resolution is the same at every frequency
# (constant Δf), unlike the CWT whose Δf scales with f (constant Q).
def stft_gaussian(x, dt, sigma_w, hop=2):
    """STFT with an L²-normalised Gaussian window of std σ_w (seconds)."""
    N         = len(x)
    n         = np.arange(N)
    centers   = np.arange(0, N, hop)
    freqs     = np.fft.rfftfreq(N, d=dt)
    S         = np.empty((len(freqs), len(centers)), dtype=complex)
    for i, c in enumerate(centers):
        win    = np.exp(-0.5 * ((n - c) * dt / sigma_w)**2)
        win   /= np.sqrt(np.sum(win**2) * dt)             # ‖g‖₂ = 1
        S[:, i] = np.fft.rfft(x * win) * dt
    return freqs, centers * dt, S

# Two window widths to illustrate the time-frequency trade-off
sigmas_w = [0.015, 0.080]                                 # narrow vs wide
specs    = []
for sw in sigmas_w:
    f_s, t_s, S = stft_gaussian(sig, dt, sw, hop=2)
    specs.append((sw, f_s, t_s, np.abs(S)**2))
    # Heisenberg: σ_f = 1/(4π σ_t)  for a Gaussian window
    print(f"[STFT] σ_t = {sw:.3f} s  →  σ_f ≈ {1/(4*np.pi*sw):.2f} Hz  "
          f"(product σ_t·σ_f = 1/(4π) ≈ {1/(4*np.pi):.4f})")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                        gridspec_kw={'height_ratios': [0.7, 1, 1]})
axes[0].plot(t_sig, sig, 'steelblue', lw=1.0)
axes[0].set_title("Signal", fontsize=11); axes[0].set_ylabel("x(t)")
axes[0].grid(alpha=0.3)

for ax, (sw, f_s, t_s, Sxx) in zip(axes[1:], specs):
    im = ax.imshow(Sxx, aspect='auto', origin='lower',
                   extent=[t_s[0], t_s[-1], f_s[0], f_s[-1]],
                   cmap='inferno', vmax=np.percentile(Sxx, 99.5))
    ax.set_ylim(0, 80)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        rf"STFT $|S(t,f)|^2$ — Gaussian window  "
        rf"$\sigma_t = {sw:.3f}$ s,  $\sigma_f \approx {1/(4*np.pi*sw):.1f}$ Hz",
        fontsize=10.5)
    plt.colorbar(im, ax=ax, label=r'$|S|^2$')

axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig('plot1e_stft.png', dpi=120)
plt.show()
print("Saved: plot1e_stft.png")
