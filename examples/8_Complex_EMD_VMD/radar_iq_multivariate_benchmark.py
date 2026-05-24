"""
=======================================================================
 Multivariate Complex-IQ Decomposition Benchmark — MCVMD vs CVMD
 ─────────────────────────────────────────────────────────────────────
 Companion to radar_iq_benchmark.py. Compares two strategies for a
 2-channel complex (IQ) radar signal:

   • CVMD per channel — apply a heterodyne-based Complex VMD to each
                        channel independently (upsample, frequency-shift,
                        real VMD, Hilbert, shift back). No cross-channel
                        constraint → mode indices may swap across
                        channels when one tone dominates differently.
   • MCVMD            — Multivariate Complex VMD: same heterodyne trick
                        per channel + MVMD (ur Rehman & Aftab 2019) on
                        the upsampled real part. Forces a SHARED
                        set of K center frequencies ω_k across all
                        channels → mode k indexes the same physical
                        target on every channel.

 Test signal (2 s @ fs = 800 Hz, 2 channels):
   x_c(t) = a_{A,c} · exp(+j 2π · 30 · t)
          + a_{B,c} · exp(-j 2π · 40 · t)
          + n_c(t)            for c = 1, 2

 Channel-specific amplitudes (asymmetric on purpose):
   ch 1: a_{A,1} = 1.0, a_{B,1} = 0.4    (target A dominant)
   ch 2: a_{A,2} = 0.4, a_{B,2} = 1.0    (target B dominant)

 Realistic scenario: two antennas / range bins see the same scene
 with different gains. CVMD-per-channel orders its 2 modes by ENERGY:
 mode 1 in ch 1 = +30 Hz (strongest there); mode 1 in ch 2 = -40 Hz
 (strongest there). Mode INDICES become unreliable as physical-source
 identifiers — bad for downstream DoA, phase-difference, beamforming.

 MCVMD picks a global (ω_A, ω_B) valid for both channels and labels
 them consistently: mode 1 = same physical target on every channel.

 Author: Philippe Ciuciu  ·  Target: UnseenLabs / Inria Academy
=======================================================================
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, resample
from vmdpy import VMD

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Multivariate Complex-IQ benchmark — CVMD per channel vs MCVMD.",
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-s', '--save', metavar='PATH', default=None,
                    help='Save the figure to PATH (PNG) instead of displaying it.')
parser.add_argument('--dpi', type=int, default=140,
                    help='DPI for the saved figure (default: 140).')
parser.add_argument('--seed', type=int, default=42,
                    help='RNG seed for the additive noise (default: 42).')
args = parser.parse_args()

np.random.seed(args.seed)


# ─────────────────────────────────────────────────────────────────────
#  Multivariate VMD (ur Rehman & Aftab 2019) — inline reference impl.
#  Replace with `from inria_academy.utils.mvmd import MVMD` if preferred.
# ─────────────────────────────────────────────────────────────────────
def mvmd(X, alpha, tau, K, DC=False, init=1, tol=1e-7, max_iter=500):
    """Joint decomposition of a multi-channel REAL signal into K modes
    that SHARE the same center frequencies ω_k across all channels.

    Parameters
    ----------
    X       : ndarray (N, C)     — N samples, C channels
    alpha   : float              — bandwidth penalty (typical 2000)
    tau     : float              — dual-ascent step (0 if no noise tol)
    K       : int                — number of modes
    DC      : bool               — enforce ω_0 = 0 (DC mode)
    init    : int                — ω init: 0=zero, 1=linspace, 2=random
    tol     : float              — convergence tol on Σ_k (Δω_k)²
    max_iter: int                — max ADMM iterations

    Returns
    -------
    u       : ndarray (K, N, C)  — modes in time domain
    omega   : ndarray (K,)       — shared center freqs (normalized 0..0.5)
    """
    N, C = X.shape
    Nm = N // 2
    # Mirror-extend to mitigate boundary artefacts
    X_ext = np.concatenate([X[Nm-1::-1], X, X[-1:-Nm-1:-1]], axis=0)
    T = X_ext.shape[0]

    # Frequency axis after fftshift (centered, cycles/sample)
    freqs = np.arange(T) / T - 0.5

    # Channel-wise FFT (centered)
    X_hat = np.fft.fftshift(np.fft.fft(X_ext, axis=0), axes=0)
    # One-sided: zero out negative frequencies
    X_hat[:T // 2, :] = 0

    # Initialize ω
    if init == 0:
        omega = np.zeros(K)
    elif init == 1:
        omega = np.linspace(0, 0.5, K + 2)[1:-1]
    else:
        omega = np.sort(0.5 * np.random.rand(K))
    if DC:
        omega[0] = 0

    # Mode spectra + dual variable
    u_hat = np.zeros((K, T, C), dtype=complex)
    lam_hat = np.zeros((T, C), dtype=complex)

    eps = 1e-12
    for it in range(max_iter):
        omega_prev = omega.copy()
        for k in range(K):
            sum_others = u_hat.sum(axis=0) - u_hat[k]
            # Wiener-like spectral filter centered at ω_k, applied per channel
            denom = 1.0 + 2.0 * alpha * (freqs[:, None] - omega[k]) ** 2
            u_hat[k] = (X_hat - sum_others + lam_hat / 2) / denom
            # Power-weighted ω update — POOLED ACROSS CHANNELS (the
            # multivariate constraint).
            if not (DC and k == 0):
                mag2 = np.abs(u_hat[k, T // 2:]) ** 2          # (T/2, C)
                num = np.sum(freqs[T // 2:, None] * mag2)
                den = np.sum(mag2) + eps
                omega[k] = num / den
        # Dual ascent
        lam_hat = lam_hat + tau * (X_hat - u_hat.sum(axis=0))
        # Convergence
        if np.sum((omega - omega_prev) ** 2) < tol:
            break

    # Reconstruct in time domain — Hermitian-symmetrize then IFFT
    u_full = np.zeros((K, T, C))
    for k in range(K):
        u_hat_full = np.zeros_like(u_hat[k])
        u_hat_full[T // 2:] = u_hat[k, T // 2:]
        u_hat_full[1:T // 2] = np.conj(u_hat[k, -1:T // 2:-1])
        u_full[k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat_full, axes=0), axis=0))

    # Crop mirror
    u = u_full[:, Nm:Nm + N, :]
    return u, omega


# ─────────────────────────────────────────────────────────────────────
#  Complex VMD (heterodyne trick) — per-channel
# ─────────────────────────────────────────────────────────────────────
def cvmd(z, fs_, K, alpha=2000, tol=1e-7):
    """Heterodyne trick: upsample×2 → shift +fs/2 → real VMD → Hilbert → shift back → downsample."""
    Nz = len(z)
    z_up = resample(z, 2 * Nz)
    n_up = np.arange(2 * Nz)
    shift_up = np.exp(+1j * np.pi / 2 * n_up)
    z_shifted = z_up * shift_up
    modes_real, _, _ = VMD(z_shifted.real, alpha, 0, K, 0, 1, tol)
    modes_analytic = np.array([hilbert(m) for m in modes_real])
    modes_back = modes_analytic * np.conj(shift_up)
    return modes_back[:, ::2]


# ─────────────────────────────────────────────────────────────────────
#  Multivariate Complex VMD — heterodyne trick per channel + MVMD shared ω
# ─────────────────────────────────────────────────────────────────────
def mcvmd(X, fs_, K, alpha=2000, tol=1e-7):
    """X: (N, C) complex. Returns u (K, N, C) complex, omega_hz (K,)."""
    N, C = X.shape
    # Upsample ×2 per channel
    X_up = np.zeros((2 * N, C), dtype=complex)
    for c in range(C):
        X_up[:, c] = resample(X[:, c], 2 * N)
    n_up = np.arange(2 * N)
    shift_up = np.exp(+1j * np.pi / 2 * n_up)
    X_shifted = X_up * shift_up[:, None]

    # MVMD on the real part — shared ω across channels (the key step)
    u_real, omega_norm = mvmd(X_shifted.real, alpha=alpha, tau=0, K=K,
                               DC=False, init=1, tol=tol)
    # u_real: (K, 2N, C)

    # Hilbert (analytic) per channel per mode
    u_analytic = np.zeros_like(u_real, dtype=complex)
    for k in range(K):
        for c in range(C):
            u_analytic[k, :, c] = hilbert(u_real[k, :, c])

    # Shift back and downsample
    u_back = u_analytic * np.conj(shift_up[None, :, None])
    u_out = u_back[:, ::2, :]

    # Convert ω from normalized (cycles/sample at fs_up = 2·fs) to signed Hz
    # In shifted real space, true freq = ω·fs_up - fs/2 = 2·ω·fs - fs/2
    omega_hz = 2 * omega_norm * fs_ - fs_ / 2
    return u_out, omega_hz


# ── 1. Synthetic 2-channel complex IQ signal ─────────────────────────
fs = 800.0
T  = 2.0
t  = np.arange(0, T, 1 / fs)
N  = len(t)
C  = 2

f_A, f_B = +30.0, -40.0
# Channel-specific amplitudes — asymmetric to force mode-index swap in
# per-channel CVMD. This is the realistic case of two antennas / range
# bins seeing the same scene with different gains.
a_A = np.array([1.0, 0.4])     # tone A amplitudes (ch1, ch2)
a_B = np.array([0.4, 1.0])     # tone B amplitudes (ch1, ch2)

X_clean = np.zeros((N, C), dtype=complex)
for c in range(C):
    X_clean[:, c] = (a_A[c] * np.exp(1j * 2 * np.pi * f_A * t)
                     + a_B[c] * np.exp(1j * 2 * np.pi * f_B * t))

SNR_dB = 15.0
X = X_clean.copy()
for c in range(C):
    sig_pow = np.mean(np.abs(X_clean[:, c]) ** 2)
    n_pow   = sig_pow / 10 ** (SNR_dB / 10)
    noise   = np.sqrt(n_pow / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    X[:, c] += noise

print(f"Signal: 2 channels @ fs = {fs:.0f} Hz, T = {T:.1f} s")
print(f"  Tone A: f = {f_A:+.0f} Hz, amplitudes ch1 = {a_A[0]:.1f}, ch2 = {a_A[1]:.1f}")
print(f"  Tone B: f = {f_B:+.0f} Hz, amplitudes ch1 = {a_B[0]:.1f}, ch2 = {a_B[1]:.1f}")
print(f"  SNR = {SNR_dB:.0f} dB\n")

K, alpha = 2, 2000

# ── 2. CVMD per channel ──────────────────────────────────────────────
print("[1/2] CVMD per channel (independent)...")
modes_cvmd = np.zeros((K, N, C), dtype=complex)
for c in range(C):
    modes_cvmd[:, :, c] = cvmd(X[:, c], fs, K, alpha)

# ── 3. MCVMD (multivariate) ──────────────────────────────────────────
print("[2/2] MCVMD (multivariate, shared ω)...")
modes_mcvmd, omega_mcvmd_hz = mcvmd(X, fs, K, alpha)


# ── 4. Estimate ω from each recovered mode's peak ────────────────────
def peak_freq(z, fs_):
    Z = np.fft.fftshift(np.fft.fft(z))
    f = np.fft.fftshift(np.fft.fftfreq(len(z), 1 / fs_))
    return f[np.argmax(np.abs(Z))]

omega_cvmd = np.zeros((K, C))
for c in range(C):
    for k in range(K):
        omega_cvmd[k, c] = peak_freq(modes_cvmd[k, :, c], fs)

print("\nRecovered center frequencies (Hz):")
print("  CVMD per channel  (independent → mode indices may swap):")
for c in range(C):
    print(f"    ch{c+1}: mode 1 = {omega_cvmd[0, c]:+6.1f} Hz   "
          f"mode 2 = {omega_cvmd[1, c]:+6.1f} Hz")
print("  MCVMD  (shared ω across channels):")
print(f"           mode 1 = {omega_mcvmd_hz[0]:+6.1f} Hz   "
      f"mode 2 = {omega_mcvmd_hz[1]:+6.1f} Hz\n")


# ── 5. Visualization ─────────────────────────────────────────────────
def spec_db(z, fs_):
    Z = np.fft.fftshift(np.fft.fft(z))
    f = np.fft.fftshift(np.fft.fftfreq(len(z), 1 / fs_))
    P = 20 * np.log10(np.abs(Z) / np.abs(Z).max() + 1e-12)
    return f, P

def plot_spec(ax, z, fs_, title, color):
    f, P = spec_db(z, fs_)
    ax.plot(f, P, color=color, lw=1.4)
    for fe, lab in [(+30, '+30'), (-40, '-40')]:
        ax.axvline(fe, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.text(fe, 5, lab, ha='center', fontsize=8.5, color='red', alpha=0.8)
    ax.set_xlim(-100, 100); ax.set_ylim(-50, 10)
    ax.set_title(title, fontsize=10, color='#1E2761',
                 fontweight='bold', loc='left')
    ax.grid(alpha=0.3)


fig = plt.figure(figsize=(14, 13))
gs  = GridSpec(5, 2, figure=fig, hspace=0.6, wspace=0.18,
               left=0.06, right=0.98, top=0.95, bottom=0.05)

# Row 0 — ground truth per channel (with annotated channel-specific gains)
for c in range(C):
    plot_spec(fig.add_subplot(gs[0, c]), X_clean[:, c], fs,
              f"({'ab'[c]})  Ground truth ch {c+1}  "
              f"(a_A = {a_A[c]:.1f}, a_B = {a_B[c]:.1f})",
              '#1a1a2e')

# Rows 1-2 — CVMD per channel: SAME color for "mode 1" across both
# channels so reader sees the SAME COLOR peaking at DIFFERENT freqs.
cvmd_colors = ['#e74c3c', '#f39c12']    # mode 1 red, mode 2 orange
for k in range(K):
    for c in range(C):
        ax = fig.add_subplot(gs[1 + k, c])
        plot_spec(ax, modes_cvmd[k, :, c], fs,
                  f"({'cdef'[2*k+c]})  CVMD per channel — ch{c+1}, mode {k+1}  "
                  f"(ω̂ ≈ {omega_cvmd[k, c]:+5.1f} Hz)",
                  cvmd_colors[k])

# Rows 3-4 — MCVMD: same idea, mode k same color across channels;
# the SAME color now peaks at the SAME freq in both channels.
mcvmd_colors = ['#2ecc71', '#3498db']   # mode 1 green, mode 2 blue
for k in range(K):
    for c in range(C):
        ax = fig.add_subplot(gs[3 + k, c])
        plot_spec(ax, modes_mcvmd[k, :, c], fs,
                  f"({'ghij'[2*k+c]})  MCVMD — ch{c+1}, mode {k+1}  "
                  f"(ω̂ ≈ {omega_mcvmd_hz[k]:+5.1f} Hz, SHARED)",
                  mcvmd_colors[k])

# Bottom-row x-labels, left-column y-labels
for ax in fig.axes[-2:]:
    ax.set_xlabel("frequency (Hz)", fontsize=10)
for i in (0, 2, 4, 6, 8):
    fig.axes[i].set_ylabel("|FFT|² (dB)", fontsize=9.5)

fig.suptitle("Multivariate Complex-IQ decomposition — CVMD per channel  vs  MCVMD (shared ω)",
             fontsize=13, fontweight='bold', y=0.985)

# Save or display
if args.save:
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or '.', exist_ok=True)
    plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved → {args.save}")
else:
    plt.show()
