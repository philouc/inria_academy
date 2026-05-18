"""
=======================================================================
 Wavelet Theory — Script 4: Long-Memory Processes & Hurst Estimation
                            (Symlet 4 version)
=======================================================================
 Same analysis as 04.long_memory.py but the wavelet-based Hurst
 estimator now uses the Symlet 4 (sym4) wavelet instead of Haar.

 Why sym4 over Haar:
   • 4 vanishing moments (vs 1 for Haar) → better decorrelation of
     fGn details across scales, hence less biased H estimates,
     especially for high H.
   • Smoother, near-symmetric support of length 8.
   • Still orthogonal, so the scaling relation
        Var[D_j] ∝ 2^{(2H+1)j}
     holds identically.
=======================================================================
 Run:  python 04.long_memory_sym4.py
 Requires: numpy, matplotlib, scipy
=======================================================================

Author: Philippe Ciuciu
Date: 04/09/2026
Target: UnseenLabs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)


# ── Symlet 4 filter coefficients ──────────────────────────────────────
# Decomposition low-pass (8 taps, near-symmetric)
SYM4_LO = np.array([
    -0.07576571478927333, -0.02963552764599851,
     0.4976186676320155,   0.8037387518059161,
     0.29785779560527736, -0.09921954357684722,
    -0.012603967262037833, 0.0322231006040427,
])
# High-pass derived via QMF relation:  g[n] = (-1)^n · h_lo[L-1-n]
SYM4_HI = ((-1.0) ** np.arange(len(SYM4_LO))) * SYM4_LO[::-1]


def sym4_dwt_step(x):
    """
    One level of DWT with sym4 filters, periodic boundary.
    Input  length N → output (approx, detail) each of length N//2.
    """
    L = len(SYM4_LO)
    # Periodic right-extension so the 'valid' correlation has length N
    x_ext = np.concatenate([x, x[:L - 1]])
    lo = np.correlate(x_ext, SYM4_LO, mode='valid')[::2]
    hi = np.correlate(x_ext, SYM4_HI, mode='valid')[::2]
    return lo, hi


# ── 1. Simulate Fractional Gaussian Noise (spectral method) ───────────
def simulate_fgn(N, H, seed=None):
    """
    Simulate Fractional Gaussian Noise with Hurst exponent H.
    PSD_fGn(f) ∝ |f|^{-(2H+1)}   (long-range behaviour)
    """
    if seed is not None:
        np.random.seed(seed)
    freqs     = np.fft.rfftfreq(N)[1:]
    psd_shape = freqs ** (-(2 * H + 1))
    phases    = np.random.uniform(0, 2 * np.pi, len(freqs))
    spectrum  = np.sqrt(psd_shape / 2) * (np.cos(phases) + 1j * np.sin(phases))
    full_spec = np.zeros(N // 2 + 1, dtype=complex)
    full_spec[1:] = spectrum
    x = np.fft.irfft(full_spec, n=N)
    return (x - x.mean()) / x.std()


#N = 4096
N = 16384

fig, axes = plt.subplots(3, 2, figsize=(14, 9))
fig.suptitle("Fractional Gaussian Noise — Effect of Hurst Exponent H",
             fontsize=12, fontweight='bold')

for row, H in enumerate([0.5, 0.7, 0.9]):
    x = simulate_fgn(N, H, seed=row)
    t = np.arange(N)

    ax = axes[row, 0]
    ax.plot(t, x, lw=0.6,
            color=['steelblue', 'darkorange', 'seagreen'][row])
    ax.set_title(f"fGn  H={H}  {'(white noise)' if H == 0.5 else '(long memory)'}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")

    ax = axes[row, 1]
    f = np.fft.rfftfreq(N)[1:N // 2]
    S = np.abs(np.fft.rfft(x)[1:N // 2]) ** 2
    ax.loglog(f, S, alpha=0.5,
              color=['steelblue', 'darkorange', 'seagreen'][row],
              lw=0.8, label='Periodogram')
    slope_theory = -(2 * H + 1)
    f_ref = np.array([f[5], f[-5]])
    ax.loglog(f_ref, S[5] * (f_ref / f[5]) ** slope_theory,
              'k--', lw=2, label=f'Slope = {slope_theory:.1f}')
    ax.legend(fontsize=8)
    ax.set_title(f"Power Spectral Density  H={H}", fontsize=10)
    ax.set_xlabel("Frequency (log)"); ax.set_ylabel("PSD (log)")

plt.tight_layout()
plt.savefig('plot3a_fgn_comparison.png', dpi=120)
plt.show()
print("Saved: plot3a_fgn_comparison.png")


# ── 2. Hurst estimation: Fourier vs Wavelet (sym4) ────────────────────
def estimate_H_fourier(x):
    """Estimate H from the periodogram slope (log-log regression)."""
    N = len(x)
    f = np.fft.rfftfreq(N)[1:N // 4]
    S = np.abs(np.fft.rfft(x)[1:N // 4]) ** 2
    slope, intercept, r, p, se = stats.linregress(np.log(f), np.log(S))
    H_est = -(slope + 1) / 2
    return H_est, slope, r ** 2


def estimate_H_wavelet(x, n_scales=8):
    """
    Estimate H using wavelet variance at dyadic scales (Symlet 4).
    For fGn:  log2(Var[D_j]) = (2H+1)·j + const
    → linear regression in log2 space gives H.
    """
    scales, log_vars = [], []
    a = np.array(x, dtype=float)
    for j in range(1, n_scales + 1):
        if len(a) < 2 * len(SYM4_LO):       # too short to decompose
            break
        lo, hi = sym4_dwt_step(a)
        if len(hi) > 8:
            scales.append(j)
            log_vars.append(np.log2(np.var(hi) + 1e-15))
        a = lo
    sj = np.array(scales); lv = np.array(log_vars)
    slope, intercept, r, p, se = stats.linregress(sj, lv)
    H_est = (slope - 1) / 2
    return H_est, slope, r ** 2, sj, lv


true_H  = np.arange(0.55, 0.97, 0.05)
H_four  = []
H_wavel = []

for H in true_H:
    reps_f, reps_w = [], []
    for rep in range(15):
        x = simulate_fgn(N, H, seed=rep * 100 + int(H * 100))
        hf, _, _ = estimate_H_fourier(x)
        hw, _, _, _, _ = estimate_H_wavelet(x)
        reps_f.append(hf); reps_w.append(hw)
    H_four.append((np.mean(reps_f),  np.std(reps_f)))
    H_wavel.append((np.mean(reps_w), np.std(reps_w)))

means_f = np.array([v[0] for v in H_four]);  stds_f = np.array([v[1] for v in H_four])
means_w = np.array([v[0] for v in H_wavel]); stds_w = np.array([v[1] for v in H_wavel])

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(true_H, true_H, 'k--', lw=1.5, label='Perfect estimator')
ax.errorbar(true_H, means_f, yerr=stds_f, fmt='o-', color='#e74c3c',
            capsize=4, lw=1.8, label='Fourier (periodogram slope)')
ax.errorbar(true_H, means_w, yerr=stds_w, fmt='s-', color='#2ecc71',
            capsize=4, lw=1.8, label='Wavelet variance (Symlet 4)')
ax.set_xlabel("True Hurst exponent H", fontsize=11)
ax.set_ylabel("Estimated H ± std (15 realisations)", fontsize=11)
ax.set_title("Hurst Exponent Estimation: Fourier vs Wavelet (sym4)\n"
             "(N=4096, 15 Monte Carlo realisations per H)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot3b_hurst_comparison.png', dpi=120)
plt.show()
print("Saved: plot3b_hurst_comparison.png")


# ── 3. Non-stationarity detection ─────────────────────────────────────
x1 = simulate_fgn(N // 2, H=0.6,  seed=10)
x2 = simulate_fgn(N // 2, H=0.92, seed=20)
x_break = np.concatenate([x1, x2])

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=False)
fig.suptitle("Non-Stationarity Detection: H changes at t=N/2\n"
             "(Fourier is blind; sym4 wavelets reveal the break)",
             fontsize=12, fontweight='bold')

axes[0].plot(x_break, lw=0.6, color='steelblue')
axes[0].axvline(N // 2, color='red', lw=2, ls='--',
                label='Structural break (H: 0.6 → 0.92)')
axes[0].legend()
axes[0].set_title("Signal with structural break", fontsize=10)
axes[0].set_ylabel("Amplitude")

f = np.fft.rfftfreq(N)[1:N // 2]
S = np.abs(np.fft.rfft(x_break)[1:N // 2]) ** 2
axes[1].loglog(f, S, 'steelblue', lw=0.8, alpha=0.8,
               label='Global PSD (entire signal)')
axes[1].loglog(f[:N // 8],
               S[:N // 8] * (f[:N // 8] / f[5]) ** (-(2 * 0.6  + 1) + 2) * S[5],
               'r--', lw=2, label='Slope H=0.6 (first half)')
axes[1].loglog(f[:N // 8],
               S[:N // 8] * (f[:N // 8] / f[5]) ** (-(2 * 0.92 + 1) + 2) * S[5] * 3,
               'g--', lw=2, label='Slope H=0.92 (second half)')
axes[1].set_title("Fourier PSD: single slope — structural break INVISIBLE",
                  fontsize=10)
axes[1].set_xlabel("Frequency (log)"); axes[1].set_ylabel("PSD (log)")
axes[1].legend(fontsize=8)

# Local sym4 wavelet variance
window = 256; step = 64
positions, H_local = [], []
for start in range(0, N - window, step):
    seg = x_break[start:start + window]
    H_w, _, _, _, _ = estimate_H_wavelet(seg, n_scales=5)
    positions.append(start + window // 2)
    H_local.append(H_w)
axes[2].plot(positions, H_local, 'o-', color='darkorange', lw=2, ms=5,
             label='Local H estimate (sym4, window=256)')
axes[2].axvline(N // 2, color='red',    lw=2,   ls='--', label='True break point')
axes[2].axhline(0.6,    color='#2ecc71', ls=':', lw=1.5, label='True H=0.6')
axes[2].axhline(0.92,   color='#9b59b6', ls=':', lw=1.5, label='True H=0.92')
axes[2].set_title("sym4 local H estimate: break DETECTED at t=N/2", fontsize=10)
axes[2].set_xlabel("Time"); axes[2].set_ylabel("Estimated H")
axes[2].set_ylim(0.3, 1.2); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot3c_nonstationarity.png', dpi=120)
plt.show()
print("Saved: plot3c_nonstationarity.png")

# Summary
x_full = simulate_fgn(N, H=0.8, seed=0)
hf, sf, r2f = estimate_H_fourier(x_full)
hw, sw, r2w, sj, lv = estimate_H_wavelet(x_full)
print(f"""
─── Summary: H estimation for H=0.8, N={N} ─────────────────────────
  Fourier (periodogram slope):
    Ĥ = {hf:.3f}   (R² = {r2f:.3f})

  Wavelet variance (Symlet 4):
    Ĥ = {hw:.3f}   (R² = {r2w:.3f})
─────────────────────────────────────────────────────────────────────
""")

print("""
─── EXERCISE ────────────────────────────────────────────────────────
1. Increase N to 16384. Does the wavelet estimator become more
   accurate? What about the Fourier estimator?

2. Add Gaussian white noise (std=0.5) to the fGn signal. Which
   estimator degrades more? Why?

3. Simulate fGn where H increases linearly from 0.55 to 0.95 over
   the signal length. Can the wavelet local estimator track this?
─────────────────────────────────────────────────────────────────────
""")
