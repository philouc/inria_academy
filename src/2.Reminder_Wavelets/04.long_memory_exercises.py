"""
=======================================================================
 Wavelet Theory — Exercises 2 & 3 (sym4 version)
=======================================================================
 2. Add Gaussian white noise (std=0.5) to fGn — compare degradation
    of the Fourier and Symlet-4 wavelet estimators.
 3. Simulate fGn whose Hurst exponent H increases linearly from 0.55
    to 0.95 along the signal. Can the local sym4 estimator track it?
=======================================================================
 Run:  python 04.long_memory_exercises.py
 Requires: numpy, matplotlib

Author: Philippe Ciuciu
Date: 04/09/2026
Target: UnseenLabs
=======================================================================

Author: Philippe Ciuciu
Date: 04/09/2026
Target: UnseenLabs
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d

np.random.seed(42)


# ── Symlet 4 filter coefficients + one-level DWT step ─────────────────
SYM4_LO = np.array([
    -0.07576571478927333, -0.02963552764599851,
     0.4976186676320155,   0.8037387518059161,
     0.29785779560527736, -0.09921954357684722,
    -0.012603967262037833, 0.0322231006040427,
])
SYM4_HI = ((-1.0) ** np.arange(len(SYM4_LO))) * SYM4_LO[::-1]


def sym4_dwt_step(x):
    L = len(SYM4_LO)
    x_ext = np.concatenate([x, x[:L - 1]])
    lo = np.correlate(x_ext, SYM4_LO, mode='valid')[::2]
    hi = np.correlate(x_ext, SYM4_HI, mode='valid')[::2]
    return lo, hi


def simulate_fgn(N, H, seed=None):
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


def estimate_H_fourier(x):
    N = len(x)
    f = np.fft.rfftfreq(N)[1:N // 4]
    S = np.abs(np.fft.rfft(x)[1:N // 4]) ** 2
    slope, *_ = stats.linregress(np.log(f), np.log(S))
    return -(slope + 1) / 2


def wavelet_log2_var(x, n_scales=8):
    """Return arrays (scales j, log2 Var[D_j]) using sym4."""
    scales, log_vars = [], []
    a = np.array(x, dtype=float)
    for j in range(1, n_scales + 1):
        if len(a) < 2 * len(SYM4_LO):
            break
        lo, hi = sym4_dwt_step(a)
        if len(hi) > 8:
            scales.append(j)
            log_vars.append(np.log2(np.var(hi) + 1e-15))
        a = lo
    return np.array(scales), np.array(log_vars)


def estimate_H_wavelet(x, n_scales=8):
    sj, lv = wavelet_log2_var(x, n_scales)
    slope, *_ = stats.linregress(sj, lv)
    return (slope - 1) / 2


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Exercise 2 — Robustness to additive white noise                    ║
# ╚═════════════════════════════════════════════════════════════════════╝
N         = 4096
noise_std = 0.5
n_reps    = 20
true_H    = np.arange(0.55, 0.97, 0.05)
rng_noise = np.random.default_rng(123)

clean_f, clean_w, noisy_f, noisy_w = [], [], [], []
for H in true_H:
    cf, cw, nf, nw = [], [], [], []
    for rep in range(n_reps):
        x = simulate_fgn(N, H, seed=rep * 100 + int(H * 100))
        cf.append(estimate_H_fourier(x))
        cw.append(estimate_H_wavelet(x))
        x_n = x + noise_std * rng_noise.standard_normal(N)
        nf.append(estimate_H_fourier(x_n))
        nw.append(estimate_H_wavelet(x_n))
    clean_f.append((np.mean(cf), np.std(cf)))
    clean_w.append((np.mean(cw), np.std(cw)))
    noisy_f.append((np.mean(nf), np.std(nf)))
    noisy_w.append((np.mean(nw), np.std(nw)))

unpack = lambda L: (np.array([v[0] for v in L]), np.array([v[1] for v in L]))
mcf, scf = unpack(clean_f); mnf, snf = unpack(noisy_f)
mcw, scw = unpack(clean_w); mnw, snw = unpack(noisy_w)

# Diagnostic: log2 Var[D_j] for ONE realisation at H=0.8 — clean vs noisy
x_demo   = simulate_fgn(N, 0.8, seed=777)
x_demo_n = x_demo + noise_std * rng_noise.standard_normal(N)
sj_c, lv_c = wavelet_log2_var(x_demo)
sj_n, lv_n = wavelet_log2_var(x_demo_n)

fig = plt.figure(figsize=(15, 6))
gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
fig.suptitle(f"Exercise 2 — Robustness to additive white noise  (σ_noise = {noise_std})",
             fontsize=12, fontweight='bold')

# Panel 1: Fourier estimator
ax = fig.add_subplot(gs[0, 0])
ax.plot(true_H, true_H, 'k--', lw=1.2, label='perfect')
ax.errorbar(true_H, mcf, yerr=scf, fmt='o-', color='#e74c3c',
            capsize=3, lw=1.5, alpha=0.9, label='Fourier — clean')
ax.errorbar(true_H, mnf, yerr=snf, fmt='o--', color='#c0392b',
            capsize=3, lw=1.5, alpha=0.55, mfc='white',
            label=f'Fourier — +noise')
ax.set_xlabel('True H'); ax.set_ylabel('Estimated H')
ax.set_title('Fourier (periodogram slope)', fontsize=10, fontweight='bold')
ax.set_ylim(0.3, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 2: Wavelet estimator
ax = fig.add_subplot(gs[0, 1])
ax.plot(true_H, true_H, 'k--', lw=1.2, label='perfect')
ax.errorbar(true_H, mcw, yerr=scw, fmt='s-', color='#2ecc71',
            capsize=3, lw=1.5, alpha=0.9, label='sym4 — clean')
ax.errorbar(true_H, mnw, yerr=snw, fmt='s--', color='#16a085',
            capsize=3, lw=1.5, alpha=0.55, mfc='white',
            label=f'sym4 — +noise')
ax.set_xlabel('True H'); ax.set_ylabel('Estimated H')
ax.set_title('Wavelet variance (sym4)', fontsize=10, fontweight='bold')
ax.set_ylim(0.3, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3: diagnostic — why sym4 degrades
ax = fig.add_subplot(gs[0, 2])
ax.plot(sj_c, lv_c, 'o-', color='#2ecc71', lw=2, ms=7, label='clean')
ax.plot(sj_n, lv_n, 's--', color='#16a085', lw=2, ms=7,
        mfc='white', label='noisy')
# Reference noise variance line: log2(σ²_n)
ax.axhline(np.log2(noise_std**2), color='gray', ls=':', lw=1.5,
           label=f'log₂(σ²_n) = {np.log2(noise_std**2):.2f}')
ax.set_xlabel('Scale j (1 = finest)')
ax.set_ylabel('log₂ Var[D_j]')
ax.set_title(f'log-variance plot  (single realisation, H=0.8)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_ex2_noise_robustness.png', dpi=120)
plt.show()
print("Saved: plot_ex2_noise_robustness.png")

print(f"""
Exercise 2 — Bias under noise (averaged over H ∈ [0.55, 0.95]):
  Fourier   clean :  bias = {np.mean(mcf - true_H):+.3f}
  Fourier   noisy :  bias = {np.mean(mnf - true_H):+.3f}
  sym4      clean :  bias = {np.mean(mcw - true_H):+.3f}
  sym4      noisy :  bias = {np.mean(mnw - true_H):+.3f}

→ Fourier barely moves (it fits the lowest quarter of frequencies,
  where fGn dominates the white noise floor).
→ sym4 underestimates H because white noise contributes a constant
  σ²_n to Var[D_j] at every scale; at fine scales this floor
  flattens the log-variance plot (right panel) and tilts the
  regression line.
""")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Exercise 3 — Time-varying Hurst exponent                            ║
# ╚═════════════════════════════════════════════════════════════════════╝
N3        = 8192
n_blocks  = 16
block_len = N3 // n_blocks
H_sched   = np.linspace(0.55, 0.95, n_blocks)

# Concatenate independent fGn segments — piecewise stationary
segments = [simulate_fgn(block_len, H_i, seed=2000 + i)
            for i, H_i in enumerate(H_sched)]
x_tv     = np.concatenate(segments)
t_axis   = np.arange(N3)
H_true   = np.repeat(H_sched, block_len)

# Sliding sym4 local estimator
window, step = 512, 32
positions, H_local = [], []
for start in range(0, N3 - window + 1, step):
    positions.append(start + window // 2)
    H_local.append(estimate_H_wavelet(x_tv[start:start + window], n_scales=6))
positions = np.array(positions)
H_local   = np.array(H_local)
H_smooth  = uniform_filter1d(H_local, size=9, mode='nearest')

fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.1, 1]})
fig.suptitle("Exercise 3 — Time-varying H : sym4 local estimator tracks the ramp",
             fontsize=12, fontweight='bold')

# Signal with block boundaries
axes[0].plot(t_axis, x_tv, color='steelblue', lw=0.35)
for k in range(1, n_blocks):
    axes[0].axvline(k * block_len, color='gray', lw=0.4, alpha=0.4)
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Piecewise fGn — {n_blocks} blocks of {block_len} samples, '
                  f'H from {H_sched[0]:.2f} to {H_sched[-1]:.2f}',
                  fontsize=10)

# Local H tracking
axes[1].plot(t_axis, H_true, 'k-', lw=2, label='True H(t) (step-wise)')
axes[1].plot(positions, H_local, 'o', color='darkorange', ms=3.5, alpha=0.5,
             label=f'Raw local sym4 (window={window}, step={step})')
axes[1].plot(positions, H_smooth, '-', color='#8e44ad', lw=2.2,
             label='Smoothed (running mean, size=9)')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('H')
axes[1].set_ylim(0.3, 1.15)
axes[1].legend(fontsize=9, loc='upper left'); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_ex3_time_varying_H.png', dpi=120)
plt.show()
print("Saved: plot_ex3_time_varying_H.png")

# Tracking quality
H_true_at_pos = H_true[positions]
mae_raw      = np.mean(np.abs(H_local  - H_true_at_pos))
mae_smooth   = np.mean(np.abs(H_smooth - H_true_at_pos))
corr_raw     = np.corrcoef(H_local,  H_true_at_pos)[0, 1]
corr_smooth  = np.corrcoef(H_smooth, H_true_at_pos)[0, 1]
print(f"""
Exercise 3 — Tracking quality
  Raw local estimate :  MAE = {mae_raw:.3f}   ρ = {corr_raw:.3f}
  Smoothed (size=9)  :  MAE = {mae_smooth:.3f}   ρ = {corr_smooth:.3f}

→ The raw local estimate is noisy (one wavelet regression over only
  512 samples, with 4–5 usable scales) but follows the ramp closely.
  A short running mean (~window/step·1/4) recovers the linear trend
  with very low bias.
""")


# ── Observations ──────────────────────────────────────────────────────
print("""
─── OBSERVATIONS ────────────────────────────────────────────────────
Exercise 2. Why wavelets degrade more than Fourier under noise:
  An orthonormal DWT preserves variance, so additive white noise
  of variance σ²_n contributes σ²_n to Var[D_j] at EVERY scale.
  For fGn, Var[D_j] grows as 2^{(2H+1)j}, i.e. very fast with j.
  At fine scales the noise contribution dominates, the log-variance
  plot flattens (visible in the right panel), and the regression
  slope (and hence Ĥ) is pulled down.
  The Fourier estimator survives because it fits ONLY the low-
  frequency quarter, where fGn power vastly exceeds the white-noise
  floor: those are precisely the frequencies the wavelet estimator
  does NOT preferentially weight.
  Practical fix for the wavelet estimator under noise: drop the
  first 1–2 scales (j=1,2) from the regression.

Exercise 3 — Local tracking of H(t):
  Because the sym4 estimator is local (each estimate uses one
  512-sample window), it can follow time-varying long-memory
  behaviour, unlike the global Fourier slope which would output
  a single number for the whole record.
  Trade-off: window must be long enough for a meaningful
  regression over scales (here 4–5 usable scales) but short enough
  to resolve the H(t) variations. A light moving average over the
  raw local estimates cleans the result with negligible lag.
─────────────────────────────────────────────────────────────────────
""")
