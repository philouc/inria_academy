"""
=======================================================================
 Wavelet Theory — Exercise solutions for Script 2
=======================================================================
 1. STFT trade-off with window sizes 16, 64, 512
 2. Two Gaussian transients at t=1.0s and t=1.5s — STFT vs DWT D1
 3. Operation counts: DWT O(N) vs FFT O(N log N) for N = 2^20
=======================================================================
 Run:  python 02.time_frequency_tiling_exercise.py
 Requires: numpy, matplotlib, scipy

 Author: Philippe Ciuciu
 Date: 04/07/2026
 Target: UnseenLabs
 =======================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft


# ── helpers (re-used from Script 2) ───────────────────────────────────
def demo_stft(sig, fs, win_sizes, title, savepath=None):
    """STFT panel with several window sizes."""
    fig, axes = plt.subplots(1, len(win_sizes) + 1, figsize=(15, 4))
    t = np.arange(len(sig)) / fs

    axes[0].plot(t, sig, 'steelblue', lw=0.8)
    axes[0].set_title("Signal", fontsize=10, fontweight='bold')
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amp")

    for ax, W in zip(axes[1:], win_sizes):
        f, t_s, Zxx = stft(sig, fs=fs, nperseg=W, noverlap=W * 3 // 4)
        ax.pcolormesh(t_s, f, np.abs(Zxx), shading='gouraud', cmap='inferno')
        ax.set_ylim(0, fs / 2)
        ax.set_title(f"W = {W}   "
                     f"Δt = {1000*W/fs:.1f} ms   Δf = {fs/W:.1f} Hz",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120)
        print(f"Saved: {savepath}")
    return fig


def haar_dwt_levels(x, n_levels):
    """Iterative Haar DWT — returns (details_finest_to_coarsest, approximation)."""
    details = []
    a = np.array(x, dtype=float)
    for _ in range(n_levels):
        N  = (len(a) // 2) * 2
        lo = (a[:N:2] + a[1:N:2]) / np.sqrt(2)
        hi = (a[:N:2] - a[1:N:2]) / np.sqrt(2)
        details.append(hi)
        a = lo
    return details, a


# ── Base signal (chirp + one transient at t=1.0s) ─────────────────────
fs    = 1000
T     = 2.0
t_arr = np.arange(int(T * fs)) / fs
sig1  = (np.sin(2 * np.pi * (10 + 80 * t_arr) * t_arr)        # chirp
       + 2.5 * np.exp(-300 * (t_arr - 1.0)**2))                # transient


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 1 — STFT windows 16, 64, 512                                  ║
# ╚═════════════════════════════════════════════════════════════════════╝
demo_stft(
    sig1, fs, win_sizes=[16, 64, 512],
    title="Part 1 — STFT trade-off: small W → fine Δt / coarse Δf,  "
          "large W → opposite",
    savepath='plot_ex1_stft_windows.png'
)
plt.show()


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 2 — Two transients at t=1.0s and t=1.5s                       ║
# ╚═════════════════════════════════════════════════════════════════════╝
sig2 = (np.sin(2 * np.pi * (10 + 80 * t_arr) * t_arr)
      + 2.5 * np.exp(-300 * (t_arr - 1.0)**2)
      + 2.5 * np.exp(-300 * (t_arr - 1.5)**2))

# (a) STFT detection at three window sizes
demo_stft(
    sig2, fs, win_sizes=[16, 64, 512],
    title="Part 2a — Two transients: STFT detects both at small W, "
          "blurs them at large W",
    savepath='plot_ex2_stft_two_transients.png'
)
plt.show()

# (b) DWT — finest detail D1 picks up sharp events
details, approx = haar_dwt_levels(sig2, n_levels=5)
d1 = details[0]
t_d1 = np.linspace(0, T, len(d1))

fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True,
                         gridspec_kw={'height_ratios': [1.2, 1, 1]})
fig.suptitle("Part 2b — DWT detail D1 (finest scale) localises both transients",
             fontsize=12, fontweight='bold')

axes[0].plot(t_arr, sig2, 'steelblue', lw=0.7)
for tk in (1.0, 1.5):
    axes[0].axvline(tk, color='crimson', lw=1, ls='--', alpha=0.5)
axes[0].set_title("Signal (chirp + 2 Gaussian transients)", fontsize=10)
axes[0].set_ylabel("Amplitude")

axes[1].plot(t_d1, d1, color='#e05c5c', lw=0.7)
for tk in (1.0, 1.5):
    axes[1].axvline(tk, color='crimson', lw=1, ls='--', alpha=0.5)
axes[1].set_title(f"D1 — band ≈ [{fs/4:.0f}, {fs/2:.0f}] Hz  "
                  f"(sharp spikes at the transient locations)", fontsize=10)
axes[1].set_ylabel("D1 coeff")

# |D1| envelope makes the localisation even more obvious
axes[2].plot(t_d1, np.abs(d1), color='#8e44ad', lw=0.9)
for tk in (1.0, 1.5):
    axes[2].axvline(tk, color='crimson', lw=1, ls='--', alpha=0.5)
axes[2].set_title("|D1|  — magnitude envelope", fontsize=10)
axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("|D1|")

plt.tight_layout()
plt.savefig('plot_ex2_dwt_d1.png', dpi=120)
plt.show()
print("Saved: plot_ex2_dwt_d1.png")


# ╔═════════════════════════════════════════════════════════════════════╗
# ║  Part 3 — DWT O(N) vs FFT O(N log N)  for  N = 2^20                  ║
# ╚═════════════════════════════════════════════════════════════════════╝
N      = 2**20
log2N  = int(np.log2(N))
fft_ops = N * log2N           # ~ N · log₂(N)
dwt_ops = 2 * N               # geometric sum N + N/2 + N/4 + ... ≈ 2N
speedup = fft_ops / dwt_ops   # theoretical

# Empirical timing — numpy FFT (C) vs pure-Python Haar
rng   = np.random.default_rng(0)
x_big = rng.standard_normal(N)

t0 = time.perf_counter(); _ = np.fft.fft(x_big);                          fft_t = time.perf_counter() - t0
t0 = time.perf_counter(); _ = haar_dwt_levels(x_big, n_levels=log2N);     dwt_t = time.perf_counter() - t0

print(f"""
Part 3 — N = 2^{log2N} = {N:,}
  Theoretical operation counts
     FFT  ~ N · log₂(N)   = {fft_ops:>14,}
     DWT  ~ 2 N           = {dwt_ops:>14,}
     Theoretical speedup ≈ log₂(N)/2 ≈ {speedup:.1f}×
     (Simpler statement: O(N log N) vs O(N)  →  factor log₂(N) = {log2N}×)

  Empirical wall-clock (this machine)
     numpy FFT (C)          : {fft_t*1000:8.2f} ms
     Python Haar DWT        : {dwt_t*1000:8.2f} ms
  → numpy.fft is heavily vectorised C, the Haar above is pure Python,
    so constant factors dominate here. With a C/SIMD DWT (e.g. pywt,
    lifting scheme) the asymptotic ~{log2N}× advantage is realised in
    practice.
""")


# ── Observations ──────────────────────────────────────────────────────
print("""
─── OBSERVATIONS ────────────────────────────────────────────────────
1. STFT trade-off (Heisenberg in action):
   • W=16   → Δt=16 ms, Δf=62.5 Hz. The transient is perfectly
              localised in time but the chirp's frequency content
              spreads into a thick horizontal blur.
   • W=64   → Δt=64 ms, Δf=15.6 Hz. Reasonable compromise.
   • W=512  → Δt=512 ms, Δf=1.95 Hz. The chirp's slope is crisp
              but the transient disappears (smeared across half a
              second).
   No window is "right" — the bound Δt·Δf ≥ 1/(4π) is binding.

2. Two transients:
   STFT at W=16 or W=64 separates the two spikes cleanly. At W=512
   they merge into a single broad smear (Δt > 1.5−1.0 = 0.5 s).
   DWT D1 lives in the highest frequency band (≈250–500 Hz) where
   the chirp has little energy but the transients (broadband)
   have plenty — so D1 shows TWO sharp peaks at exactly t=1.0 s
   and t=1.5 s, regardless of any "window choice". Multi-resolution
   tiling adapts automatically.

3. Asymptotic cost:
   For N = 2^20, FFT needs ~N·log₂N ≈ 2.1 × 10^7 ops; the Haar
   DWT needs ~2N ≈ 2.1 × 10^6 ops. That is a theoretical ~10×
   speedup (or ~20× if one ignores the factor of 2 in the
   geometric sum). In real code, constant factors and SIMD make
   the practical ratio implementation-dependent.
─────────────────────────────────────────────────────────────────────
""")
