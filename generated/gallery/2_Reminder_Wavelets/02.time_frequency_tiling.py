"""
=======================================================================
 Wavelet Theory — Script 2: Time-Frequency Tiling & Heisenberg Principle
=======================================================================
 Topics covered:
   • STFT (Short-Time Fourier Transform) — uniform tiling
   • Discrete Wavelet Transform (DWT) — multi-resolution tiling
   • Heisenberg uncertainty: Δt · Δω ≥ 1/2
   • Why wavelets adapt resolution to the signal's content
=======================================================================
 Run:  python 02.time_frequency_tiling.py
 Requires: numpy, matplotlib, scipy
=======================================================================

Author: Philippe Ciuciu
Date: 04/08/2026
Target: UnseenLabs
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import stft

# ── 1. STFT — uniform Heisenberg boxes ────────────────────────────────
def demo_stft(sig, fs, win_sizes, title="STFT comparison"):
    """Show how window size affects the STFT time-frequency trade-off."""
    fig, axes = plt.subplots(1, len(win_sizes) + 1, figsize=(14, 4))

    # Plot signal
    t = np.arange(len(sig)) / fs
    axes[0].plot(t, sig, 'steelblue', lw=1)
    axes[0].set_title("Signal", fontsize=10)
    axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Amplitude")

    for ax, W in zip(axes[1:], win_sizes):
        f, t_s, Zxx = stft(sig, fs=fs, nperseg=W, noverlap=W * 3 // 4)
        ax.pcolormesh(t_s, f, np.abs(Zxx), shading='gouraud', cmap='inferno')
        ax.set_ylim(0, fs / 2)
        ax.set_title(f"STFT  window={W}\nΔt={W/fs:.3f}s  Δf={fs/W:.1f}Hz",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Freq (Hz)")

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


# Chirp + transient test signal
fs     = 1000
T      = 2.0
t_arr  = np.arange(int(T * fs)) / fs
sig    = (np.sin(2 * np.pi * (10 + 80 * t_arr) * t_arr)    # chirp
        + 2.5 * np.exp(-200 * (t_arr - 1.0)**2))            # Gaussian transient

fig = demo_stft(sig, fs, win_sizes=[32, 128, 512],
                title="STFT: rigid trade-off between time & frequency resolution")
fig.savefig('plot2a_stft_comparison.png', dpi=120)
plt.show()
print("Saved: plot2a_stft_comparison.png")


# ── 2. DWT — Haar multi-resolution decomposition ──────────────────────
def haar_dwt_levels(x, n_levels):
    """
    Iterative Haar DWT.
    Returns list of detail coefficients [d1, d2, ..., dn]
    and the final approximation.
    Details d1 are finest (high frequency), dn are coarsest (low frequency).
    """
    details = []
    a = np.array(x, dtype=float)
    for _ in range(n_levels):
        N  = (len(a) // 2) * 2
        lo = (a[:N:2] + a[1:N:2]) / np.sqrt(2)   # approximation
        hi = (a[:N:2] - a[1:N:2]) / np.sqrt(2)   # detail
        details.append(hi)
        a = lo
    return details, a  # (finest→coarsest details, final approx)


# Build a piecewise signal with components at different scales
N = 512
t_d = np.linspace(0, 1, N)
x   = (np.sin(2 * np.pi * 4   * t_d)                   # low-freq
     + 0.5 * np.sin(2 * np.pi * 32  * t_d)              # mid-freq
     + 0.3 * np.sin(2 * np.pi * 128 * t_d)              # high-freq
     + 1.5 * np.exp(-300 * (t_d - 0.3)**2))             # transient

n_lev  = 5
details, approx = haar_dwt_levels(x, n_lev)

fig, axes = plt.subplots(n_lev + 2, 1, figsize=(13, 10), sharex=False)
fig.suptitle("Haar DWT — Multi-Resolution Decomposition\n"
             "Each level captures a different frequency band",
             fontsize=12, fontweight='bold')

axes[0].plot(t_d, x, 'steelblue', lw=1.2)
axes[0].set_title("Original signal", fontsize=9); axes[0].set_ylabel("Amp")

clr = ['#e05c5c', '#e09c2a', '#27ae60', '#2980b9', '#8e44ad']
for i, (d, c) in enumerate(zip(details, clr)):
    t_lev = np.linspace(0, 1, len(d))
    axes[i + 1].plot(t_lev, d, color=c, lw=1.2)
    freq_lo = fs / (2 ** (i + 2))
    freq_hi = fs / (2 ** (i + 1))
    axes[i + 1].set_title(
        f"Detail D{i+1}  (freq band ≈ [{freq_lo:.0f}–{freq_hi:.0f}] Hz)", fontsize=9)
    axes[i + 1].set_ylabel("Coeff")

t_app = np.linspace(0, 1, len(approx))
axes[-1].plot(t_app, approx, color='#2c3e50', lw=1.5)
axes[-1].set_title(f"Approximation A{n_lev}  (lowest frequencies)", fontsize=9)
axes[-1].set_xlabel("Time"); axes[-1].set_ylabel("Amp")

plt.tight_layout()
plt.savefig('plot2b_haar_dwt.png', dpi=120)
plt.show()
print("Saved: plot2b_haar_dwt.png")


# ── 3. Heisenberg boxes: STFT vs DWT ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Heisenberg Boxes in the Time-Frequency Plane", fontsize=12, fontweight='bold')

# STFT — uniform grid
ax = axes[0]
ax.set_title("STFT — Uniform Tiling\n(fixed Δt and Δf)", fontsize=10)
ax.set_xlabel("Time"); ax.set_ylabel("Frequency")
colors6 = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']
ci = 0
for t0 in np.arange(0, 8, 2):
    for f0 in np.arange(0, 5, 1):
        ax.add_patch(mpatches.FancyBboxPatch(
            [t0 + 0.05, f0 + 0.05], 1.9, 0.9,
            boxstyle="round,pad=0.05", alpha=0.40,
            facecolor=colors6[ci % 6], edgecolor=colors6[ci % 6], linewidth=1.3))
        ci += 1
ax.set_xlim(0, 8); ax.set_ylim(0, 5)
ax.text(4, -0.5, "Δt fixed, Δf fixed — poor resolution in both dimensions",
        ha='center', fontsize=9, color='#c0392b', style='italic')

# DWT — dyadic tiling
ax = axes[1]
ax.set_title("Wavelet DWT — Dyadic Multi-Resolution Tiling\n"
             "(fine Δt at high freq, fine Δf at low freq)", fontsize=10)
ax.set_xlabel("Time"); ax.set_ylabel("Frequency (scale)")
# High freq: narrow time windows (many small boxes)
for t0 in np.arange(0, 8, 0.5):
    ax.add_patch(mpatches.FancyBboxPatch(
        [t0 + 0.02, 3.55], 0.46, 1.4,
        boxstyle="round,pad=0.02", facecolor='#e74c3c', alpha=0.40,
        edgecolor='#e74c3c', linewidth=1))
# Mid freq
for t0 in np.arange(0, 8, 1):
    ax.add_patch(mpatches.FancyBboxPatch(
        [t0 + 0.04, 1.55], 0.92, 1.9,
        boxstyle="round,pad=0.02", facecolor='#f39c12', alpha=0.40,
        edgecolor='#f39c12', linewidth=1))
# Low freq: wide time windows
for t0 in np.arange(0, 8, 2):
    ax.add_patch(mpatches.FancyBboxPatch(
        [t0 + 0.06, 0.06], 1.88, 1.45,
        boxstyle="round,pad=0.02", facecolor='#2ecc71', alpha=0.40,
        edgecolor='#2ecc71', linewidth=1))
ax.set_xlim(0, 8); ax.set_ylim(0, 5)
ax.text(0.2, 4.4, "High freq: fine time, coarse freq",  fontsize=8, color='#e74c3c')
ax.text(0.2, 2.5, "Mid freq",  fontsize=8, color='#f39c12')
ax.text(0.2, 0.8, "Low freq: coarse time, fine freq",   fontsize=8, color='#2ecc71')
ax.text(4, -0.5, "Heisenberg bound Δt·Δω = 1/2 at every level",
        ha='center', fontsize=9, color='#27ae60', style='italic')

plt.tight_layout()
plt.savefig('plot2c_heisenberg_boxes.png', dpi=120)
plt.show()
print("Saved: plot2c_heisenberg_boxes.png")

print("""
─── EXERCISE ────────────────────────────────────────────────────────
1. Modify the chirp + transient signal and try STFT window sizes of
   16, 64 and 512 samples. Observe the trade-off:
     small window  → good time resolution, poor frequency resolution
     large window  → good frequency resolution, poor time resolution

2. Add a second transient at t=1.5s. Does the STFT detect both?
   Does the DWT detail D1 (finest level) pick them up?

3. Compute the number of operations for DWT vs FFT:
     DWT: O(N)   |   FFT: O(N log N)
   For N=2^20, how much faster is DWT in theory?
─────────────────────────────────────────────────────────────────────
""")
