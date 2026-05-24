"""
=======================================================================
 Wavelet Theory — Script 4: Wavelet Denoising  (Symlet 4 version)
=======================================================================
 Topics covered:
   • Symlet-4 DWT filter bank via PyWavelets  (smoother than Haar:
     8 taps, 4 vanishing moments, near-symmetric — ideal for smooth
     signals where Haar's box-like reconstruction creates staircase
     artefacts).
   • Soft vs hard thresholding of detail coefficients.
   • Universal threshold (Donoho & Johnstone 1994):  λ = σ √(2 ln N)
     with σ estimated by the MAD of the finest detail level.
   • Periodic boundary handling so the length round-trips exactly.
   • SNR comparison across methods + visual reconstruction +
     level-by-level energy analysis + sparsity histogram +
     direct comparison Symlet 4 vs Haar to motivate the choice.
=======================================================================
 Run:  python 05.wavelet_denoising.py
 Requires: numpy, matplotlib, PyWavelets
   pip install PyWavelets


 Author: Philippe Ciuciu
 Date: 04/09/2026
 Target: UnseenLabs
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pywt

np.random.seed(0)

# ── Configuration ─────────────────────────────────────────────────────
WAVELET = 'sym4'          # smooth, near-symmetric, 4 vanishing moments
LEVEL   = 5
MODE    = 'periodization' # length-preserving boundary handling


# ── Helpers ───────────────────────────────────────────────────────────
def dwt_decompose(x, wavelet=WAVELET, level=LEVEL, mode=MODE):
    """Forward DWT.  Returns [cA_L, cD_L, …, cD_1] (coarse → fine).

    With mode='periodization' the total coefficient count equals len(x)
    so reconstruction round-trips with no length adjustment."""
    return pywt.wavedec(x, wavelet, level=level, mode=mode)


def dwt_reconstruct(coeffs, N_out, wavelet=WAVELET, mode=MODE):
    """Inverse DWT, truncated to N_out samples."""
    return pywt.waverec(coeffs, wavelet, mode=mode)[:N_out]


def threshold_coeffs(coeffs, lam, kind='soft'):
    """Apply soft or hard thresholding to detail levels only.
    The approximation coeffs[0] is left untouched."""
    return [coeffs[0]] + [pywt.threshold(c, lam, mode=kind) for c in coeffs[1:]]


def universal_lambda(coeffs, N):
    """Donoho-Johnstone universal threshold.
    σ is estimated robustly from the finest detail via the MAD."""
    finest = coeffs[-1]
    sigma  = np.median(np.abs(finest)) / 0.6745
    return sigma, sigma * np.sqrt(2.0 * np.log(N))


def snr_db(clean, estimated):
    e = np.asarray(estimated)[:len(clean)]
    num = np.sum(clean ** 2)
    den = np.sum((clean - e) ** 2) + 1e-15
    return 10.0 * np.log10(num / den)


# ── Test signal ───────────────────────────────────────────────────────
# Three localised features: smooth pulse, mid-frequency burst, narrow spike.
# The smooth pulse is where Haar's blocky reconstruction visibly hurts.
N     = 512
t     = np.linspace(0, 1, N)
clean = (np.exp(-80  * (t - 0.20) ** 2) * np.sin(30 * np.pi * t)        # smooth pulse
       + 0.6 * np.exp(-50  * (t - 0.60) ** 2) * np.sin(80 * np.pi * t)  # mid burst
       + 0.4 * np.exp(-200 * (t - 0.85) ** 2))                          # narrow spike

NOISE_LEVEL = 0.35
noisy = clean + NOISE_LEVEL * np.random.randn(N)

print(f"Wavelet:        {WAVELET}  ({pywt.Wavelet(WAVELET).dec_len} taps, "
      f"{pywt.Wavelet(WAVELET).vanishing_moments_psi} vanishing moments)")
print(f"Decomp. levels: {LEVEL},  boundary mode: {MODE}")
print(f"Noisy signal SNR: {snr_db(clean, noisy):.2f} dB")


# ── Universal-threshold denoising (Symlet 4) ──────────────────────────
coeffs = dwt_decompose(noisy)
sigma_hat, lam_univ = universal_lambda(coeffs, N)

coeffs_soft = threshold_coeffs(coeffs, lam_univ, kind='soft')
coeffs_hard = threshold_coeffs(coeffs, lam_univ, kind='hard')
den_soft = dwt_reconstruct(coeffs_soft, N)
den_hard = dwt_reconstruct(coeffs_hard, N)

print(f"\nNoise σ (true):       {NOISE_LEVEL:.4f}")
print(f"Noise σ (MAD est.):   {sigma_hat:.4f}")
print(f"Universal threshold λ: {lam_univ:.4f}")
print(f"Soft threshold SNR:    {snr_db(clean, den_soft):.2f} dB")
print(f"Hard threshold SNR:    {snr_db(clean, den_hard):.2f} dB")


# ── Reference run with Haar for direct comparison ─────────────────────
coeffs_haar = pywt.wavedec(noisy, 'haar', level=LEVEL, mode=MODE)
sigma_haar  = np.median(np.abs(coeffs_haar[-1])) / 0.6745
lam_haar    = sigma_haar * np.sqrt(2.0 * np.log(N))
coeffs_haar_soft = ([coeffs_haar[0]] +
                    [pywt.threshold(c, lam_haar, mode='soft') for c in coeffs_haar[1:]])
den_haar = pywt.waverec(coeffs_haar_soft, 'haar', mode=MODE)[:N]
print(f"\nHaar (reference) soft-threshold SNR: {snr_db(clean, den_haar):.2f} dB")


# ── Full visual comparison ────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)
fig.suptitle(f"Wavelet Denoising — Symlet-4 DWT, Universal Threshold",
             fontsize=13, fontweight='bold')

# Row 1: signals
ax = fig.add_subplot(gs[0, 0])
ax.plot(t, clean, 'steelblue', lw=1.5)
ax.set_title("Clean signal", fontsize=10)
ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")

ax = fig.add_subplot(gs[0, 1])
ax.plot(t, noisy, color='#e74c3c', lw=0.7, alpha=0.85)
ax.set_title(f"Noisy signal  (SNR={snr_db(clean, noisy):.1f} dB)", fontsize=10)
ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")

ax = fig.add_subplot(gs[0, 2])
ax.plot(t, clean,    'steelblue', lw=1.0, alpha=0.4, label='Clean (ref)')
ax.plot(t, den_soft, '#2ecc71',   lw=1.8, label=f'sym4 soft ({snr_db(clean, den_soft):.1f} dB)')
ax.plot(t, den_hard, '#f39c12',   lw=1.2, alpha=0.75, ls='--',
        label=f'sym4 hard ({snr_db(clean, den_hard):.1f} dB)')
ax.legend(fontsize=8)
ax.set_title("Denoised signals (Symlet 4)", fontsize=10)
ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")

# Row 2 — Detail level 1 thresholding
ax = fig.add_subplot(gs[1, 0])
d_r = coeffs[-1]; d_s = coeffs_soft[-1]
xc = np.linspace(0, 1, len(d_r))
ax.bar(xc, d_r, width=1.0 / len(d_r) * 0.9, color='#e74c3c', alpha=0.6, label='Before')
ax.bar(xc, d_s, width=1.0 / len(d_r) * 0.9, color='#2ecc71', alpha=0.9, label='After (soft)')
ax.axhline( lam_univ, color='gold', lw=1.5, ls='--', label=f'±λ={lam_univ:.2f}')
ax.axhline(-lam_univ, color='gold', lw=1.5, ls='--')
ax.legend(fontsize=8)
ax.set_title("Detail D1 (finest scale) — before/after soft threshold", fontsize=9)
ax.set_xlabel("Position"); ax.set_ylabel("Coefficient value")

# Row 2 mid — All detail levels stacked
ax = fig.add_subplot(gs[1, 1])
clv = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
# coeffs is [cA_L, cD_L, …, cD_1].  Plot finest first (D1) at bottom.
detail_levels = coeffs[1:][::-1]            # D1 … D_L
detail_soft   = coeffs_soft[1:][::-1]
for k, (cr, cs, ck) in enumerate(zip(detail_levels, detail_soft, clv)):
    offset = k * 2.8
    xk = np.linspace(0, 1, len(cr))
    ax.plot(xk, cr + offset, color=ck, lw=0.7, alpha=0.4)
    ax.plot(xk, cs + offset, color=ck, lw=1.4, label=f'D{k + 1}')
ax.legend(fontsize=8, loc='upper right')
ax.set_title("All detail levels (D1 = finest)\nbefore → after soft threshold", fontsize=9)
ax.set_xlabel("Position"); ax.set_ylabel("Level + offset")

# Row 2 right — SNR vs threshold
ax = fig.add_subplot(gs[1, 2])
lam_grid = np.linspace(0.01, lam_univ * 2.5, 80)
snrs_s = [snr_db(clean,
                 dwt_reconstruct(threshold_coeffs(coeffs, lam, 'soft'), N))
          for lam in lam_grid]
snrs_h = [snr_db(clean,
                 dwt_reconstruct(threshold_coeffs(coeffs, lam, 'hard'), N))
          for lam in lam_grid]
ax.plot(lam_grid, snrs_s, '#2ecc71', lw=2, label='Soft threshold')
ax.plot(lam_grid, snrs_h, '#f39c12', lw=2, ls='--', label='Hard threshold')
ax.axvline(lam_univ, color='red', lw=1.5, ls=':', label=f'Universal λ={lam_univ:.2f}')
ax.axhline(snr_db(clean, noisy), color='grey', ls='--', lw=1, label='Noisy baseline')
ax.set_xlabel("Threshold λ"); ax.set_ylabel("SNR (dB)")
ax.set_title("SNR vs threshold — optimal λ trade-off", fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 3 — sparsity histogram
ax = fig.add_subplot(gs[2, 0])
all_det = np.concatenate([c.ravel() for c in coeffs[1:]])
ax.hist(all_det, bins=80, color='steelblue', alpha=0.75, edgecolor='none',
        label='Detail coeffs')
ax.axvline( lam_univ, color='red', lw=2, ls='--', label=f'±λ={lam_univ:.2f}')
ax.axvline(-lam_univ, color='red', lw=2, ls='--')
pct_kept = 100.0 * np.mean(np.abs(all_det) >= lam_univ)
ax.set_title(f"Sparsity (sym4): only {pct_kept:.1f}% of coefficients kept",
             fontsize=9)
ax.set_xlabel("Coefficient value"); ax.set_ylabel("Count")
ax.legend(fontsize=8)

# Row 3 mid — per-level energy before/after
ax = fig.add_subplot(gs[2, 1])
level_labels = [f'A{LEVEL}'] + [f'D{LEVEL - i}' for i in range(LEVEL)]   # cA, then coarse→fine
energies_noisy = [np.sum(c ** 2) for c in coeffs]
energies_soft  = [np.sum(c ** 2) for c in coeffs_soft]
x_pos = np.arange(len(level_labels))
ax.bar(x_pos - 0.2, energies_noisy, width=0.38, color='#e74c3c', alpha=0.8, label='Noisy')
ax.bar(x_pos + 0.2, energies_soft,  width=0.38, color='#2ecc71', alpha=0.8, label='Denoised (soft)')
ax.set_xticks(x_pos); ax.set_xticklabels(level_labels, fontsize=8)
ax.set_xlabel("DWT level (D1 = finest)"); ax.set_ylabel("Energy")
ax.set_title("Energy per DWT level: before vs after thresholding", fontsize=9)
ax.legend(fontsize=8)

# Row 3 right — Direct comparison Symlet 4 vs Haar on the smooth pulse zoom
ax = fig.add_subplot(gs[2, 2])
zoom = (t > 0.10) & (t < 0.32)         # the smooth pulse where Haar staircases
ax.plot(t[zoom], clean[zoom],     'steelblue', lw=1.5, alpha=0.45, label='Clean')
ax.plot(t[zoom], den_haar[zoom],  '#7f8c8d',   lw=1.4, ls='--',
        label=f'Haar soft  ({snr_db(clean, den_haar):.1f} dB)')
ax.plot(t[zoom], den_soft[zoom],  '#2ecc71',   lw=1.9,
        label=f'sym4 soft  ({snr_db(clean, den_soft):.1f} dB)')
ax.set_title("Zoom on smooth pulse:\nsym4 vs Haar reconstruction", fontsize=9)
ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.savefig('plot4_denoising.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nSaved: plot4_denoising.png")


# ── Notes & exercises ─────────────────────────────────────────────────
print(f"""
─── Why Symlet 4 over Haar ──────────────────────────────────────────
• Haar (2 taps, 1 vanishing moment) reconstructs as a piecewise-
  constant signal → visible staircase artefacts on smooth features.
• Symlet 4 (8 taps, 4 vanishing moments, near-symmetric) reproduces
  polynomials up to degree 3 exactly → no staircase, very mild phase
  distortion, well suited to smooth signals + transient mixtures.
• See the bottom-right zoom panel for a side-by-side example.

─── EXERCISE ────────────────────────────────────────────────────────
1. Re-run with WAVELET = 'db4', 'coif2', 'sym8'.  Which family gives
   the best SNR on this signal?  Why?

2. Try level-dependent thresholds:  λ_j = σ_j · √(2 ln N_j)
   where σ_j and N_j are estimated separately at each detail level.
   Compare with the universal threshold above.

3. Replace the universal rule with SURE (Stein's Unbiased Risk
   Estimate) thresholding (pywt.threshold ‘sureshrink’ via custom code
   or scikit-image).  Which wins on the smooth pulse vs the spike?

4. What happens if the clean signal itself has a 1/fᵅ power spectrum
   (long-memory)?  Wavelet thresholding then attacks part of the
   signal — link to slide 4 of the deck and the scale-wise variance
   discussion.
─────────────────────────────────────────────────────────────────────
""")
