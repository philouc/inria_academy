"""
=======================================================================
 Wavelet Theory — Script 5: Wavelet Denoising  (Symlet 4 version)
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


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                                                                      ║
# ║   EXERCISE SOLUTIONS                                                 ║
# ║                                                                      ║
# ╚═════════════════════════════════════════════════════════════════════╝
np.random.seed(0)   # reset for reproducibility of the exercises below


# ── Exercise 1 — Wavelet family comparison ────────────────────────────
print("\n" + "═" * 68)
print(" EXERCISE 1 — Wavelet family comparison")
print("═" * 68)

families = ['haar', 'db2', 'db4', 'db8',
            'sym4', 'sym8',
            'coif1', 'coif2', 'coif4']
ex1 = []
for fam in families:
    w  = pywt.Wavelet(fam)
    cf = pywt.wavedec(noisy, fam, level=LEVEL, mode=MODE)
    sj = np.median(np.abs(cf[-1])) / 0.6745
    lj = sj * np.sqrt(2.0 * np.log(N))
    ct = [cf[0]] + [pywt.threshold(c, lj, 'soft') for c in cf[1:]]
    den = pywt.waverec(ct, fam, mode=MODE)[:N]
    sv  = snr_db(clean, den)
    ex1.append((fam, w.dec_len, w.vanishing_moments_psi, sv, den))
    print(f"  {fam:7s}  taps={w.dec_len:2d}  vmom={w.vanishing_moments_psi}  "
          f"SNR = {sv:5.2f} dB")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Exercise 1 — Wavelet family comparison "
             "(universal soft-threshold, same noisy signal)",
             fontsize=12, fontweight='bold')

# Bar chart
fams_x = [r[0] for r in ex1]; snrs = [r[3] for r in ex1]
cols   = ['#7f8c8d' if f == 'haar'
          else ('#2ecc71' if f == WAVELET else '#3498db') for f in fams_x]
bars   = axes[0].bar(fams_x, snrs, color=cols, alpha=0.85,
                     edgecolor='black', lw=0.6)
for b, v in zip(bars, snrs):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08,
                 f"{v:.2f}", ha='center', fontsize=8)
axes[0].axhline(snr_db(clean, noisy), color='red', ls='--', lw=1,
                label=f'noisy baseline ({snr_db(clean, noisy):.2f} dB)')
axes[0].set_ylabel('Output SNR (dB)'); axes[0].set_xlabel('Wavelet')
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis='y')
axes[0].tick_params(axis='x', rotation=30)

# Zoom on smooth pulse: top-3 families
zoom_pulse = (t > 0.10) & (t < 0.32)
axes[1].plot(t[zoom_pulse], clean[zoom_pulse], 'k', lw=1.6, alpha=0.5,
             label='Clean')
top3 = sorted(ex1, key=lambda r: -r[3])[:3]
for fam, _, _, sv, den in top3:
    axes[1].plot(t[zoom_pulse], den[zoom_pulse], lw=1.5,
                 label=f'{fam} ({sv:.2f} dB)')
axes[1].set_title('Zoom on smooth pulse — top-3 by SNR',
                  fontsize=10, fontweight='bold')
axes[1].set_xlabel('Time'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plot_ex1_wavelet_families.png', dpi=120)
plt.show()


# ── Exercise 2 — Level-dependent threshold λ_j ────────────────────────
print("\n" + "═" * 68)
print(" EXERCISE 2 — Level-dependent threshold  λ_j = σ_j · √(2 ln N_j)")
print("═" * 68)

def threshold_per_level(coeffs, kind='soft'):
    """λ_j with σ_j and N_j estimated separately at each detail level."""
    out, info = [coeffs[0]], []
    for c in coeffs[1:]:
        s_j = np.median(np.abs(c)) / 0.6745
        l_j = s_j * np.sqrt(2.0 * np.log(len(c)))
        info.append((s_j, l_j, len(c)))
        out.append(pywt.threshold(c, l_j, mode=kind))
    return out, info

coeffs_pl, info_pl = threshold_per_level(coeffs, 'soft')
den_pl = dwt_reconstruct(coeffs_pl, N)

print(f"  Universal threshold:                  λ = {lam_univ:.4f}    "
      f"→ SNR = {snr_db(clean, den_soft):5.2f} dB")
print("  Per-level thresholds (coeffs is [cA_L, cD_L, …, cD_1]):")
for j, (sj, lj, nj) in enumerate(info_pl):
    actual = LEVEL - j
    print(f"     D{actual} (n={nj:3d}):  σ_j = {sj:.4f}   λ_j = {lj:.4f}")
print(f"  Per-level (level-dependent λ_j):                  "
      f"→ SNR = {snr_db(clean, den_pl):5.2f} dB")


# ── Exercise 3 — SURE thresholding (per level) ────────────────────────
print("\n" + "═" * 68)
print(" EXERCISE 3 — SURE (Stein's Unbiased Risk Estimate)")
print("═" * 68)

def sure_threshold(c, sigma):
    """
    Optimal soft-threshold by minimising the SURE risk estimate.
       SURE(t; y) = Σ min(y_i², t²) - n + 2·#{|y_i| > t},   y = c/σ.
    Candidate thresholds = sorted |y_i| values, plus t = 0
    (which has SURE = n).
    """
    n = len(c)
    if n == 0 or sigma <= 0:
        return 0.0
    y  = c / sigma
    s  = np.sort(np.abs(y))
    cs = np.cumsum(s ** 2)
    k  = np.arange(n)
    sure = cs + (n - k - 1) * s ** 2 + 2 * (n - k - 1) - n
    if n <= np.min(sure):       # t = 0 (no thresholding) wins
        return 0.0
    return s[np.argmin(sure)] * sigma


def threshold_sure(coeffs):
    """Apply per-level SURE soft-threshold (level-wise MAD σ_j)."""
    out, info = [coeffs[0]], []
    for c in coeffs[1:]:
        s_j = np.median(np.abs(c)) / 0.6745
        l_j = sure_threshold(c, s_j) if s_j > 0 else 0.0
        info.append((s_j, l_j, len(c)))
        out.append(pywt.threshold(c, l_j, mode='soft'))
    return out, info

coeffs_sure, info_sure = threshold_sure(coeffs)
den_sure = dwt_reconstruct(coeffs_sure, N)

print("  SURE thresholds per level:")
for j, (sj, lj, nj) in enumerate(info_sure):
    actual = LEVEL - j
    print(f"     D{actual} (n={nj:3d}):  σ_j = {sj:.4f}   λ_SURE = {lj:.4f}")
print(f"  SURE soft-threshold:                              "
      f"→ SNR = {snr_db(clean, den_sure):5.2f} dB")
print("  (Donoho-Johnstone hybrid would fall back on the universal rule")
print("   when a level is 'too sparse'. Not implemented here.)")


# ── Combined plot for Exercises 2 & 3 ─────────────────────────────────
fig = plt.figure(figsize=(15, 8))
gs2 = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42)
fig.suptitle("Exercises 2 & 3 — Three thresholding rules with Symlet 4",
             fontsize=12, fontweight='bold')

# Thresholds per level
ax = fig.add_subplot(gs2[0, 0])
levels_idx     = np.arange(LEVEL)
lams_pl_v      = [v[1] for v in info_pl]
lams_sure_v    = [v[1] for v in info_sure]
ax.plot(levels_idx, [lam_univ] * LEVEL, 'o-', color='#3498db', lw=2,
        label='Universal (global λ)')
ax.plot(levels_idx, lams_pl_v,    's-', color='#e67e22', lw=2,
        label='Per-level λ_j')
ax.plot(levels_idx, lams_sure_v, '^-', color='#9b59b6', lw=2,
        label='SURE λ_j')
ax.set_xticks(levels_idx)
ax.set_xticklabels([f'D{LEVEL - j}' for j in levels_idx])
ax.set_xlabel('Detail level (D5 = coarsest)')
ax.set_ylabel('Threshold value')
ax.set_title('Threshold per level — three rules',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# SNR bar chart
ax = fig.add_subplot(gs2[0, 1])
methods   = ['Noisy', 'Universal', 'Per-level', 'SURE']
snr_vals  = [snr_db(clean, noisy),  snr_db(clean, den_soft),
             snr_db(clean, den_pl), snr_db(clean, den_sure)]
mcolors   = ['#e74c3c', '#3498db', '#e67e22', '#9b59b6']
bars = ax.bar(methods, snr_vals, color=mcolors, alpha=0.85,
              edgecolor='black', lw=0.5)
for b, v in zip(bars, snr_vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
            f"{v:.2f}", ha='center', fontsize=8)
ax.set_ylabel('SNR (dB)')
ax.set_title('Output SNR (sym4)', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# Fraction kept per level
ax = fig.add_subplot(gs2[0, 2])
def frac_kept(coeffs):
    return [np.mean(np.abs(c) > 1e-12) for c in coeffs[1:]]
fu, fp, fs = frac_kept(coeffs_soft), frac_kept(coeffs_pl), frac_kept(coeffs_sure)
xb, wb = np.arange(LEVEL), 0.27
ax.bar(xb - wb, fu, wb, color='#3498db', label='Universal', alpha=0.85)
ax.bar(xb,      fp, wb, color='#e67e22', label='Per-level', alpha=0.85)
ax.bar(xb + wb, fs, wb, color='#9b59b6', label='SURE',      alpha=0.85)
ax.set_xticks(xb)
ax.set_xticklabels([f'D{LEVEL - j}' for j in range(LEVEL)])
ax.set_xlabel('Detail level'); ax.set_ylabel('Fraction kept')
ax.set_title('Coefficients kept per level',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# Zoom: smooth pulse
ax = fig.add_subplot(gs2[1, 0])
ax.plot(t[zoom_pulse], clean[zoom_pulse],    'k',       lw=1.6, alpha=0.5,
        label='Clean')
ax.plot(t[zoom_pulse], den_soft[zoom_pulse], '#3498db', lw=1.5, label='Universal')
ax.plot(t[zoom_pulse], den_pl[zoom_pulse],   '#e67e22', lw=1.5, label='Per-level')
ax.plot(t[zoom_pulse], den_sure[zoom_pulse], '#9b59b6', lw=1.5, label='SURE')
ax.set_title('Zoom: smooth pulse', fontsize=10, fontweight='bold')
ax.set_xlabel('Time'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Zoom: narrow spike
ax = fig.add_subplot(gs2[1, 1])
zoom_spike = (t > 0.78) & (t < 0.92)
ax.plot(t[zoom_spike], clean[zoom_spike],    'k',       lw=1.6, alpha=0.5,
        label='Clean')
ax.plot(t[zoom_spike], den_soft[zoom_spike], '#3498db', lw=1.5, label='Universal')
ax.plot(t[zoom_spike], den_pl[zoom_spike],   '#e67e22', lw=1.5, label='Per-level')
ax.plot(t[zoom_spike], den_sure[zoom_spike], '#9b59b6', lw=1.5, label='SURE')
ax.set_title('Zoom: narrow spike', fontsize=10, fontweight='bold')
ax.set_xlabel('Time'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Full reconstruction
ax = fig.add_subplot(gs2[1, 2])
ax.plot(t, clean,    'k',       lw=1.2, alpha=0.45, label='Clean')
ax.plot(t, den_soft, '#3498db', lw=1.0, alpha=0.85, label='Universal')
ax.plot(t, den_pl,   '#e67e22', lw=1.0, alpha=0.85, label='Per-level')
ax.plot(t, den_sure, '#9b59b6', lw=1.0, alpha=0.85, label='SURE')
ax.set_xlabel('Time'); ax.set_title('Full reconstruction',
                                       fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot_ex23_threshold_rules.png', dpi=120)
plt.show()


# ── Exercise 4 — Long-memory (1/f^α) clean signal ─────────────────────
print("\n" + "═" * 68)
print(" EXERCISE 4 — Wavelet thresholding on a 1/f^α long-memory signal")
print("═" * 68)

def simulate_fgn(N, H, seed=None):
    """Fractional Gaussian Noise via spectral method.  PSD ∝ |f|^{-(2H+1)}."""
    if seed is not None:
        np.random.seed(seed)
    f    = np.fft.rfftfreq(N)[1:]
    psd  = f ** (-(2.0 * H + 1.0))
    phi  = np.random.uniform(0, 2 * np.pi, len(f))
    sp   = np.sqrt(psd / 2.0) * (np.cos(phi) + 1j * np.sin(phi))
    spec = np.zeros(N // 2 + 1, dtype=complex)
    spec[1:] = sp
    x = np.fft.irfft(spec, n=N)
    return (x - x.mean()) / x.std()

H_lm     = 0.9          # strong long memory (energy concentrated at coarse scales)
sigma_lm = NOISE_LEVEL  # same noise level as the spiky test signal
# Match the spiky-signal variance so both denoising tasks have the SAME
# input SNR — only the basis-vs-signal mismatch then differs.
clean_lm = simulate_fgn(N, H_lm, seed=42) * np.sqrt(np.var(clean))
noisy_lm = clean_lm + sigma_lm * np.random.randn(N)

c_lm        = pywt.wavedec(noisy_lm, WAVELET, level=LEVEL, mode=MODE)
c_clean_lm  = pywt.wavedec(clean_lm, WAVELET, level=LEVEL, mode=MODE)
sigma_lm_hat = np.median(np.abs(c_lm[-1])) / 0.6745
lam_lm      = sigma_lm_hat * np.sqrt(2.0 * np.log(N))
ct_lm       = [c_lm[0]] + [pywt.threshold(c, lam_lm, 'soft') for c in c_lm[1:]]
den_lm      = pywt.waverec(ct_lm, WAVELET, mode=MODE)[:N]

print(f"  Long-memory clean signal: fGn with H = {H_lm}")
print(f"  Noisy SNR:              {snr_db(clean_lm, noisy_lm):5.2f} dB")
print(f"  Denoised SNR (soft):    {snr_db(clean_lm, den_lm):5.2f} dB")
print(f"  ΔSNR gain comparison (at matched input variance):")
print(f"     Spiky test signal:   {snr_db(clean, den_soft) - snr_db(clean, noisy):+5.2f} dB")
print(f"     1/f signal (this):   {snr_db(clean_lm, den_lm) - snr_db(clean_lm, noisy_lm):+5.2f} dB")
print(f"  Counter-intuitive: the 1/f gain looks LARGER, but see panel 3 —")
print(f"  the threshold is essentially low-pass filtering and ~kills the")
print(f"  fine-scale signal entirely (>50% energy lost at D1).")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle(f"Exercise 4 — Wavelet denoising on a 1/f^α signal  "
             f"(fGn, H = {H_lm}, σ_noise = {sigma_lm})",
             fontsize=12, fontweight='bold')

# Signals
axes[0].plot(t, clean_lm, 'steelblue', lw=1.0, label='Clean fGn')
axes[0].plot(t, noisy_lm, '#e74c3c',   lw=0.4, alpha=0.5, label='Noisy')
axes[0].plot(t, den_lm,   '#2ecc71',   lw=1.2, label='Soft-threshold')
axes[0].set_title('Signals', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Amplitude')
axes[0].legend(fontsize=8)

# Per-level variance: clean signal vs noise floor (the smoking gun)
clean_var = [np.var(c) for c in c_clean_lm[1:]]
x_p       = np.arange(LEVEL)
labels    = [f'D{LEVEL - j}' for j in range(LEVEL)]
axes[1].semilogy(x_p, clean_var, 'o-', color='steelblue', lw=2, ms=8,
                 label='Var(clean) per level')
axes[1].semilogy(x_p, [sigma_lm ** 2] * LEVEL, 's--', color='#e74c3c',
                 lw=2, ms=8, label=f'Noise σ² = {sigma_lm**2:.3f}')
axes[1].set_xticks(x_p); axes[1].set_xticklabels(labels)
axes[1].set_xlabel('Detail level (D5 coarsest → D1 finest)')
axes[1].set_ylabel('Variance (log scale)')
axes[1].set_title('Per-level variance: clean fGn vs noise floor\n'
                  '(curves meet at fine scales = bad for thresholding)',
                  fontsize=10, fontweight='bold')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, which='both')

# What the threshold actually destroys: clean energy at killed positions
clean_lost = []
for c_clean_j, c_noisy_j in zip(c_clean_lm[1:], c_lm[1:]):
    killed = np.abs(c_noisy_j) <= lam_lm
    clean_lost.append(np.sum(c_clean_j[killed] ** 2)
                      / (np.sum(c_clean_j ** 2) + 1e-15))
axes[2].bar(x_p, clean_lost, color='#9b59b6', alpha=0.85,
            edgecolor='black', lw=0.5)
for i, v in enumerate(clean_lost):
    axes[2].text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=8)
axes[2].set_xticks(x_p); axes[2].set_xticklabels(labels)
axes[2].set_xlabel('Detail level'); axes[2].set_ylabel('Fraction of clean energy CUT')
axes[2].set_ylim(0, 1.05)
axes[2].set_title('Clean-signal energy destroyed by the threshold\n'
                  '(very high at fine scales)',
                  fontsize=10, fontweight='bold')
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plot_ex4_long_memory.png', dpi=120)
plt.show()


# ── Closing commentary ────────────────────────────────────────────────
print(f"""
─── COMMENTARY ──────────────────────────────────────────────────────
Ex 1 — Wavelet family
   Ranking on this signal: sym8 (3.82) > coif1 ≈ sym4 ≈ db2 (≈ 3.0)
   > Haar (2.14) > db4 ≈ coif4 ≈ db8 (≈ 2.0).
   Two competing effects: (a) more vanishing moments → better
   sparsity on smooth parts, (b) longer support → more edge leakage
   near transients.  sym8 hits a sweet spot — 8 vanishing moments
   AND near-symmetric.  Daubechies db8 has the same support as sym8
   but is markedly asymmetric, hence the phase distortion on the
   smooth pulse and the worst SNR here.  Haar is decent on the
   spike but staircases the smooth pulse, ending mid-pack.

Ex 2 — Level-dependent threshold
   The MAD estimator is robust (50 % breakdown) but NOT immune to
   signal energy concentrated in a single level.  In this run the
   coarse-level σ_j is INFLATED by the signal's pulse-and-burst
   content (e.g. σ_{{D4}} = 0.72 vs the true noise 0.35), so the
   per-level rule actually OVER-thresholds at coarse levels and
   ends up slightly below the universal SNR.  Lesson: blind
   level-wise MAD only helps if the signal is sparse at every
   level.  A safer compromise is a single σ (from the finest
   level) with N_j-dependent log term: λ_j = σ_fine √(2 ln N_j).

Ex 3 — SURE
   SURE adapts to the local sparsity of each level.  On levels
   where the signal is concentrated, λ_SURE is small (don't cut
   the signal); on levels dominated by noise, λ_SURE ≈ universal.
   Typical winner on mixed smooth-+-transient content like ours.
   Caveat: pure SURE can be unstable when a level is too sparse;
   the standard fix is the Donoho-Johnstone hybrid (universal as
   a fallback when Σ y² ≤ n + (log₂ n)^{{3/2}} / √n).

Ex 4 — 1/f^α clean signal  →  the limit of wavelet denoising
   Wavelet thresholding rests on a sparsity assumption: the *clean*
   signal should have few large coefficients and many small ones,
   so a threshold can cleanly separate signal from white noise.
   A long-memory (1/f^α) signal is the opposite — its detail
   variance Var[D_j] ∝ 2^{{(2H+1)j}} DECAYS rapidly toward fine
   scales, so the signal energy is essentially concentrated at the
   coarsest levels.
   Counter-intuitive observation: the raw SNR gain looks GREATER
   than on the spiky signal.  That is misleading — what the
   threshold actually does on a 1/f signal is *low-pass filtering*:
   it keeps the coarse-scale energy and kills everything finer,
   noise and signal alike.  Panel 3 confirms the damage: typically
   > 50 % of the clean energy at D1 is destroyed.
   This bias is invisible if you only care about ||x̂ − x||², but
   it corrupts every downstream task that depends on the fine-scale
   structure: re-estimating H, multifractal analysis, spectral
   slope, scale-wise variance, etc.
   Better tools for 1/f signals:
     • level-wise *Bayesian* shrinkage that uses Var[D_j] of the
       signal class as a prior;
     • truncating the decomposition at a finite scale rather than
       thresholding all the way down.
─────────────────────────────────────────────────────────────────────
""")
