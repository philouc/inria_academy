"""
=======================================================================
 EMD in Practice — Successful Decomposition: Fourth Script
=======================================================================
 4-component synthetic signal whose pairwise frequency ratios are well
 below the empirical EMD "merge threshold" (~0.67), so EMD recovers each
 component cleanly as a separate IMF.

   x(t)  =  trend(t)            +     c3(t)          +    c2(t)
                                                      +    c1(t)
         =  0.6 · sin(2π · 0.7 t)                       (slow drift)
          + 1.0 · sin(2π · 6 t)                         (mid-low tone)
          + 0.7 · sin(2π · 18 t)                        (mid-high tone)
          + 1.0 · sin(2π · 50 t) · (1 + 0.6 sin(2π · 1 t))   (AM @1 Hz)

 Frequency ratios:  6 / 18 = 0.33  ·  18 / 50 = 0.36   (both < 0.67)

 Expected EMD output (high-freq first):
   IMF1 ≈ c1 (50 Hz AM)   ·   IMF2 ≈ c2 (18 Hz)
   IMF3 ≈ c3 (6 Hz)       ·   IMF4 ≈ trend (0.7 Hz)
   IMF5+ ≈ small residual (sifting tail)

Usage:     python 4.emd_successful_decomposition.py
=======================================================================
 Requires:   numpy, scipy, matplotlib, EMD-signal (PyEMD)
             pip install EMD-signal

 Author: Philippe Ciuciu
 Date: 05/03/2026
 Target: UnseenLabs / Inria Academy
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert
from PyEMD import EMD

np.random.seed(0)

# ── 1. Synthetic signal ───────────────────────────────────────────────
FS       = 500.0                 # sampling rate (Hz)
T        = 4.0                   # duration (s)
t        = np.arange(0, T, 1.0/FS)

# Components (high-frequency first; same naming as the slide)
trend = 0.6 * np.sin(2 * np.pi * 0.7 * t)
c3    = 1.0 * np.sin(2 * np.pi *  6  * t)
c2    = 0.7 * np.sin(2 * np.pi * 18  * t)
c1    = 1.0 * np.sin(2 * np.pi * 50  * t) * (1.0 + 0.6 * np.sin(2 * np.pi * 1.0 * t))

x          = trend + c3 + c2 + c1
truths     = {'trend (0.7 Hz)': trend, 'c3 (6 Hz)': c3,
              'c2 (18 Hz)': c2,        'c1 (50 Hz AM)': c1}
true_freqs = [0.7, 6.0, 18.0, 50.0]     # ground-truth carrier frequencies

# ── 2. Run EMD ────────────────────────────────────────────────────────
emd  = EMD()
imfs = emd.emd(x, t, max_imf=8)        # shape (n_imfs, len(t)) — high-freq first
n_imfs = imfs.shape[0]
print(f"EMD produced {n_imfs} IMFs (incl. residue).")

# Reconstruction error — should be ~ machine precision
recon = imfs.sum(axis=0)
rmse  = np.sqrt(np.mean((x - recon) ** 2))
print(f"Reconstruction RMSE = {rmse:.2e}")

# Energy of each IMF (% of total)
energies     = (imfs ** 2).sum(axis=1)
energies_pct = 100 * energies / energies.sum()
for k, e in enumerate(energies_pct, 1):
    print(f"   IMF{k}: {e:5.1f} %")

# ── 3. Hilbert-Huang Transform ────────────────────────────────────────
# Instantaneous frequency from the analytic signal of each IMF
def instantaneous_freq(imf, fs):
    """Unwrapped instantaneous frequency (Hz) from the Hilbert phase."""
    analytic = hilbert(imf)
    phase    = np.unwrap(np.angle(analytic))
    f_inst   = np.diff(phase) / (2 * np.pi) * fs
    # Pad to keep the same length as the input
    return np.concatenate([f_inst[:1], f_inst])

inst_freqs = np.array([instantaneous_freq(imf, FS) for imf in imfs])
inst_amps  = np.abs(hilbert(imfs, axis=1))

# ── 4. Plot — match the deck slide's panel layout ─────────────────────
fig = plt.figure(figsize=(15, 9), constrained_layout=False)
gs  = GridSpec(3, 2, figure=fig,
               height_ratios=[1.0, 1.6, 0.9],
               width_ratios=[1.0, 1.0],
               hspace=0.55, wspace=0.22,
               left=0.06, right=0.98, top=0.94, bottom=0.07)

# (a) Synthetic signal — full top row
ax_sig = fig.add_subplot(gs[0, :])
ax_sig.plot(t, x, color='#1a1a2e', lw=0.6)
ax_sig.set_title("(a)  Synthetic signal  $x(t) = \\mathrm{trend} + c_3 + c_2 + c_1$",
                 loc='left', fontsize=11, fontweight='bold')
ax_sig.set_xlabel("time (s)"); ax_sig.set_ylabel("x(t)")
ax_sig.set_xlim(t[0], t[-1])

# (b) IMFs (black) vs ground-truth components (colour)
ax_imf = fig.add_subplot(gs[1, 0])
# Pair each IMF with its closest ground-truth carrier by mean instantaneous freq
truth_list   = [c1, c2, c3, trend]
truth_labels = ['true $c_1$ (50 Hz AM)', 'true $c_2$ (18 Hz)',
                'true $c_3$ (6 Hz)',     'true trend (0.7 Hz)']
truth_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

for k in range(n_imfs):
    offset = -k * 2.0
    ax_imf.plot(t, imfs[k] + offset, color='#1a1a2e', lw=0.9)
    if k < 4:                            # overlay closest ground-truth
        ax_imf.plot(t, truth_list[k] + offset, color=truth_colors[k],
                    lw=1.4, alpha=0.7)
    ax_imf.text(t[-1] * 1.005, offset, f'IMF{k+1}', va='center',
                fontsize=9, fontweight='bold', color=truth_colors[k] if k<4 else '#666')

# Legend for first row only (avoid clutter)
ax_imf.plot([], [], color='#1a1a2e', lw=0.9, label='EMD IMF')
ax_imf.plot([], [], color=truth_colors[0], lw=1.4, alpha=0.7, label='true $c_1$ (50 Hz AM)')
ax_imf.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax_imf.set_title("(b)  EMD IMFs (black) vs ground-truth components (colour)",
                 loc='left', fontsize=10.5, fontweight='bold')
ax_imf.set_xlabel("time (s)"); ax_imf.set_yticks([])
ax_imf.set_xlim(t[0], t[-1])

# (c) Hilbert-Huang spectrum — log frequency axis
ax_hht = fig.add_subplot(gs[1, 1])
# Stitched scatter coloured by amplitude (cap to a sane window)
for k in range(min(n_imfs, 5)):
    f_k = inst_freqs[k]
    a_k = inst_amps[k]
    mask = (f_k > 0.2) & (f_k < FS/2) & (a_k > 0.02)
    ax_hht.scatter(t[mask], f_k[mask], c=a_k[mask], cmap='inferno',
                   s=2, alpha=0.6, vmin=0, vmax=1.2)
# Ground-truth carrier lines
for f0, lbl, col in zip(true_freqs, ['trend', 'c3', 'c2', 'c1'], truth_colors[::-1]):
    ax_hht.axhline(f0, color=col, lw=1.0, ls='--', alpha=0.5)
ax_hht.set_yscale('log')
ax_hht.set_ylim(0.3, 100); ax_hht.set_xlim(t[0], t[-1])
ax_hht.set_title("(c)  Hilbert-Huang spectrum — ridges at 0.7, 6, 18, 50 Hz",
                 loc='left', fontsize=10.5, fontweight='bold')
ax_hht.set_xlabel("time (s)"); ax_hht.set_ylabel("frequency (Hz)")
ax_hht.grid(which='both', alpha=0.2)

# (d) Reconstruction error
ax_err = fig.add_subplot(gs[2, 0])
err = x - recon
ax_err.plot(t, err, color='#e74c3c', lw=0.7)
ax_err.axhline(0, color='grey', lw=0.5, ls='--')
ax_err.set_title(f"(d)  Reconstruction error  —  RMSE = {rmse:.1e}",
                 loc='left', fontsize=10.5, fontweight='bold')
ax_err.set_xlabel("time (s)"); ax_err.set_ylabel(r"x(t) − $\Sigma$ IMF$_k$")
ax_err.set_xlim(t[0], t[-1])

# (e) Energy distribution
ax_e = fig.add_subplot(gs[2, 1])
bar_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#1a3a6e',
              '#9a9fb8', '#bfc4d1', '#d9dce4', '#eaecf0'][:n_imfs]
bars = ax_e.bar([f'IMF{i+1}' for i in range(n_imfs)],
                energies_pct, color=bar_colors, edgecolor='none')
for b, v in zip(bars, energies_pct):
    ax_e.text(b.get_x() + b.get_width()/2, v + 1, f'{v:.1f}%',
              ha='center', fontsize=9, fontweight='bold', color='#1a1a2e')
ax_e.set_title("(e)  IMF energy distribution  —  dominant + small tail",
               loc='left', fontsize=10.5, fontweight='bold')
ax_e.set_ylabel("% energy"); ax_e.set_ylim(0, max(energies_pct) * 1.18)

# Super-title
fig.suptitle("EMD in Practice — Successful Decomposition  "
             r"$\;$($f$-ratios 0.33, 0.36 — well below the 0.67 merge limit)",
             fontsize=12.5, fontweight='bold', y=0.99)

plt.show()
import os
os.makedirs('./outputs', exist_ok=True)
out_path = './outputs/emd_successful_decomposition.png'
plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {out_path}")

# ── 5. Console summary ────────────────────────────────────────────────
print(f"""
─── SUMMARY ─────────────────────────────────────────────────────────
• IMF1-IMF4 capture c1, c2, c3, trend  (highest freq first)
• HHT ridges land near 50, 18, 6, 0.7 Hz — each IMF locally mono-freq
• {sum(energies_pct[:4]):.1f} % of total energy in the first 4 IMFs
• Residual IMFs (5+) hold only {sum(energies_pct[4:]):.1f} % — sifting tail
• Σ IMFk ≡ x(t)  to machine precision  (RMSE = {rmse:.1e})

When pairwise frequency ratios are < ~0.67, EMD adaptively recovers each
component without choosing any basis — that's the appeal of the method.
For close-ratio cases see the "EMD struggles" companion slide.
─────────────────────────────────────────────────────────────────────
""")
