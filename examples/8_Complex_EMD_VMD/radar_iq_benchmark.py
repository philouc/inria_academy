"""
=======================================================================
 Complex-Valued Radar IQ Decomposition Benchmark
 ─────────────────────────────────────────────────────────────────────
 Compare 3 strategies on a synthetic CW-radar micro-Doppler signal
 (V. Chen-style model):

   Method 1 — REAL-PART-ONLY VMD       (naive, loses Doppler sign)
   Method 2 — AUGMENTED EMD            (Tanaka et al.: concat Re, rev(Im))
   Method 3 — SHIFT-AND-VMD (MCVMD)    (Hu et al. 2022 idea: shift the
                                        spectrum to be unilateral, apply
                                        standard VMD, shift back)

 Test signal (complex IQ at f_s = 800 Hz, duration 2 s):
   s(t) = A_body · exp(j 2π f_b t)              [body, +50 Hz Doppler]
        + A_rotor · exp(j (f_dmax / f_r) · sin(2π f_r t))
                                                 [blade, FM ±80 Hz @ 4 Hz]
        + complex AWGN (SNR ≈ 15 dB)

 Run:        python radar_iq_benchmark.py
 Requires:   numpy, scipy, matplotlib, EMD-signal (PyEMD), vmdpy

 Author: Philippe Ciuciu
 Date:   2026-05
 Target: UnseenLabs / Inria Academy
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import stft, hilbert
from PyEMD import EMD
from vmdpy import VMD

np.random.seed(42)

# ── 1. Synthetic CW-radar IQ signal ───────────────────────────────────
fs = 800.0
T  = 2.0
t  = np.arange(0, T, 1/fs)
N  = len(t)

# Body return — constant complex tone (target approaching at +50 Hz Doppler)
f_body, A_body = 50.0, 0.6
body = A_body * np.exp(1j * 2*np.pi * f_body * t)

# Rotor blade — sinusoidal FM, swings between -fd_max and +fd_max
# Phase: φ(t) = (f_dmax / f_r) · sin(2π f_r t)
# This is V. Chen's textbook micro-Doppler model
fd_max, f_rotor, A_rotor = 80.0, 4.0, 1.0
blade = A_rotor * np.exp(1j * (fd_max/f_rotor) * np.sin(2*np.pi*f_rotor*t))

clean = body + blade

# Complex AWGN at fixed SNR
SNR_dB = 15.0
sig_pow = np.mean(np.abs(clean)**2)
n_pow   = sig_pow / 10**(SNR_dB/10)
noise   = np.sqrt(n_pow/2) * (np.random.randn(N) + 1j*np.random.randn(N))
s       = clean + noise

print(f"Signal: f_s = {fs} Hz, T = {T} s, N = {N} samples")
print(f"Body Doppler = +{f_body} Hz   ·   Blade FM ±{fd_max} Hz @ {f_rotor} Hz rotor")
print(f"SNR_in = {SNR_dB:.1f} dB")

# ── 2. Method 1: REAL-PART VMD ────────────────────────────────────────
# Treat the IQ signal as if it were real — keep only Re{s}.
# This loses the sign of the Doppler (real-spectrum is symmetric).
K, alpha = 2, 2000
modes_M1, _, _ = VMD(s.real, alpha, 0, K, 0, 1, 1e-7)
# modes_M1 are REAL signals — their STFT will be ±-symmetric

# ── 3. Method 2: AUGMENTED EMD (Tanaka et al.) ────────────────────────
# Concatenate Re(s) with the time-flipped Im(s) into one real signal,
# run EMD ONCE, then recover bivariate (complex) IMFs by splitting back.
aug = np.concatenate([s.real, s.imag[::-1]])
emd_imfs = EMD().emd(aug, max_imf=6)
n_imf = emd_imfs.shape[0]
biv_imfs = np.array([emd_imfs[k, :N] + 1j * emd_imfs[k, N:][::-1]
                     for k in range(n_imf)])
print(f"Method 2: Augmented EMD → {n_imf} bivariate IMFs")

# ── 4. Method 3: CHANNEL-WISE VMD (I and Q separately, recombine) ────
# Apply real VMD to I = Re{s} and Q = Im{s} INDEPENDENTLY, then pair
# the resulting modes by center-frequency and recombine as I_k + j Q_k.
# Because  cos(2π f t) + j sin(2π f t) = e^{j 2π f t},  this is enough
# to recover the SIGN of the Doppler — and is the simplest baseline
# of the "complex VMD family" found in the literature.

modes_I, omegas_I, _ = VMD(s.real, alpha, 0, K, 0, 1, 1e-7)
modes_Q, omegas_Q, _ = VMD(s.imag, alpha, 0, K, 0, 1, 1e-7)

# Match Q-modes to I-modes by closest centre frequency on the LAST iteration
omI = omegas_I[-1]; omQ = omegas_Q[-1]
order = [int(np.argmin(np.abs(omQ - w))) for w in omI]
modes_Q_aligned = modes_Q[order]

modes_M3 = modes_I + 1j * modes_Q_aligned        # complex modes
print(f"Method 3: Channel-wise VMD  ω_I = {omI*fs/(2*np.pi)}  →  matched to ω_Q")



# ── 5. Helpers for visualisation ──────────────────────────────────────
def stft_bilateral(x, fs, nperseg=160):
    """Bilateral STFT (full frequency axis, including negative)."""
    f, tt, Z = stft(x, fs=fs, nperseg=nperseg, return_onesided=False, boundary=None)
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    return f, tt, Z

def plot_stft(ax, x, fs, title, ylim=(-150, 150), vmax=None):
    f_, tt_, Z = stft_bilateral(x, fs)
    P = np.abs(Z)
    if vmax is None:
        vmax = P.max()
    ax.pcolormesh(tt_, f_, P, shading='gouraud', cmap='inferno', vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10.5, color='#1E2761', fontweight='bold', loc='left')
    ax.set_ylim(*ylim)
    ax.set_ylabel("freq (Hz)", fontsize=9)


# ── 6. Figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 10), constrained_layout=False)
gs  = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.18,
               left=0.07, right=0.97, top=0.94, bottom=0.06)

# Row 1: ground truth (clean) + observed noisy
plot_stft(fig.add_subplot(gs[0, 0]), clean, fs,
          "(a) Ground truth |STFT|  —  body @ +50 Hz  +  blade FM ±80 Hz")
plot_stft(fig.add_subplot(gs[0, 1]), s, fs,
          f"(b) Observed |STFT| with complex AWGN  (SNR = {SNR_dB:.0f} dB)")

# Row 2: Method 1 — Real VMD on Re{s}
plot_stft(fig.add_subplot(gs[1, 0]), modes_M1[0], fs,
          "(c)  M1  Real VMD on Re{s} — mode 1  (note ±-symmetric: SIGN LOST)")
plot_stft(fig.add_subplot(gs[1, 1]), modes_M1[1], fs,
          "(d)  M1  Real VMD on Re{s} — mode 2  (also ±-symmetric)")

# Row 3: Method 2 — Augmented EMD (show the two strongest bivariate IMFs)
# Pick the two IMFs with the highest energy
energies = (np.abs(biv_imfs)**2).sum(axis=1)
top2 = np.argsort(energies)[::-1][:2]
plot_stft(fig.add_subplot(gs[2, 0]), biv_imfs[top2[0]], fs,
          f"(e)  M2  Augmented EMD — bivariate IMF{top2[0]+1}  (signed Doppler preserved)")
plot_stft(fig.add_subplot(gs[2, 1]), biv_imfs[top2[1]], fs,
          f"(f)  M2  Augmented EMD — bivariate IMF{top2[1]+1}")

# Row 4: Method 3 — Channel-wise VMD (I & Q separately)
plot_stft(fig.add_subplot(gs[3, 0]), modes_M3[0], fs,
          "(g)  M3  Channel-wise VMD (I, Q sep.) — complex mode 1  (signed Doppler)")
plot_stft(fig.add_subplot(gs[3, 1]), modes_M3[1], fs,
          "(h)  M3  Channel-wise VMD — complex mode 2")

# x-labels on bottom row
for ax in fig.axes[-2:]:
    ax.set_xlabel("time (s)", fontsize=9)

fig.suptitle("CW-radar IQ decomposition — three approaches compared",
             fontsize=13, fontweight='bold', y=0.985)

import os
os.makedirs('/mnt/user-data/outputs', exist_ok=True)
out = '/mnt/user-data/outputs/radar_iq_benchmark.png'
plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved → {out}")
