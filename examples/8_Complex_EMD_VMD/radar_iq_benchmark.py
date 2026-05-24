"""
=======================================================================
 Complex-Valued IQ Decomposition Benchmark  (v2)
 ─────────────────────────────────────────────────────────────────────
 Four strategies compared on a synthetic complex signal with bilateral
 spectrum (mimicking IQ data with both incoming AND outgoing targets):

   M1 — Naive REAL VMD on Re{s}                  (loses sign of Doppler)
   M2 — Channel-wise VMD (I, Q independent)      (simplest complex-aware)
   M3 — Bivariate EMD (Rilling–Flandrin 2007)    (rotating envelopes)
   M4 — MCVMD (heterodyne trick)                 (upsample + shift + real VMD)

 Test signal  (length 2 s @ fs = 800 Hz):
   s(t) = exp(+j 2π·30 t) + 0.7 · exp(-j 2π·40 t) + complex AWGN
        = component A at +30 Hz  (approaching, amplitude 1.0)
        + component B at -40 Hz  (receding,    amplitude 0.7)

 Pedagogical point: a properly complex-aware decomposition recovers
 each tone with the correct SIGN of the Doppler shift; the naive
 real-only approach produces ±-symmetric mode spectra (sign lost).

 Author: Philippe Ciuciu  ·  Target: UnseenLabs / Inria Academy
=======================================================================
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, resample, find_peaks
from scipy.interpolate import CubicSpline
from vmdpy import VMD

# ── CLI ───────────────────────────────────────────────────────────────
# Default behaviour: pop up an interactive matplotlib window.
# Pass `-s out.png` (or `--save out.png`) to write to disk instead.
parser = argparse.ArgumentParser(
    description="Complex-IQ decomposition benchmark — 4 methods compared.",
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-s', '--save', metavar='PATH', default=None,
                    help='Save the figure to PATH (PNG) instead of displaying it.')
parser.add_argument('--dpi', type=int, default=140,
                    help='DPI for the saved figure (default: 140).')
parser.add_argument('--seed', type=int, default=42,
                    help='RNG seed for the additive noise (default: 42).')
args = parser.parse_args()

np.random.seed(args.seed)

# ── 1. Synthetic complex signal ───────────────────────────────────────
fs = 800.0; T = 2.0
t  = np.arange(0, T, 1/fs); N = len(t)

f_A, A_A = +30.0, 1.0          # Doppler +30 Hz (approaching)
f_B, A_B = -40.0, 0.7          # Doppler -40 Hz (receding)
clean = A_A*np.exp(1j*2*np.pi*f_A*t) + A_B*np.exp(1j*2*np.pi*f_B*t)

SNR_dB = 15.0
sig_pow = np.mean(np.abs(clean)**2)
n_pow   = sig_pow / 10**(SNR_dB/10)
noise   = np.sqrt(n_pow/2) * (np.random.randn(N) + 1j*np.random.randn(N))
s       = clean + noise
print(f"Signal: tones at +{f_A} Hz (A={A_A}) and {f_B} Hz (A={A_B}) · SNR = {SNR_dB:.0f} dB")

K, alpha = 2, 2000

# ── 2. M1: Real VMD on Re{s} ──────────────────────────────────────────
modes_M1, _, _ = VMD(s.real, alpha, 0, K, 0, 1, 1e-7)

# ── 3. M2: Channel-wise VMD ───────────────────────────────────────────
modes_I, om_I, _ = VMD(s.real, alpha, 0, K, 0, 1, 1e-7)
modes_Q, om_Q, _ = VMD(s.imag, alpha, 0, K, 0, 1, 1e-7)
order = [int(np.argmin(np.abs(om_Q[-1] - w))) for w in om_I[-1]]
modes_M2 = modes_I + 1j * modes_Q[order]

# ── 4. M3: Bivariate EMD (Rilling–Flandrin 2007) ──────────────────────
def envelope_complex(z, idx, t_):
    if len(idx) < 2:
        return np.zeros_like(z)
    cs_re = CubicSpline(idx, z[idx].real, extrapolate=True)
    cs_im = CubicSpline(idx, z[idx].imag, extrapolate=True)
    return cs_re(t_) + 1j*cs_im(t_)

def bemd_sift(z, n_directions=8, max_sift=30, tol=0.05, fs_=800.0, f_max=60.0):
    """Sift one bivariate IMF.  Mean envelope = avg over directions of
    (upper + lower)/2.  KEY: extrema spacing constrained to ≥ 1 carrier
    period (≈ fs/f_max samples) to avoid cubic-spline overshoot for
    densely-oscillating signals."""
    t_ = np.arange(len(z))
    h  = z.copy()
    min_dist = max(int(fs_ / f_max), 3)            # minimum samples between extrema
    for _ in range(max_sift):
        directions = np.linspace(0, 2*np.pi, n_directions, endpoint=False)
        mean_envs = []
        for phi in directions:
            proj = np.real(h * np.exp(-1j*phi))
            max_idx, _ = find_peaks(proj, distance=min_dist)
            min_idx, _ = find_peaks(-proj, distance=min_dist)
            if len(max_idx) < 4 or len(min_idx) < 4:
                continue
            env_upper = envelope_complex(h, max_idx, t_)
            env_lower = envelope_complex(h, min_idx, t_)
            mean_envs.append((env_upper + env_lower) / 2)
        if not mean_envs:
            return h
        mean_env = np.mean(mean_envs, axis=0)
        h_new    = h - mean_env
        delta    = np.sum(np.abs(h_new - h)**2) / (np.sum(np.abs(h)**2) + 1e-12)
        h        = h_new
        if delta < tol:
            break
    return h

def bemd(z, max_imf=4):
    imfs, r = [], z.copy()
    for _ in range(max_imf):
        imf = bemd_sift(r)
        if np.sum(np.abs(imf)**2) < 1e-8:
            break
        imfs.append(imf)
        r = r - imf
        if np.sum(np.abs(r)**2) < 1e-6 * np.sum(np.abs(z)**2):
            break
    return np.array(imfs)

modes_M3 = bemd(s, max_imf=3)
print(f"M3 (BEMD): {len(modes_M3)} bivariate IMFs")

# ── 5. M4: MCVMD (heterodyne trick: upsample + spectral shift + VMD) ──
def mcvmd(z, fs_, K, alpha=2000, tol=1e-7):
    """Upsample → shift +fs/2 → take Re → real VMD → Hilbert → shift back → downsample."""
    Nz = len(z)
    z_up = resample(z, 2*Nz)
    n_up = np.arange(2*Nz)
    shift_up = np.exp(+1j * np.pi/2 * n_up)
    z_shifted = z_up * shift_up
    modes_real, _, _ = VMD(z_shifted.real, alpha, 0, K, 0, 1, tol)
    modes_analytic = np.array([hilbert(m) for m in modes_real])
    modes_back = modes_analytic * np.conj(shift_up)
    return modes_back[:, ::2]

modes_M4 = mcvmd(s, fs, K, alpha)
print(f"M4 (MCVMD): {K} complex modes")

# ── 6. Plot — bilateral magnitude spectrum of each recovered mode ─────
def spec_db(z, fs_):
    Z = np.fft.fftshift(np.fft.fft(z))
    f = np.fft.fftshift(np.fft.fftfreq(len(z), 1/fs_))
    P = 20*np.log10(np.abs(Z) / np.abs(Z).max() + 1e-12)
    return f, P

fig = plt.figure(figsize=(14, 11))
gs  = GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.18,
               left=0.06, right=0.98, top=0.95, bottom=0.05)

def plot_spec(ax, z, fs_, title, color):
    f, P = spec_db(z, fs_)
    ax.plot(f, P, color=color, lw=1.4)
    for fe, lab in [(+30, '+30'), (-40, '-40')]:
        ax.axvline(fe, color='red', ls='--', lw=0.8, alpha=0.5)
        ax.text(fe, 5, lab, ha='center', fontsize=8.5, color='red', alpha=0.8)
    ax.set_xlim(-100, 100); ax.set_ylim(-50, 10)
    ax.set_title(title, fontsize=10.5, color='#1E2761',
                 fontweight='bold', loc='left')
    ax.grid(alpha=0.3)

# Row 1: ground truth + observed
plot_spec(fig.add_subplot(gs[0, 0]), clean, fs,
          "(a)  Ground truth  |FFT|² (dB)  —  two signed-Doppler tones", '#1a1a2e')
plot_spec(fig.add_subplot(gs[0, 1]), s, fs,
          f"(b)  Observed  (SNR = {SNR_dB:.0f} dB)", '#1a1a2e')

# Row 2: M1
for k in range(2):
    plot_spec(fig.add_subplot(gs[1, k]), modes_M1[k], fs,
              f"({'cd'[k]})  M1  Real VMD on ℜ{{s}} — mode {k+1}  "
              + ("(mirrored: SIGN LOST)" if k==0 else "(also mirrored)"),
              '#e74c3c')

# Row 3: M2
for k in range(2):
    plot_spec(fig.add_subplot(gs[2, k]), modes_M2[k], fs,
              f"({'ef'[k]})  M2  Channel-wise VMD — mode {k+1}",
              '#f39c12')

# Row 4: M3 (BEMD)
for k in range(min(2, len(modes_M3))):
    plot_spec(fig.add_subplot(gs[3, k]), modes_M3[k], fs,
              f"({'gh'[k]})  M3  BEMD (Rilling–Flandrin) — IMF {k+1}",
              '#2ecc71')

# Row 5: M4 (MCVMD)
for k in range(2):
    plot_spec(fig.add_subplot(gs[4, k]), modes_M4[k], fs,
              f"({'ij'[k]})  M4  MCVMD (heterodyne + VMD) — mode {k+1}",
              '#3498db')

for ax in fig.axes[-2:]:
    ax.set_xlabel("frequency (Hz)", fontsize=10)
for k in (0, 2, 4, 6, 8):
    fig.axes[k].set_ylabel("|FFT|² (dB)", fontsize=9.5)

fig.suptitle("Complex-IQ decomposition — 4 methods on two signed-Doppler tones",
             fontsize=13, fontweight='bold', y=0.985)

import os
if args.save:
    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or '.', exist_ok=True)
    plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved → {args.save}")
else:
    plt.show()
