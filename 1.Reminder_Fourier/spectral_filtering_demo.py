"""
Spectral Filtering Demo — Linear Filtering in the Fourier Domain
=================================================================
Dependencies : numpy, matplotlib
Run          : python spectral_filtering_demo.py

The script generates the full filtering pipeline figure:
  x(t) -> FT -> ×H(f) -> IFT -> y(t)

Signal : 5 Hz sine + 50 Hz sine (noise) + Gaussian noise
Filter : ideal low-pass, cutoff at 20 Hz
Author: Philippe Ciuciu
Date: 02/04/2026
Target: UnseenLabs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

BG      = '#0D1B2A'
CYAN    = '#00C2FF'
RED     = '#FF6B6B'
GREEN   = '#A8FF78'
GOLD    = '#FFD700'
TEXT    = '#E0E6ED'
MUTED   = '#A0AEC0'

# ── Signal Definition & Filter ──────────────────────────────────────────────────────────
N  = 1000           # number of samples
fs = 1000           # sampling frequency (Hz)
t  = np.linspace(0, 1, N, endpoint=False)
freqs = np.fft.rfftfreq(N, 1 / fs)

np.random.seed(42)
x = (  np.sin(2 * np.pi * 5  * t)          # 5 Hz component  (kept)
     + 0.5 * np.sin(2 * np.pi * 50 * t)    # 50 Hz component (removed)
     + 0.3 * np.random.randn(N))            # Gaussian noise

# Low-pass filter — cutoff 20 Hz, smooth transition up to 30 Hz (cosine roll-off)
H = np.zeros(len(freqs))
H[freqs <= 20] = 1.0
mask = (freqs > 20) & (freqs < 30)
H[mask] = 0.5 * (1 + np.cos(np.pi * (freqs[mask] - 20) / 10))

# Fourier-domain filtering
X = np.fft.rfft(x)          # Forward FT
Y = X * H                   # Spectral multiplication  ← the key operation
y = np.fft.irfft(Y, n=N)    # Inverse FT

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6), facecolor=BG)
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        hspace=0.55, wspace=0.45,
                        left=0.05, right=0.97,
                        top=0.88,  bottom=0.12)

def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#334155')

# ── Column 0 : time-domain signal ────────────────────────────────────────────
ax0 = fig.add_subplot(gs[:, 0])
ax0.plot(t[:300], x[:300], color=CYAN, lw=0.9, alpha=0.85)
ax0.set_xlabel('Time (s)', color=TEXT, fontsize=8)
ax0.set_title('Signal x(t)', color=CYAN, fontsize=9, fontweight='bold')
ax0.set_xlim(0, 0.3)
style_ax(ax0)

# ── Column 1 top : amplitude spectrum X(f) ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(freqs[:80], np.abs(X[:80]), color=RED, lw=1.2)
ax1.set_xlabel('Frequency (Hz)', color=TEXT, fontsize=7)
ax1.set_title('|X(f)| — Signal FT', color=RED, fontsize=8.5, fontweight='bold')
style_ax(ax1)

# ── Column 1 bottom : filter frequency response H(f) ─────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(freqs[:80], H[:80], color=GREEN, lw=1.5)
ax2.fill_between(freqs[:80], H[:80], alpha=0.18, color=GREEN)
ax2.set_xlabel('Frequency (Hz)', color=TEXT, fontsize=7)
ax2.set_title('H(f) — Low-pass Filter', color=GREEN, fontsize=8.5, fontweight='bold')
style_ax(ax2)

# ── Column 2 top : filtered spectrum Y(f) ────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(freqs[:80], np.abs(Y[:80]), color=GOLD, lw=1.2)
ax3.set_xlabel('Frequency (Hz)', color=TEXT, fontsize=7)
ax3.set_title('|Y(f)| = |X(f)·H(f)|', color=GOLD, fontsize=8.5, fontweight='bold')
style_ax(ax3)

# ── Column 2 bottom : filtered time-domain signal y(t) ───────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(t[:300], y[:300], color=GOLD, lw=1.0, alpha=0.9)
ax4.set_xlabel('Time (s)', color=TEXT, fontsize=7)
ax4.set_title('Filtered signal y(t)', color=GOLD, fontsize=8.5, fontweight='bold')
ax4.set_xlim(0, 0.3)
style_ax(ax4)

# ── Operators (×, IFT, →) ────────────────────────────────────────────────────
fig.text(0.500, 0.72, '×',   color=TEXT,  fontsize=18, ha='center', va='center', fontweight='bold')
fig.text(0.500, 0.28, '×',   color=TEXT,  fontsize=18, ha='center', va='center', fontweight='bold')
fig.text(0.500, 0.50, '↕ IFT', color=MUTED, fontsize=8,  ha='center', va='center')
fig.text(0.745, 0.72, '→',   color=TEXT,  fontsize=22, ha='center', va='center')
fig.text(0.745, 0.28, '→',   color=TEXT,  fontsize=22, ha='center', va='center')

# ── Column 3 : Convolution Theorem reminder ───────────────────────────────────
ax5 = fig.add_subplot(gs[:, 3])
ax5.set_facecolor('#0F2233')
ax5.axis('off')
for sp in ax5.spines.values():
    sp.set_color('#334155')

annotations = [
    ("Convolution",              0.85, TEXT,  9,    False),
    ("Theorem",                  0.76, TEXT,  9,    False),
    ("",                         0.68, TEXT,  8,    False),
    ("y = h * x",                0.60, CYAN,  13,   True ),
    ("\u27fa",                   0.50, TEXT,  14,   False),
    ("Y(f) = H(f)\u00b7X(f)",   0.40, GREEN, 11,   True ),
    ("",                         0.30, TEXT,  8,    False),
    ("Convolution \u2192 Multiplication", 0.20, MUTED, 7.5, False),
    ("in the Fourier domain",    0.12, MUTED, 7.5,  False),
]
for txt, ypos, col, fsize, bold in annotations:
    ax5.text(0.5, ypos, txt,
             color=col, fontsize=fsize,
             ha='center', va='center',
             fontweight='bold' if bold else 'normal',
             transform=ax5.transAxes)

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Spectral filtering pipeline:  x(t)  →  FT  →  ×H(f)  →  IFT  →  y(t)",
    color=TEXT, fontsize=10.5, fontweight='bold', y=0.97
)

plt.savefig('spectral_filtering_pipeline.png', dpi=160,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("Figure saved to spectral_filtering_pipeline.png")
