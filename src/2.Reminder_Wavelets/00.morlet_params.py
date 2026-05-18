"""
Demo Morlet wavelet
====================
Dependencies : numpy, matplotlib
Run          : python 00.morlet_params.py

Author: Philippe Ciuciu
Date: 04/04/2026
Target: UnseenLabs
"""


import numpy as np
import matplotlib.pyplot as plt

def morlet(t, sigma=1.0, c=2.5):
    """Morlet mère : ψ(t) = π^{-1/4} · e^{i·2c·t} · e^{-t²/(2σ²)}, partie réelle."""
    psi = np.pi**(-0.25) * np.exp(1j * 2.0 * c * t) * np.exp(-t**2 / (2.0 * sigma**2))
    return psi.real

# Grille temporelle
t = np.linspace(-6, 6, 2001)
dt = t[1] - t[0]
N = len(t)

# Grille fréquentielle (Hz) pour la FFT (signal réel → fréquences positives)
freqs = np.fft.rfftfreq(N, d=dt)

def spectrum(psi):
    """Spectre d'amplitude normalisé (côté positif)."""
    S = np.abs(np.fft.rfft(psi)) * dt
    return S

# Palettes
cmap_sigma = plt.cm.viridis(np.linspace(0.15, 0.85, 3))
cmap_c     = plt.cm.plasma(np.linspace(0.15, 0.80, 3))

sigmas = [0.5, 1.0, 2.0]
cs     = [1.0, 2.5, 5.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)

# ---- Ligne 1 : variation de σ, c fixé ----
ax = axes[0, 0]
for s, col in zip(sigmas, cmap_sigma):
    psi = morlet(t, sigma=s, c=2.5)
    ax.plot(t, psi, color=col, lw=1.6, label=rf"$\sigma={s}$")
    # enveloppe gaussienne (pointillé)
    env = np.pi**(-0.25) * np.exp(-t**2 / (2.0 * s**2))
    ax.plot(t, env,  color=col, lw=0.9, ls='--', alpha=0.5)
    ax.plot(t, -env, color=col, lw=0.9, ls='--', alpha=0.5)
ax.axhline(0, color='k', lw=0.4)
ax.set_title(r"Domaine temporel — variation de $\sigma$  (c = 2.5)", fontsize=11)
ax.set_xlabel("t"); ax.set_ylabel(r"$\Re\{\psi(t)\}$")
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.grid(alpha=0.25)

ax = axes[0, 1]
for s, col in zip(sigmas, cmap_sigma):
    psi = morlet(t, sigma=s, c=2.5)
    S = spectrum(psi)
    ax.plot(freqs, S / S.max(), color=col, lw=1.6, label=rf"$\sigma={s}$")
ax.axvline(2*2.5/(2*np.pi), color='k', lw=0.5, ls=':',
           label=rf"$f_0 = \omega_0/2\pi \approx {2*2.5/(2*np.pi):.2f}$ Hz")
ax.set_title(r"Domaine fréquentiel — variation de $\sigma$", fontsize=11)
ax.set_xlabel("f (Hz)"); ax.set_ylabel(r"$|\hat\psi(f)|$  (normalisé)")
ax.set_xlim(0, 2.5)
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.grid(alpha=0.25)

# ---- Ligne 2 : variation de c, σ fixé ----
ax = axes[1, 0]
for c, col in zip(cs, cmap_c):
    psi = morlet(t, sigma=1.0, c=c)
    ax.plot(t, psi, color=col, lw=1.6, label=rf"$c={c}\ \Rightarrow\ \omega_0={2*c}$")
env = np.pi**(-0.25) * np.exp(-t**2 / 2.0)
ax.plot(t,  env, color='k', lw=0.8, ls='--', alpha=0.4, label="enveloppe gaussienne")
ax.plot(t, -env, color='k', lw=0.8, ls='--', alpha=0.4)
ax.axhline(0, color='k', lw=0.4)
ax.set_title(r"Domaine temporel — variation de $c$  ($\sigma = 1$)", fontsize=11)
ax.set_xlabel("t"); ax.set_ylabel(r"$\Re\{\psi(t)\}$")
ax.legend(frameon=False, loc='upper right', fontsize=8.5)
ax.grid(alpha=0.25)

ax = axes[1, 1]
for c, col in zip(cs, cmap_c):
    psi = morlet(t, sigma=1.0, c=c)
    S = spectrum(psi)
    ax.plot(freqs, S / S.max(), color=col, lw=1.6,
            label=rf"$c={c}$  ($f_0\approx{2*c/(2*np.pi):.2f}$ Hz)")
ax.set_title(r"Domaine fréquentiel — variation de $c$", fontsize=11)
ax.set_xlabel("f (Hz)"); ax.set_ylabel(r"$|\hat\psi(f)|$  (normalisé)")
ax.set_xlim(0, 2.5)
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.grid(alpha=0.25)

fig.suptitle(r"Influence de $\sigma$ et $c$ sur l'ondelette de Morlet  "
             r"$\psi(t) = \pi^{-1/4}\, e^{\,i\,2c\,t}\, e^{-t^2/(2\sigma^2)}$",
             fontsize=12.5, y=1.02)

out = "./morlet_sigma_c.png"
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print("Saved:", out)
