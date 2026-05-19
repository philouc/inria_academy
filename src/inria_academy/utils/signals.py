"""Synthetic signal generators used across the example gallery.

These functions were originally defined inline inside specific demo
scripts. They are exposed here so that other demos can import them
without relying on script filenames (which start with a digit + dot and
are therefore not valid Python module names).

Signatures match the original ``make_signal`` functions to keep the
existing demos working unchanged.
"""

from __future__ import annotations

import numpy as np


def close_frequencies():
    """Two close tones at 25 Hz and 32 Hz, equal amplitude, no noise.

    Used as the canonical test signal for the *close-frequencies*
    scenario across modules 5 (SST) and 6 (VMD).

    Returns
    -------
    t : ndarray of shape (1600,)
        Time vector.
    x : ndarray of shape (1600,)
        Signal samples.
    fs : float
        Sampling frequency (Hz).
    true_if : dict of {str: ndarray}
        Ground-truth instantaneous frequency of each component, keyed
        by component name.
    """
    fs, T = 400.0, 4.0
    t = np.arange(int(fs * T)) / fs
    f1, f2 = 25.0, 32.0
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
    true_if = {
        "tone1": f1 * np.ones_like(t),
        "tone2": f2 * np.ones_like(t),
    }
    return t, x, fs, true_if


def chirp_tone_noise(seed: int = 0):
    """Linear chirp + steady 60 Hz tone + AWGN at 0 dB SNR.

    Scenario-1 signal contaminated by additive white Gaussian noise
    whose power is matched to the signal power.

    Parameters
    ----------
    seed : int, default 0
        Seed for the NumPy random generator, for reproducibility.

    Returns
    -------
    t : ndarray of shape (1600,)
        Time vector.
    x : ndarray of shape (1600,)
        Noisy signal samples.
    fs : float
        Sampling frequency (Hz).
    true_if : dict of {str: ndarray}
        Ground-truth instantaneous frequency of each clean component.
    """
    fs, T = 400.0, 4.0
    t = np.arange(int(fs * T)) / fs
    f0, f1 = 5.0, 30.0
    chirp_phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2)
    clean = np.cos(chirp_phase) + np.cos(2 * np.pi * 60.0 * t)
    # AWGN at 0 dB
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(clean**2)
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(sig_pow / np.mean(noise**2))  # match noise power to signal power
    x = clean + noise
    true_if = {
        "chirp": f0 + (f1 - f0) / T * t,
        "tone": 60.0 * np.ones_like(t),
    }
    return t, x, fs, true_if


__all__ = ["close_frequencies", "chirp_tone_noise"]
