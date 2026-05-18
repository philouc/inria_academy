"""
SWT vs EMD-HHT  ·  Scenario 3: Chirp + Tone in Noise
=====================================================

Signal: Scenario 1 signal  +  additive white Gaussian noise.
        Realised SNR = 0 dB (signal power == noise power).
        fs = 400 Hz, T = 4 s. Random seed fixed for reproducibility.

SST: fragments at chirp edge; the IF estimator becomes unstable on noise.
EMD: noise spreads across many IMFs (6 of 8 kept vs only 2 in the noiseless
case). HHT spectrum becomes chaotic.

Both methods degrade. EEMD / CEEMDAN (noise-assisted) or VMD with α tuned
give cleaner decompositions at this SNR.

Requires:  pip install ssqueezepy EMD-signal numpy scipy matplotlib
Usage:     python scenario3_chirp_tone_noise.py            # interactive display
           python scenario3_chirp_tone_noise.py out.png    # save to file instead
"""
import numpy as np
from swt_emd_helpers import compute_sst, compute_emd_hht, plot_comparison


def make_signal(seed=0):
    """Scenario 1 signal + AWGN at 0 dB SNR."""
    fs, T = 400.0, 4.0
    t = np.arange(int(fs * T)) / fs
    f0, f1 = 5.0, 30.0
    chirp_phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2)
    clean = np.cos(chirp_phase) + np.cos(2 * np.pi * 60.0 * t)
    # AWGN at 0 dB
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(clean**2)
    noise = rng.standard_normal(len(t))
    noise *= np.sqrt(sig_pow / np.mean(noise**2))   # match noise power to signal power
    x = clean + noise
    true_if = {
        "chirp": f0 + (f1 - f0) / T * t,
        "tone":  60.0 * np.ones_like(t),
    }
    return t, x, fs, true_if


def main(savepath=None):
    t, x, fs, true_if = make_signal()
    f_max = 100.0
    print("Computing SST / CWT...")
    Tx, Wx, ssq_freqs, cwt_freqs = compute_sst(x, fs)
    print("Computing EMD / HHT...")
    imfs, sig_imfs, f_hht, H, n_kept, n_total = compute_emd_hht(
        x, t, fs, f_max=f_max
    )
    print(f"  IMFs total {n_total}, kept {n_kept}")
    plot_comparison(
        t, x, fs, true_if,
        Tx, Wx, ssq_freqs, cwt_freqs,
        sig_imfs, n_kept, n_total, f_hht, H,
        title_a="5→30 Hz chirp  +  60 Hz tone  +  AWGN",
        subtitle_a="     (SNR = 0 dB, fs = 400 Hz, T = 4 s)",
        f_max=f_max, ticks=(5, 10, 30, 60, 100),
        savepath=savepath,
    )
    if savepath is None:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    import sys
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
