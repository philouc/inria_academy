"""
SWT vs EMD-HHT  ·  Scenario 1: Chirp + Tone (noiseless)
========================================================

Signal: linear chirp 5→30 Hz  +  60 Hz tone.
        fs = 400 Hz, T = 4 s. Both modes have unit amplitude.

Both methods recover the two components cleanly because the modes are
well-separated in frequency. SST gives sharper TF maps; EMD provides
explicit modes that can be re-used for downstream analysis.

Requires:  pip install ssqueezepy EMD-signal numpy scipy matplotlib
Usage:     python scenario1_chirp_tone.py            # interactive display
           python scenario1_chirp_tone.py out.png    # save to file instead
"""
import numpy as np
from swt_emd_helpers import compute_sst, compute_emd_hht, plot_comparison


def make_signal():
    """5→30 Hz linear chirp + 60 Hz tone."""
    fs, T = 400.0, 4.0
    t = np.arange(int(fs * T)) / fs
    f0, f1 = 5.0, 30.0
    chirp_phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t**2)
    x = np.cos(chirp_phase) + np.cos(2 * np.pi * 60.0 * t)
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
        title_a="5→30 Hz chirp  +  60 Hz tone",
        subtitle_a="     (fs = 400 Hz, T = 4 s, noiseless)",
        f_max=f_max, ticks=(5, 10, 30, 60, 100),
        savepath=savepath,
    )
    if savepath is None:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    import sys
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
