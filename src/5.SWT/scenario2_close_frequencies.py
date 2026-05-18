"""
SWT vs EMD-HHT  ·  Scenario 2: Close Frequencies
=================================================

Signal: 25 Hz + 32 Hz tones, equal amplitude.
        Ratio f1/f2 = 0.78  ·  Rilling-Flandrin merge regime (ratio > 0.67).
        fs = 400 Hz, T = 4 s.

EMD merges both tones into a single beating IMF (Rilling-Flandrin 2008).
SST sharpens toward two ridges but they remain fused — the wavelet
bandwidth exceeds the 7 Hz gap. VMD with K = 2 is the right tool.

Requires:  pip install ssqueezepy EMD-signal numpy scipy matplotlib
Usage:     python scenario2_close_frequencies.py            # interactive display
           python scenario2_close_frequencies.py out.png    # save to file instead
"""
import numpy as np
from swt_emd_helpers import compute_sst, compute_emd_hht, plot_comparison


def make_signal():
    """25 Hz + 32 Hz tones, equal amplitude."""
    fs, T = 400.0, 4.0
    t = np.arange(int(fs * T)) / fs
    f1, f2 = 25.0, 32.0
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
    true_if = {
        "tone1": f1 * np.ones_like(t),
        "tone2": f2 * np.ones_like(t),
    }
    return t, x, fs, true_if


def main(savepath=None):
    t, x, fs, true_if = make_signal()
    f_max = 80.0
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
        title_a="25 Hz  +  32 Hz tones",
        subtitle_a="     (ratio 0.78 — Rilling-Flandrin merge regime)",
        f_max=f_max, ticks=(5, 10, 25, 32, 60),
        savepath=savepath,
    )
    if savepath is None:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    import sys
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
