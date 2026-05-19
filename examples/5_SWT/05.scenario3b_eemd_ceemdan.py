"""
SWT vs EMD-HHT  ·  Scenario 3b: noise-robust EMD variants
==========================================================

Same noisy signal as Scenario 3 (chirp + tone + AWGN at SNR = 0 dB),
analysed by three EMD-family decompositions side by side:

    EMD         basic  (Huang 1998)
    EEMD        Ensemble EMD  (Wu & Huang 2009)            — adds noise upfront
    CEEMDAN     Complete EEMD with Adaptive Noise          — adds noise per stage
                (Torres et al. 2011)

EEMD and CEEMDAN dramatically clean up the IMFs at this SNR.  But the
gain has costs:  100×-300× slower (ensemble averaging), the user has to
tune the noise std and trial count, and the decomposition is still
recursive (one IMF at a time, no parallelism, no theoretical bound).

Requires:  pip install ssqueezepy EMD-signal numpy scipy matplotlib
Usage:     python scenario3b_eemd_ceemdan.py            # interactive display
           python scenario3b_eemd_ceemdan.py out.png    # save to file instead
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from PyEMD import EMD, EEMD, CEEMDAN
from inria_academy.utils.swt_emd import (
    NAVY, LIME, DARK_BG,
    hilbert_spectrum, _tf_imshow, _set_tf_axes, _overlay_true_if,
)
from inria_academy.utils.signals import chirp_tone_noise as make_signal


def keep_significant(imfs, thresh=0.01):
    energies = np.array([np.sum(imf**2) for imf in imfs])
    keep = energies / energies.sum() > thresh
    return imfs[keep], int(keep.sum()), len(imfs)


def run_methods(x, t, fs, f_max,
                eemd_trials=100, eemd_noise=0.2,
                ceemdan_trials=100, ceemdan_eps=0.2):
    """Return a dict of {method_name: (sig_imfs, n_kept, n_total, H, t_run)}."""
    results = {}

    # EMD ----------------------------------------------------------------
    t0 = time.time()
    imfs = EMD()(x, t)
    sig, n_kept, n_tot = keep_significant(imfs)
    _, H = hilbert_spectrum(sig, fs, f_max=f_max)
    results["EMD"] = (sig, n_kept, n_tot, H, time.time() - t0)
    print(f"  EMD     : {n_tot} IMFs, {n_kept} kept   ({results['EMD'][-1]:.1f}s)")

    # EEMD ---------------------------------------------------------------
    t0 = time.time()
    eemd = EEMD(trials=eemd_trials, noise_width=eemd_noise)
    imfs = eemd.eemd(x, t)
    sig, n_kept, n_tot = keep_significant(imfs)
    _, H = hilbert_spectrum(sig, fs, f_max=f_max)
    results["EEMD"] = (sig, n_kept, n_tot, H, time.time() - t0)
    print(f"  EEMD    : {n_tot} IMFs, {n_kept} kept   "
          f"({results['EEMD'][-1]:.1f}s, "
          f"{eemd_trials} trials, noise σ = {eemd_noise})")

    # CEEMDAN ------------------------------------------------------------
    t0 = time.time()
    ceemdan = CEEMDAN(trials=ceemdan_trials, epsilon=ceemdan_eps)
    imfs = ceemdan.ceemdan(x, t)
    sig, n_kept, n_tot = keep_significant(imfs)
    _, H = hilbert_spectrum(sig, fs, f_max=f_max)
    results["CEEMDAN"] = (sig, n_kept, n_tot, H, time.time() - t0)
    print(f"  CEEMDAN : {n_tot} IMFs, {n_kept} kept   "
          f"({results['CEEMDAN'][-1]:.1f}s, "
          f"{ceemdan_trials} trials, ε = {ceemdan_eps})")

    return results


def plot_3way_comparison(t, x, fs, true_if, results, f_max=100.0,
                         ticks=(5, 10, 30, 60, 100), savepath=None):
    fig = plt.figure(figsize=(6.5, 3.7), dpi=200)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 1.1],
        hspace=0.45, wspace=0.18,
        left=0.07, right=0.97, top=0.90, bottom=0.10,
    )

    methods = ["EMD", "EEMD", "CEEMDAN"]
    method_labels = ["(a)  EMD", "(b)  EEMD", "(c)  CEEMDAN"]

    # --- Row 0: IMFs ----------------------------------------------------
    for col, m in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        sig_imfs, n_kept, n_tot, H, t_run = results[m]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_kept, 1)))
        for k, imf in enumerate(sig_imfs):
            amp = np.max(np.abs(imf)) + 1e-9
            offset = (n_kept - 1 - k) * 2.4
            ax.plot(t, imf / amp + offset, color=colors[k], lw=0.3)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(-1.6, max(n_kept, 1) * 2.4)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_title(f"{method_labels[col]} — {n_kept}/{n_tot} IMFs   "
                     f"({t_run:.1f}s)",
                     loc="left", color=NAVY, fontsize=7,
                     fontweight="bold", pad=2)
        for s in ax.spines.values():
            s.set_color("#888"); s.set_linewidth(0.4)

    # --- Row 1: HHT spectra ---------------------------------------------
    for col, m in enumerate(methods):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(DARK_BG)
        _, _, _, H, _ = results[m]
        f_grid_hht = np.linspace(0, f_max, H.shape[0])
        _tf_imshow(ax, H, t, f_grid_hht, f_max,
                   log_norm=True, floor_ratio=5e-3)
        _overlay_true_if(ax, t, true_if)
        _set_tf_axes(ax, t, f_max, ticks=ticks, xticklabels=True)
        ax.set_xlabel("time [s]", fontsize=6.5, labelpad=1)
        if col == 0:
            ax.set_ylabel("freq [Hz]", fontsize=6.5, labelpad=1)
        ax.set_title(f"HHT spectrum  ({m})",
                     loc="left", color=NAVY, fontsize=7,
                     fontweight="bold", pad=2)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


def main(savepath=None):
    t, x, fs, true_if = make_signal(seed=0)
    f_max = 100.0
    print("Running three EMD-family decompositions on Scenario 3 signal...")
    results = run_methods(x, t, fs, f_max,
                          eemd_trials=100, eemd_noise=0.2,
                          ceemdan_trials=100, ceemdan_eps=0.2)
    plot_3way_comparison(t, x, fs, true_if, results,
                         f_max=f_max, ticks=(5, 10, 30, 60, 100),
                         savepath=savepath)
    if savepath is None:
        plt.show()


if __name__ == "__main__":
    main(savepath=sys.argv[1] if len(sys.argv) > 1 else None)
