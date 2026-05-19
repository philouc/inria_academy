"""
Shared helpers for the SST vs EMD-HHT scenario comparisons.

Provides:
    compute_sst(x, fs)              -> (Tx, Wx, ssq_freqs, cwt_freqs)
    compute_emd_hht(x, t, fs, fmax) -> (imfs, sig_imfs, H, n_kept, n_total)
    plot_comparison(...)            -> the 5-panel figure
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter
from PyEMD import EMD
from ssqueezepy import ssq_cwt
from ssqueezepy.experimental import scale_to_freq

# --- style ---------------------------------------------------------------
NAVY     = "#0b1f3a"
RED      = "#c9191e"
CYAN     = "#7FE0FF"
LIME     = "#7CFC00"        # used for true IF overlay (contrasts with magma)
GREEN    = "#1f8a4c"
GREY     = "#777777"
DARK_BG  = "#0a0a18"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#888", "axes.linewidth": 0.5,
    "xtick.color": "#444", "ytick.color": "#444",
})


# --- SST / CWT computation ----------------------------------------------
def compute_sst(x, fs):
    """Compute CWT and synchrosqueezed CWT using ssqueezepy.

    Uses the default Generalized Morse Wavelet (gmw), for which
    `scale_to_freq` gives the correct frequency mapping for the CWT.
    """
    Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(x, fs=fs)
    cwt_freqs = scale_to_freq(scales, wavelet="gmw",
                              N=len(x), fs=fs)
    return Tx, Wx, ssq_freqs, cwt_freqs


# --- EMD / HHT helpers --------------------------------------------------
def imf_inst_amp_freq(imf, fs):
    """Return (amplitude, instantaneous-frequency-in-Hz) of one IMF."""
    z = hilbert(imf)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    return amp, inst_freq


def hilbert_spectrum(imfs, fs, f_max, n_freqs=500, sigma=(2.5, 1.2)):
    """H[i,j] = total IMF amplitude reaching f_i at time t_j (then smoothed)."""
    n_t = imfs.shape[1]
    H = np.zeros((n_freqs, n_t))
    for imf in imfs:
        amp, inst_freq = imf_inst_amp_freq(imf, fs)
        for j in range(n_t):
            f = inst_freq[j]
            if 0 <= f < f_max:
                idx = int(f / f_max * (n_freqs - 1))
                H[idx, j] += amp[j]
    H = gaussian_filter(H, sigma=sigma)
    f_grid = np.linspace(0, f_max, n_freqs)
    return f_grid, H


def compute_emd_hht(x, t, fs, f_max, energy_thresh=0.01):
    """Run EMD, keep IMFs with >energy_thresh fraction of energy,
    and compute the Hilbert-Huang spectrum on those."""
    emd = EMD()
    imfs = emd(x, t)
    energies = np.array([np.sum(imf**2) for imf in imfs])
    energy_frac = energies / energies.sum()
    keep = energy_frac > energy_thresh
    sig_imfs = imfs[keep]
    f_grid, H = hilbert_spectrum(sig_imfs, fs, f_max=f_max)
    return imfs, sig_imfs, f_grid, H, int(keep.sum()), len(imfs)


def _tf_imshow(ax, M, t, freqs, f_max, log_norm=True, floor_ratio=5e-3,
               pct_vmax=99.5):
    """Display a TF magnitude map M with frequency axis `freqs` (any order).

    Uses pcolormesh so log-spaced frequencies are positioned correctly
    on the log y-axis (imshow would place them linearly in pixel space).
    """
    from matplotlib.colors import LogNorm
    freqs = np.asarray(freqs)
    order = np.argsort(freqs)
    M_sorted = M[order]
    f_sorted = freqs[order]
    # Only the visible band — clip to f_max + a bit
    mask = (f_sorted >= 0.5) & (f_sorted <= f_max * 1.5)
    f_use = f_sorted[mask]
    M_use = M_sorted[mask]
    Mmax = M_use.max() if M_use.size else 1.0
    T_grid, F_grid = np.meshgrid(t, f_use)
    if log_norm and Mmax > 0:
        floor = max(Mmax * floor_ratio, 1e-12)
        ax.pcolormesh(T_grid, F_grid, np.maximum(M_use, floor),
                      cmap="magma", shading="auto",
                      norm=LogNorm(vmin=floor, vmax=Mmax),
                      rasterized=True)
    else:
        vmax = np.percentile(M_use, pct_vmax) if Mmax > 0 else 1
        ax.pcolormesh(T_grid, F_grid, M_use,
                      cmap="magma", shading="auto",
                      vmin=0, vmax=vmax,
                      rasterized=True)


# --- plotting ------------------------------------------------------------
def _set_tf_axes(ax, t, f_max, *, log=True, ticks=(5, 10, 30, 60, 100),
                 xticklabels=True):
    """Common TF-plot axis cosmetics."""
    ax.set_ylim(max(2, ticks[0]), f_max)
    if log:
        ax.set_yscale("log")
        ax.set_yticks(list(ticks))
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(t[0], t[-1])
    if not xticklabels:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=6.5)
    for s in ax.spines.values():
        s.set_color(NAVY); s.set_linewidth(0.5)


def _overlay_true_if(ax, t, true_if_dict):
    """Plot true instantaneous frequencies as thin lime lines."""
    label_set = False
    for name, freq_arr in true_if_dict.items():
        ax.plot(t, freq_arr, color=LIME, lw=0.55, ls="-", alpha=0.85,
                label="true IF" if not label_set else None)
        label_set = True
    leg = ax.legend(loc="upper left", frameon=False, fontsize=5.5,
                    labelcolor="white", handlelength=1.2, handletextpad=0.4,
                    borderpad=0.2)
    if leg:
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)


def plot_comparison(t, x, fs, true_if, Tx, Wx, ssq_freqs, cwt_freqs,
                    sig_imfs, n_kept, n_total, f_grid_hht, H,
                    title_a, subtitle_a, f_max=100.0, ticks=(5, 10, 30, 60, 100),
                    savepath=None):
    """5-panel comparison figure for one scenario."""
    fig = plt.figure(figsize=(6.5, 3.7), dpi=200)
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[0.55, 1.4, 1.4],
        hspace=0.55, wspace=0.18,
        left=0.08, right=0.97, top=0.91, bottom=0.10,
    )

    # --- (a) signal -----------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, x, color=NAVY, lw=0.35)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(np.min(x) * 1.1, np.max(x) * 1.1)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"(a)  Signal:  {title_a}{subtitle_a}",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)
    for s in ax.spines.values():
        s.set_color("#888"); s.set_linewidth(0.4)

    # --- (b) CWT --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor(DARK_BG)
    _tf_imshow(ax, np.abs(Wx), t, cwt_freqs, f_max,
               log_norm=True, floor_ratio=1e-3)
    _overlay_true_if(ax, t, true_if)
    _set_tf_axes(ax, t, f_max, ticks=ticks, xticklabels=False)
    ax.set_ylabel("freq [Hz]", fontsize=7, labelpad=1)
    ax.set_title(r"(b)  CWT  $|W_x(a,b)|$",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)

    # --- (c) SST --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor(DARK_BG)
    _tf_imshow(ax, np.abs(Tx), t, ssq_freqs, f_max,
               log_norm=True, floor_ratio=1e-3)
    _overlay_true_if(ax, t, true_if)
    _set_tf_axes(ax, t, f_max, ticks=ticks, xticklabels=False)
    ax.set_title(r"(c)  SST  $|T_x(\omega,b)|$",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)

    # --- (d) IMFs -------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    n_keep = sig_imfs.shape[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_keep, 1)))
    for k, imf in enumerate(sig_imfs):
        amp_max = np.max(np.abs(imf)) + 1e-9
        offset = (n_keep - 1 - k) * 2.4
        ax.plot(t, imf / amp_max + offset, color=colors[k], lw=0.4)
        ax.text(t[-1] + 0.05, offset, f"IMF{k+1}",
                color=colors[k], fontsize=6, va="center",
                fontweight="bold")
    ax.set_xlim(t[0], t[-1] * 1.06)
    ax.set_ylim(-1.6, max(n_keep, 1) * 2.4)
    ax.set_xlabel("time [s]", fontsize=7, labelpad=1)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"(d)  EMD — kept {n_keep} of {n_total} IMFs (>1% energy)",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)
    for s in ax.spines.values():
        s.set_color("#888"); s.set_linewidth(0.4)

    # --- (e) HHT spectrum ----------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor(DARK_BG)
    _tf_imshow(ax, H, t, f_grid_hht, f_max,
               log_norm=True, floor_ratio=5e-3)
    _overlay_true_if(ax, t, true_if)
    _set_tf_axes(ax, t, f_max, ticks=ticks, xticklabels=True)
    ax.set_xlabel("time [s]", fontsize=7, labelpad=1)
    ax.set_title(r"(e)  Hilbert-Huang spectrum  $H(\omega,t)$",
                 loc="left", color=NAVY, fontsize=7.5,
                 fontweight="bold", pad=2)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig
