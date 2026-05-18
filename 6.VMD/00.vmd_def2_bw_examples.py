"""
VMD — Bandwidth illustrated  (Dragomiretskiy & Zosso 2014, Fig. 2)
====================================================================

The AM-FM signal model used in the paper:

    f(t) = (1 + a_AM·cos(2π f_AM t)) · cos(2π f_c t + (Δf/f_FM)·cos(2π f_FM t))

Four panels make the three contributions to bandwidth visible one at
a time:

    (a) Pure AM            Δf = 0                  → BW dominated by f_AM
    (b) FM, f_FM >> Δf     rapid + small deviation → BW dominated by f_FM
    (c) FM, f_FM << Δf     slow  + large deviation → BW dominated by Δf
    (d) Combined AM-FM     f_AM ~ f_FM ~ Δf        → all three add

For each panel:
    - Left:  time-domain signal (1 s window)
    - Right: log-log magnitude spectrum, with the carrier f_c marked
             by a solid red line and the band limits f_c ± BW/2 by
             dotted red lines.

Palette and style match the rest of the deck (NAVY signal, RED carrier,
white background suitable for the Inria Academy template).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec


# --- Palette (matches deck) -----------------------------------------------
NAVY  = "#0b1f3a"
RED   = "#c9191e"
AMBER = "#e08a1f"
GREEN = "#1f8a4c"
GREY  = "#777777"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#666",
    "axes.linewidth": 0.5,
    "xtick.color": "#444",
    "ytick.color": "#444",
})


# --- Signal model ---------------------------------------------------------
def am_fm_signal(t, f_c, f_AM=0.0, a_AM=0.5, f_FM=0.0, df=0.0):
    """f(t) = (1 + a_AM cos(2π f_AM t)) · cos(2π f_c t + β cos(2π f_FM t))

    β = Δf / f_FM is the FM modulation index. Avoid div-by-zero when
    f_FM == 0 by also forcing Δf = 0 (purely-AM case).
    """
    amp = 1.0 + (a_AM * np.cos(2 * np.pi * f_AM * t) if f_AM > 0 else 0.0)
    if f_FM > 0 and df > 0:
        beta = df / f_FM
        carrier = np.cos(2 * np.pi * f_c * t
                         + beta * np.cos(2 * np.pi * f_FM * t))
    else:
        carrier = np.cos(2 * np.pi * f_c * t)
    return amp * carrier


def bandwidth(df, f_FM, f_AM):
    """Total practical bandwidth BW = 2 (Δf + f_FM + f_AM)."""
    return 2.0 * (df + f_FM + f_AM)


# --- The four scenarios ---------------------------------------------------
def get_scenarios():
    """Return list of (label, caption, f_AM, f_FM, df, accent_color)."""
    return [
        ("(a)  AM",
         r"$\Delta f = 0$  ·  pure AM at $f_{AM}=50$ Hz",
         50.0, 0.0,  0.0, NAVY),

        ("(b)  FM  ($f_{FM} \gg \Delta f$)",
         r"rapid, small deviation:  $f_{FM}=80$, $\Delta f=15$ Hz",
         0.0, 80.0, 15.0, GREEN),

        ("(c)  FM  ($f_{FM} \ll \Delta f$)",
         r"slow, large deviation:  $f_{FM}=8$, $\Delta f=80$ Hz",
         0.0,  8.0, 80.0, AMBER),

        ("(d)  AM-FM",
         r"$f_{AM} \sim f_{FM} \sim \Delta f \approx 30$ Hz",
         30.0, 30.0, 30.0, RED),
    ]


# --- Plot one scenario row ------------------------------------------------
def plot_row(ax_t, ax_f, t, x, fs, f_c, df, f_FM, f_AM, label, caption,
             accent):
    # ----- time-domain (left) -----
    ax_t.plot(t, x, color=NAVY, lw=0.55)
    ax_t.set_xlim(t[0], t[-1])
    y_max = np.max(np.abs(x)) * 1.10
    ax_t.set_ylim(-y_max, y_max)
    ax_t.set_yticks([-1, 0, 1])
    ax_t.tick_params(labelsize=6.5)
    # No xlabel here — only on the bottom row (added in main)

    # ----- spectrum (right) -----
    # Compute mag spectrum, single-sided
    N = len(x)
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(N, 1.0 / fs)

    ax_f.loglog(freqs[1:], np.maximum(mag[1:], 1e-2),
                color=NAVY, lw=0.55)
    # carrier (solid red line)
    ax_f.axvline(f_c, color=RED, lw=1.2)
    # band limits (dotted red)
    BW = bandwidth(df, f_FM, f_AM)
    f_lo = max(f_c - BW / 2.0, 1.0)
    f_hi = f_c + BW / 2.0
    ax_f.axvline(f_lo, color=RED, lw=0.7, ls=":")
    ax_f.axvline(f_hi, color=RED, lw=0.7, ls=":")
    ax_f.set_xlim(10, 1000)
    ax_f.set_ylim(1e-1, 5e2)
    ax_f.set_xticks([10, 100, 1000])
    ax_f.set_xticklabels(
        [r"$10^1$", r"$10^2$", r"$10^3$"])
    ax_f.tick_params(labelsize=6.5)
    ax_f.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")

    # ----- panel label (above the time plot) -----
    ax_t.set_title(label, loc="left", color=accent,
                   fontsize=8, fontweight="bold", pad=2)
    # caption below time plot (in the space we left between rows)
    ax_t.text(0.0, -0.30, caption, transform=ax_t.transAxes,
              fontsize=6.5, color=GREY, ha="left", va="top")
    # BW annotation on spectrum
    ax_f.text(0.98, 0.92,
              f"BW = 2 (Δf + f$_{{FM}}$ + f$_{{AM}}$) = {BW:.0f} Hz",
              transform=ax_f.transAxes, ha="right", va="top",
              fontsize=6.5, color=accent, fontweight="bold")


# --- Main figure -----------------------------------------------------------
def main(savepath=None):
    fs = 2000.0
    T = 1.0
    t = np.arange(int(fs * T)) / fs
    f_c = 200.0

    scenarios = get_scenarios()

    fig = plt.figure(figsize=(7.5, 4.6), dpi=200)
    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[1, 1, 1, 1],
        width_ratios=[1.0, 1.0],
        hspace=0.60, wspace=0.20,
        left=0.07, right=0.97, top=0.96, bottom=0.07,
    )

    for i, (label, caption, f_AM, f_FM, df, accent) in enumerate(scenarios):
        x = am_fm_signal(t, f_c=f_c, f_AM=f_AM, f_FM=f_FM, df=df)
        ax_t = fig.add_subplot(gs[i, 0])
        ax_f = fig.add_subplot(gs[i, 1])
        plot_row(ax_t, ax_f, t, x, fs, f_c, df, f_FM, f_AM,
                 label, caption, accent)
        # Only label x-axis on bottom row
        if i == len(scenarios) - 1:
            ax_t.set_xlabel("t [s]", fontsize=6.5, labelpad=1)
            ax_f.set_xlabel("freq [Hz]", fontsize=6.5, labelpad=1)
        else:
            ax_t.set_xticklabels([])
            ax_f.set_xticklabels([])

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        print(f"Saved: {savepath}")
    return fig


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    main(savepath=out)
    if out is None:
        plt.show()
