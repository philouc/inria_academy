# Inria Academy — Advanced Signal Processing Tools

Training material for advanced signal processing methods.

The course starts from Fourier and wavelet foundations and progressively
introduces tools dedicated to the analysis of **non-stationary** and
**multi-component** signals: STFT, WVD/SPWVD, EMD/EEMD/CEEMDAN, the
synchrosqueezing wavelet transform (SST/SWT), VMD, and its multivariate
extension MVMD.

Each module is a self-contained directory of numbered Python scripts that
can be executed independently to reproduce every figure and result.

## Course outline

| # | Module | Topic |
|---|--------|-------|
| 1 | `1.Reminder_Fourier` | Fourier analysis — spectral filtering refresher |
| 2 | `2.Reminder_Wavelets` | Wavelet bases, time-frequency tiling, long-memory processes, denoising |
| 3 | `3.STFT_WVD_SPWVD` | Short-Time Fourier and Wigner-Ville distributions |
| 4 | `4.EMD_EEMD_CEEMDAN` | Empirical Mode Decomposition and its noise-assisted variants |
| 5 | `5.SWT` | Synchrosqueezing wavelet transform, compared against EMD |
| 6 | `6.VMD` | Variational Mode Decomposition |
| 7 | `7.MVMD` | Multivariate VMD on multichannel signals (EEG, biosignals) |

## Getting started

- [Usage guide](usage.md) — installation, environment, how to run a script
- [Reference](reference.md) — annotated index of every script

## Author

Philippe Ciuciu — Inria, NeuroSpin.

## License

The pedagogical material (slides, figures, written notes) is released under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
See [`LICENSE`](https://github.com/philouc/inria_academy/blob/main/LICENSE).
