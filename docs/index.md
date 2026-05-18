# Inria Academy — Signal Processing Demonstrations

A collection of self-contained Python demonstrations covering time-frequency analysis,
decomposition methods, and their applications to biomedical signals.

## Topics

| # | Topic | Methods |
|---|---|---|
| 1 | Fourier Basics | FFT, spectral filtering |
| 2 | Wavelets | CWT, wavelet bases, time-frequency tiling, denoising |
| 3 | Time-Frequency Distributions | STFT, Wigner-Ville, Smooth Wigner-Ville |
| 4 | EMD Family | EMD, EEMD, CEEMDAN, mode mixing |
| 5 | SWT | Stationary Wavelet Transform + EMD hybrids |
| 6 | VMD | Variational Mode Decomposition |
| 7 | MVMD | Multivariate VMD |

## Quick Start

```bash
uv sync
uv run python src/1.Reminder_Fourier/spectral_filtering_demo.py
```

Each script is self-contained and generates PNG figures alongside the source file.

## Documentation

- [Usage](usage.md) — setup and prerequisites
- [Gallery](gallery.md) — all demos with generated figures
- [API Reference](reference.md) — library modules
