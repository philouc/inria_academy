# inria_academy

**Inria Academy — Training on advanced signal processing methods.**

A set of Python scripts covering the foundations of Fourier and (continuous /
discrete) wavelet analysis, and then introducing tools for the analysis of
(i) non-stationary signals and (ii) multi-component mixed signals: the STFT
and Wigner-Ville distributions for time-frequency analysis, and EMD, SST,
VMD, and the multivariate VMD extension for the extraction of Intrinsic
Mode Functions (IMFs).

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) to manage the virtual environment

## Installation

```bash
git clone https://github.com/philouc/inria_academy.git
cd inria_academy
uv sync
```

A virtual environment is created automatically in `.venv/` and all
dependencies are installed from `pyproject.toml` and pinned via `uv.lock`.

To also install the development tools (mkdocs, pytest, ruff…):

```bash
uv sync --group dev
```

## Usage

Each script under `examples/` is a self-contained demo. Pick one and run it:

```bash
uv run python examples/1_Reminder_Fourier/spectral_filtering_demo.py
uv run python examples/7_MVMD/04.mvmd_ex4_alpha_eeg.py
```

See the [usage guide](docs/usage.md) for more details.

## Structure

```
inria_academy/
├── docs/                       MkDocs documentation sources
├── examples/                   Demo scripts (the gallery)
│   ├── 1_Reminder_Fourier/     Fourier domain refresher
│   ├── 2_Reminder_Wavelets/    Wavelet bases, denoising, long memory
│   ├── 3_STFT_WVD_SPWVD/       Short-Time Fourier & Wigner-Ville
│   ├── 4_EMD_EEMD_CEEMDAN/     Empirical mode decomposition family
│   ├── 5_SWT/                  Synchrosqueezing wavelet transform
│   ├── 6_VMD/                  Variational mode decomposition
│   └── 7_MVMD/                 Multivariate VMD (multichannel signals)
├── src/inria_academy/          Importable package (shared helpers)
│   └── utils/
│       ├── mvmd.py             Reference MVMD implementation
│       └── swt_emd.py          SST / EMD-HHT comparison helpers
├── pyproject.toml              Project metadata and dependencies
├── uv.lock                     Locked dependency versions (reproducibility)
└── README.md                   This file
```

## Documentation

The documentation is generated with [MkDocs](https://www.mkdocs.org/) and
the [Material](https://squidfunk.github.io/mkdocs-material/) theme, and is
published on [GitHub Pages](https://philouc.github.io/inria_academy/) after
every push to `main`.

To preview it locally:

```bash
uv run mkdocs serve
```

Then open <http://localhost:8000>.

## Citation

If you use this material in academic work, please cite it using the metadata
in [`CITATION.cff`](CITATION.cff), or use the *Cite this repository* button
at the top of the GitHub page.

## License

This work is licensed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
See [`LICENSE`](LICENSE) for the full text.

## Author

Philippe Ciuciu — Inria, NeuroSpin.
