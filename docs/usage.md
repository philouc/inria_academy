
# Usage guide

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) — fast Python package and environment manager

Install uv (once, on your machine):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or, on Windows (PowerShell):
# irm https://astral.sh/uv/install.ps1 | iex
```

## Installation

```bash
git clone https://github.com/philouc/inria_academy.git
cd inria_academy
uv sync
```

This creates a `.venv/` and installs every dependency at the exact version
pinned in `uv.lock`. To also install the development tools (mkdocs, ruff…):

```bash
uv sync --group dev
```

## Running a script

Each script in `src/` is fully self-contained — pick one, run it, get the
figure.

```bash
uv run python src/1.Reminder_Fourier/spectral_filtering_demo.py
uv run python src/2.Reminder_Wavelets/00.morlet_params.py
uv run python src/7.MVMD/04.mvmd_ex4_alpha_eeg.py
```

Most scripts open a matplotlib window. If you are on a headless server,
set a non-interactive backend before running:

```bash
export MPLBACKEND=Agg
```

## Helper modules

Two modules contain reusable building blocks rather than runnable demos:

- `src/5.SWT/swt_emd_helpers.py` — shared helpers (SST computation, EMD-HHT,
  comparison plot) used by every scenario in module 5
- `src/7.MVMD/mvmd.py` — a pure-Python reference implementation of MVMD
  (ur Rehman & Aftab 2019) used by every example in module 7

Both are imported by the demo scripts in their own directory; they are not
meant to be executed directly.

## Suggested learning path

1. Run module 1 first to refresh the Fourier-domain filtering pipeline.
2. Move through modules 2 → 4 in order: each builds on the time-frequency
   tools introduced previously.
3. Modules 5, 6, 7 can be approached as three parallel methods solving the
   same family of problems (mode decomposition of non-stationary signals);
   the side-by-side comparison scripts in module 5 (`03.scenario2b_*`,
   `05.scenario3b_*`) and module 6 (`02.scenario2c_*`) are designed
   specifically for that purpose.

## Troubleshooting

**`ModuleNotFoundError: No module named 'PyEMD'`** — the package on PyPI is
`EMD-signal`, not `PyEMD` (the import name differs from the package name).
`uv sync` already installs it correctly; this only fails if you bypass uv.

**Matplotlib windows do not appear on macOS** — install a GUI backend or
save figures with `plt.savefig(...)` instead of `plt.show()`.

**Long runtime on VMD/EEMD scripts** — these methods are iterative.
A 500-iteration ADMM solve on a multichannel signal can take 30–60 s,
which is normal.
