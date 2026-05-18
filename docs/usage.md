# Usage

## Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management

## Installation

```bash
uv sync          # installs from pyproject.toml + uv.lock
```

## Running Scripts

Each topic directory under `src/` contains standalone scripts:

```bash
uv run python src/X.Topic/script.py
```

Scripts generate PNG figures saved alongside the source file. These are gitignored
(`*.png`, `outputs/`).

## Library Modules

Only two modules are designed for import:

| Module | Location | Purpose |
|---|---|---|
| `mvmd` | `src/7.MVMD/mvmd.py` | Multivariate VMD implementation |
| `swt_emd_helpers` | `src/5.SWT/swt_emd_helpers.py` | SWT + EMD utility functions |

## Dev Commands

```bash
uv run ruff check src/    # lint
uv run mkdocs serve       # preview docs
```
