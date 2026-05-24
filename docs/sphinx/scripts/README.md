# Documentation gallery scripts

Two small helpers to regenerate the static images embedded in
[`reference.rst`](../reference.rst).

| Script | Role |
|---|---|
| `run_and_capture.py` | Generic wrapper: runs a single example script with a headless matplotlib backend and saves every figure it produces. |
| `regen_gallery.py`   | Orchestrator: knows which 8 example scripts (one per module) are featured in the gallery and regenerates all of them in one command. |

## Regenerate the full gallery

From the project root, with the `uv`-managed venv active (or via `uv run`):

```bash
uv run python docs/sphinx/scripts/regen_gallery.py
```

This produces eight (or ten, counting the multi-figure module 2) PNGs
in `docs/sphinx/_static/gallery/`.

## Regenerate a subset

```bash
uv run python docs/sphinx/scripts/regen_gallery.py --only 7 8
```

Only re-runs modules 7 (MVMD on EEG) and 8 (multivariate IQ radar).

## Regenerate one arbitrary script

If you want to capture a script that is not in the gallery list — e.g.
to preview a figure before deciding whether to feature it:

```bash
PYTHONPATH=src uv run python docs/sphinx/scripts/run_and_capture.py \
    examples/6_VMD/05.vmd_ex5_ecg.py \
    /tmp/vmd_ecg_preview
```

This writes `/tmp/vmd_ecg_preview.png` (or `_fig01.png`, `_fig02.png`,
… if the script creates several figures).

## Changing which script is featured per module

Edit the `REPRESENTATIVES` list at the top of `regen_gallery.py`. Each
entry is a tuple `(module_number, script_relative_path, output_basename)`.
The `output_basename` (e.g. `module7_mvmd_eeg`) must match the filename
referenced by the corresponding `.. figure::` directive in
`reference.rst`, or you'll need to update both.

## Implementation notes

- Each example script runs in its **own** subprocess so that global
  matplotlib state, numpy random seeds, and module-level caches do not
  leak between modules.
- `MPLBACKEND=Agg` is forced and `plt.show()` is monkey-patched to a
  no-op so interactive scripts terminate normally.
- `PYTHONPATH` is set so the `inria_academy` package is importable
  even when the venv is not formally activated.
- Heavy dependencies (`PyEMD`/`EMD-signal`, `ssqueezepy`, `vmdpy`,
  `neurokit2`) must actually be installed — `autodoc_mock_imports` in
  `conf.py` is only consulted at Sphinx build time, not when running
  the demo scripts.
