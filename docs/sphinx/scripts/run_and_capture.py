"""Run an example script, capture every matplotlib figure it produces, and
save each figure as a PNG file.

The wrapper sets ``MPLBACKEND=Agg`` and neutralises ``plt.show()`` so that
``examples/*.py`` scripts written for interactive display can be executed
non-interactively. Any figure left open at the end of the script (or
created before an unexpected exception) is then saved.

Usage
-----
    python run_and_capture.py <script_path> <output_stem> [--dpi N]

If the script creates a single figure, the file is saved as
``<output_stem>.png``. If it creates several, they are numbered
``<output_stem>_fig01.png``, ``<output_stem>_fig02.png``, …

Examples
--------
Regenerate a single gallery image::

    cd <project_root>
    PYTHONPATH=src python docs/sphinx/scripts/run_and_capture.py \\
        examples/7_MVMD/04.mvmd_ex4_alpha_eeg.py \\
        docs/sphinx/_static/gallery/module7_mvmd_eeg

Notes
-----
- The script's parent directory is prepended to ``sys.path`` so it can
  import sibling helper modules; the working directory is also changed
  to that folder for the duration of the run.
- ``sys.argv`` is reset before execution so argparse-based scripts do
  not see the wrapper's own arguments.
- The ``inria_academy`` package must be importable. The simplest way is
  to set ``PYTHONPATH=src`` (or ``uv run``, which handles this
  automatically since the project uses a ``src/`` layout).
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

# Headless backend — must be set BEFORE matplotlib is imported.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib                                  # noqa: E402
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt                    # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a Python script and save every open matplotlib "
                    "figure to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("script_path", type=Path,
                   help="Path to the .py script to execute.")
    p.add_argument("output_stem", type=Path,
                   help="Output path without extension. With one figure, "
                        "saves '<stem>.png'; with several, saves "
                        "'<stem>_fig01.png', '<stem>_fig02.png', ...")
    p.add_argument("--dpi", type=int, default=120,
                   help="DPI for the saved PNG (default: 120).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_path = args.script_path.resolve()
    out_stem    = args.output_stem.resolve()
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    if not script_path.is_file():
        print(f"error: {script_path} not found", file=sys.stderr)
        return 1

    # Neutralise blocking calls so the script terminates.
    plt.show = lambda *a, **k: None  # type: ignore[assignment]

    # The script may import sibling modules from its own directory and
    # may open data files with relative paths.
    sys.path.insert(0, str(script_path.parent))
    cwd_before = Path.cwd()
    os.chdir(script_path.parent)

    # Reset argv so argparse-based scripts don't see our flags.
    sys.argv = [str(script_path)]

    print(f"[run_and_capture] executing {script_path.name} ...")
    rc = 0
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as e:                  # CLI scripts often sys.exit(0)
        if e.code not in (None, 0):
            print(f"[run_and_capture] script exited with code {e.code}",
                  file=sys.stderr)
            rc = int(e.code) if isinstance(e.code, int) else 1
    except Exception as exc:
        print(f"[run_and_capture] EXCEPTION: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        rc = 2
        # Even on exception, fall through and try to save any figure
        # that was created before the crash.
    finally:
        os.chdir(cwd_before)

    # Save every open figure.
    fignums = plt.get_fignums()
    if not fignums:
        print("[run_and_capture] WARNING: no figures were created",
              file=sys.stderr)
        return rc or 3

    if len(fignums) == 1:
        path = out_stem.with_suffix(".png")
        plt.figure(fignums[0]).savefig(
            path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"[run_and_capture] saved {path}")
    else:
        for i, n in enumerate(fignums, 1):
            path = out_stem.parent / f"{out_stem.name}_fig{i:02d}.png"
            plt.figure(n).savefig(
                path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            print(f"[run_and_capture] saved {path}")

    plt.close("all")
    return rc


if __name__ == "__main__":
    sys.exit(main())
