"""Regenerate every image in the static documentation gallery.

This orchestrator runs the eight representative example scripts (one per
module) and saves their figures into
``docs/sphinx/_static/gallery/``. Internally it shells out to
``run_and_capture.py`` for each script so that any leftover state from
one run does not contaminate the next (heavy IMF/CWT computations,
global matplotlib styles, …).

Usage
-----
From the project root (with the ``inria_academy`` package importable —
e.g. inside the ``uv``-managed venv)::

    python docs/sphinx/scripts/regen_gallery.py

Optional arguments::

    --only 1 7      regenerate only the modules listed (here: 1 and 7)
    --dpi 150       override the default PNG DPI

The list of representative scripts is hard-coded below; edit
``REPRESENTATIVES`` to change which script of each module is featured in
the gallery.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Project root = the directory two levels above this file.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent      # scripts → sphinx → docs → root

# (module_number, script_relative_path, output_basename)
REPRESENTATIVES: list[tuple[int, str, str]] = [
    (1, "examples/1_Reminder_Fourier/spectral_filtering_demo.py",
        "module1_spectral_filtering"),
    (2, "examples/2_Reminder_Wavelets/03.benchmark_CWT_vs_Fourier_STFTGabor.py",
        "module2_cwt_vs_fourier"),
    (3, "examples/3_STFT_WVD_SPWVD/03.amfm_tf_comparison.py",
        "module3_amfm_tf"),
    (4, "examples/4_EMD_EEMD_CEEMDAN/05.scenario4_mode_mixing.py",
        "module4_mode_mixing"),
    (5, "examples/5_SWT/04.scenario3_chirp_tone_noise.py",
        "module5_sst_noise"),
    (6, "examples/6_VMD/04.vmd_ex4_synthetic.py",
        "module6_vmd_synthetic"),
    (7, "examples/7_MVMD/04.mvmd_ex4_alpha_eeg.py",
        "module7_mvmd_eeg"),
    (8, "examples/8_Complex_EMD_VMD/radar_iq_multivariate_benchmark.py",
        "module8_radar_iq_mv"),
]

GALLERY_DIR = PROJECT_ROOT / "docs" / "sphinx" / "_static" / "gallery"
WRAPPER     = HERE / "run_and_capture.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate every static image in the doc gallery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--only", type=int, nargs="+", metavar="N",
                   help="Only regenerate the listed modules (e.g. --only 1 7).")
    p.add_argument("--dpi", type=int, default=120,
                   help="DPI for the saved PNGs (default: 120).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    targets = REPRESENTATIVES
    if args.only:
        wanted = set(args.only)
        targets = [t for t in REPRESENTATIVES if t[0] in wanted]
        missing = wanted - {t[0] for t in REPRESENTATIVES}
        if missing:
            print(f"warning: unknown module number(s): "
                  f"{sorted(missing)}", file=sys.stderr)

    if not targets:
        print("nothing to do", file=sys.stderr)
        return 1

    # Make sure src/ is on PYTHONPATH for the child Python processes.
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if "PYTHONPATH" not in env
        else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    env["MPLBACKEND"] = "Agg"

    failures: list[int] = []
    for module_n, rel_script, stem in targets:
        script_path = PROJECT_ROOT / rel_script
        out_stem    = GALLERY_DIR / stem
        print()
        print(f"{'═' * 70}")
        print(f" Module {module_n}: {rel_script}")
        print(f"{'═' * 70}")

        if not script_path.is_file():
            print(f"  SKIP — script not found at {script_path}",
                  file=sys.stderr)
            failures.append(module_n)
            continue

        result = subprocess.run(
            [sys.executable, str(WRAPPER), str(script_path), str(out_stem),
             "--dpi", str(args.dpi)],
            env=env,
        )
        if result.returncode != 0:
            failures.append(module_n)

    print()
    print(f"{'═' * 70}")
    if failures:
        print(f" Done with {len(failures)} failure(s): "
              f"modules {failures}")
        print(f"{'═' * 70}")
        return 1
    print(f" Done — {len(targets)}/{len(targets)} modules regenerated.")
    print(f" Output: {GALLERY_DIR.relative_to(PROJECT_ROOT)}/")
    print(f"{'═' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
