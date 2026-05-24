Usage guide
===========

Running a script
----------------

Each script in ``examples/`` is fully self-contained — pick one, run it, get
the figure.

.. code-block:: bash

   uv run python examples/1_Reminder_Fourier/spectral_filtering_demo.py
   uv run python examples/2_Reminder_Wavelets/00.morlet_params.py
   uv run python examples/7_MVMD/04.mvmd_ex4_alpha_eeg.py

Most scripts open a matplotlib window. If you are on a headless server, set a
non-interactive backend before running:

.. code-block:: bash

   export MPLBACKEND=Agg


Helper modules
--------------

Three modules under :mod:`inria_academy.utils` contain reusable building
blocks rather than runnable demos:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Role
   * - :mod:`inria_academy.utils.mvmd`
     - Pure-Python reference implementation of Multivariate Variational Mode
       Decomposition (ur Rehman & Aftab 2019), used by every example in
       ``examples/7_MVMD/``.
   * - :mod:`inria_academy.utils.swt_emd`
     - Shared computation and plotting helpers (SST, EMD-HHT, 5-panel
       comparison figure) used by every scenario in ``examples/5_SWT/``
       and ``examples/6_VMD/``.
   * - :mod:`inria_academy.utils.signals`
     - Synthetic signal generators (close frequencies, chirp + tone + AWGN,
       …) shared across scenarios.
   * - :mod:`inria_academy.utils.k_selection`
     - Information criteria, knee detection, permutation entropy and
       multi-init averaging for choosing *K* in VMD / MVMD.

These are imported by the demo scripts in their own directory; they are not
meant to be executed directly.


Suggested learning path
-----------------------

1. Run **module 1** first to refresh the Fourier-domain filtering pipeline.
2. Move through **modules 2 → 4** in order: each builds on the
   time-frequency tools introduced previously.
3. **Modules 5, 6, 7** can be approached as three parallel methods solving
   the same family of problems (mode decomposition of non-stationary
   signals); the side-by-side comparison scripts in module 5
   (``03.scenario2b_*``, ``05.scenario3b_*``) and module 6
   (``02.scenario2c_*``) are designed specifically for that purpose.
4. **Module 8** revisits the same machinery in the complex / analytic-signal
   setting (Doppler, IQ data).


Troubleshooting
---------------

.. dropdown:: ``ModuleNotFoundError: No module named 'PyEMD'``
   :icon: alert

   The package on PyPI is ``EMD-signal``, **not** ``PyEMD`` — the import name
   differs from the package name. ``uv sync`` already installs it correctly;
   this only fails if you bypass ``uv``.

.. dropdown:: Matplotlib windows do not appear on macOS
   :icon: alert

   Install a GUI backend (``pip install PyQt6`` for instance), or save
   figures with ``plt.savefig(...)`` instead of ``plt.show()``.

.. dropdown:: Long runtime on VMD / EEMD scripts
   :icon: clock

   These methods are iterative. A 500-iteration ADMM solve on a multichannel
   signal can take 30–60 s, which is normal.

.. dropdown:: Autodoc warns about missing imports during the doc build
   :icon: book

   The Sphinx config mocks ``PyEMD``, ``ssqueezepy``, ``vmdpy``, ``neurokit2``
   and ``pymultifracs`` via ``autodoc_mock_imports``. If you add a new
   third-party dependency that is not installed in the doc-build environment
   and the build complains, add it to that list in ``docs/sphinx/conf.py``.
