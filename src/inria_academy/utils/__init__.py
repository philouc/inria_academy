"""Reusable helpers for the Inria Academy example gallery.

Three helper modules live here:

- :mod:`inria_academy.utils.mvmd` — pure-Python reference implementation
  of Multivariate Variational Mode Decomposition (ur Rehman & Aftab 2019),
  used by the examples in ``examples/7_MVMD/``.
- :mod:`inria_academy.utils.swt_emd` — shared plotting and computation
  helpers for the SST vs EMD-HHT comparison scenarios, used by the
  examples in ``examples/5_SWT/`` and ``examples/6_VMD/``.
- :mod:`inria_academy.utils.signals` — synthetic signal generators
  (close frequencies, chirp + tone + AWGN, ...) shared across scenarios.
"""

from inria_academy.utils.mvmd import MVMD
from inria_academy.utils import signals, swt_emd

__all__ = ["MVMD", "signals", "swt_emd"]
