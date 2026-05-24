Changelog
=========

Version 0.1.0
-------------

* Initial release of the ``inria_academy`` training material.
* Eight modules of example scripts covering Fourier and wavelet foundations,
  STFT / WVD / SPWVD, the EMD family, SST, VMD, MVMD and complex-valued
  EMD / VMD extensions.
* Importable helpers in :mod:`inria_academy.utils`:

  * :func:`~inria_academy.utils.mvmd.MVMD` — pure-Python reference
    implementation of Multivariate Variational Mode Decomposition;
  * :mod:`~inria_academy.utils.signals` — synthetic signal generators;
  * :mod:`~inria_academy.utils.swt_emd` — SST + EMD-HHT helpers and
    5-panel comparison figure;
  * :mod:`~inria_academy.utils.k_selection` — information criteria, knee
    detection, permutation entropy and multi-init averaging for
    *K*-selection in VMD / MVMD.
