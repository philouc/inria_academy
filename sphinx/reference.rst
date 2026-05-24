Example gallery — reference
============================

Annotated index of every demo in ``examples/``. Source files are on GitHub at
`philouc/inria_academy/tree/main/examples
<https://github.com/philouc/inria_academy/tree/main/examples>`_.

Helper modules live in the importable package :mod:`inria_academy.utils` and
are documented separately under :doc:`api/index`.

----

Module 1 — Reminder: Fourier
----------------------------

Refresher on the Fourier domain and linear filtering.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Topic
   * - ``spectral_filtering_demo.py``
     - Spectral filtering — the ``x(t) → FT → ×H(f) → IFT → y(t)`` pipeline,
       illustrated on a 5 Hz + 50 Hz signal with Gaussian noise and an ideal
       low-pass filter.


Module 2 — Reminder: Wavelets
-----------------------------

From mother wavelets and bases to long-memory processes and denoising.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``00.morlet_params.py``
     - Morlet wavelet — effect of the bandwidth and centre frequency parameters.
   * - ``01.wavelet_bases.py``
     - Mother wavelets and basis functions (Haar, Daubechies, Symlets, Coiflets).
   * - ``01.wavelet_mexican_hat.py``
     - Exercise solutions for script 1 — Mexican hat focus.
   * - ``02.time_frequency_tiling.py``
     - Time-frequency tiling and the Heisenberg uncertainty principle.
   * - ``02.time_frequency_tiling_exercise.py``
     - Exercise solutions for script 2.
   * - ``03.benchmark_CWT_vs_Fourier_STFTGabor.py``
     - Benchmark: CWT vs Fourier vs short-time Fourier (Gabor).
   * - ``04.long_memory.py``
     - Long-memory processes and Hurst exponent estimation.
   * - ``04.long_memory_sym4.py``
     - Same, using the Symlet 4 wavelet.
   * - ``04.long_memory_exercises.py``
     - Exercises 2 & 3 (sym4 version).
   * - ``05.wavelet_denoising.py``
     - Wavelet denoising — soft/hard thresholding with Symlet 4.
   * - ``05.wavelet_denoising_exercise.py``
     - Denoising exercises (Symlet 4).


Module 3 — STFT, WVD, SPWVD
---------------------------

Quadratic time-frequency representations and their resolution trade-offs.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``01.stft_vs_wvd_chirp.py``
     - STFT vs Wigner-Ville distribution on a linear chirp.
   * - ``02.wvd_cross_terms.py``
     - Multi-component WVD cross-terms — two parallel chirps.
   * - ``03.amfm_tf_comparison.py``
     - AM-FM benchmark — STFT vs WVD vs SPWVD.


Module 4 — EMD, EEMD, CEEMDAN
-----------------------------

Empirical Mode Decomposition and its noise-assisted variants.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``01.emd_sifting_inaction.py``
     - The sifting algorithm step by step on the AM-FM benchmark.
   * - ``02.emd_hht_closing.py``
     - EMD + Hilbert-Huang Transform, closing the loop on the AM-FM benchmark.
   * - ``03.spwvd_vs_hht.py``
     - SPWVD vs HHT — side-by-side comparison on the AM-FM benchmark.
   * - ``04.emd_successful_decomposition.py``
     - EMD in practice — a successful decomposition example.
   * - ``05.scenario4_mode_mixing.py``
     - EMD vs EEMD vs CEEMDAN on the canonical Wu & Huang mode-mixing test.


Module 5 — SWT / SST
--------------------

Synchrosqueezing wavelet transform, benchmarked against EMD-HHT.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``01.scenario1_chirp_tone.py``
     - Scenario 1 — chirp + tone, noiseless.
   * - ``02.scenario2_close_frequencies.py``
     - Scenario 2 — two close frequencies.
   * - ``03.scenario2b_four_methods.py``
     - Close frequencies — SST vs the full EMD family.
   * - ``04.scenario3_chirp_tone_noise.py``
     - Scenario 3 — chirp + tone in noise.
   * - ``05.scenario3b_eemd_ceemdan.py``
     - Scenario 3b — noise-robust EMD variants under SST comparison.


Module 6 — VMD
--------------

Variational Mode Decomposition (Dragomiretskiy & Zosso 2014).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``00.vmd_def2_bw_examples.py``
     - Bandwidth definition illustrated (Dragomiretskiy & Zosso 2014, Fig. 2).
   * - ``01.vmd_ex1_tone_detection.py``
     - Example 1 — pure tone detection across the spectrum.
   * - ``02.scenario2c_vmd_close_freq.py``
     - Close frequencies — SST + EMD family + VMD (resolved).
   * - ``03.vmd_ex3_noisy_triharmonic.py``
     - Example 3 — noise robustness on a tri-harmonic.
   * - ``04.vmd_ex4_synthetic.py``
     - Example 4 — synthetic signal (Dragomiretskiy & Zosso 2014, Fig. 8).
   * - ``04.vmd_ex4b_compare_classical.py``
     - Example 4 continued — classical methods on the same signal.
   * - ``05.vmd_ex5_ecg.py``
     - Example 5 — real-world ECG decomposition.


Module 7 — MVMD
---------------

Multivariate VMD on multichannel signals (ur Rehman & Aftab 2019).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Script
     - Topic
   * - ``01.mvmd_ex1_alignment.py``
     - Example 1 — mode alignment across channels.
   * - ``02.mvmd_ex2_filterbank.py``
     - Example 2 — filterbank behaviour on white Gaussian noise.
   * - ``03.mvmd_ex3_orthogonality.py``
     - Example 3 — quasi-orthogonality of recovered modes.
   * - ``04.mvmd_ex4_alpha_eeg.py``
     - Example 4 — α rhythms in 4-channel EEG.
   * - ``05.mvmd_ex5_ctg.py``
     - Example 5 — cardiotocography (FHR + UC).


Module 8 — Complex EMD / VMD
----------------------------

Complex-valued and analytic-signal extensions of EMD and VMD (Doppler / IQ).

----

References
----------

* Dragomiretskiy, K. & Zosso, D. (2014). *Variational Mode Decomposition.*
  IEEE Transactions on Signal Processing, 62(3), 531–544.
* ur Rehman, N. & Aftab, H. (2019). *Multivariate Variational Mode
  Decomposition.* IEEE Transactions on Signal Processing.
* Wu, Z. & Huang, N. E. (2009). *Ensemble Empirical Mode Decomposition: A
  Noise-Assisted Data Analysis Method.* Advances in Adaptive Data Analysis,
  1(1), 1–41.
* Daubechies, I., Lu, J. & Wu, H.-T. (2011). *Synchrosqueezed wavelet
  transforms: An empirical mode decomposition-like tool.* Applied and
  Computational Harmonic Analysis, 30(2), 243–261.
* Akaike, H. (1974). IEEE Trans. Aut. Control 19(6), 716–723.
* Schwarz, G. (1978). *Estimating the dimension of a model.* Annals of
  Statistics 6(2), 461–464.
* Bandt, C. & Pompe, B. (2002). *Permutation entropy: a natural complexity
  measure for time series.* Phys. Rev. Lett. 88(17), 174102.
* Satopaa, V. et al. (2011). *Finding a kneedle in a haystack.* ICDCSW.
