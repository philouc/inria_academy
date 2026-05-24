"""
Multivariate Variational Mode Decomposition  (MVMD)
====================================================

Reference:
    ur Rehman, N. & Aftab, H. (2019).  Multivariate Variational Mode
    Decomposition.  IEEE Transactions on Signal Processing.

Given a multivariate signal  x(t) = [x_1(t), ..., x_C(t)]^T  with C
channels, MVMD finds K modes such that:

    -  each mode  u_k(t) = [u_{k,1}(t), ..., u_{k,C}(t)]^T  is a
       multichannel signal;
    -  all channels of mode k share the same instantaneous frequency
       ω_k(t) (mode alignment);
    -  the sum of modes reconstructs each channel:
       Σ_k u_{k,c}(t) = x_c(t)  for each c.

The variational problem and its ADMM solution are direct extensions of
univariate VMD.  The key difference is the frequency update::

    ω_k^{n+1}  =  Σ_c ∫ ω |û_{k,c}(ω)|² dω
                  ──────────────────────────
                  Σ_c ∫ |û_{k,c}(ω)|² dω

i.e. the centre frequency is a power-weighted centroid AGGREGATED
across all channels — which is what enforces mode alignment.

This implementation follows the same conventions as Vinícius Rezende
Carvalho's vmdpy port of Dragomiretskiy's MATLAB code.
"""
import numpy as np


def MVMD(signal, alpha=2000.0, tau=0.0, K=3, DC=0, init=1, tol=1e-7,
         max_iter=500):
    """
    Parameters
    ----------
    signal : array of shape (C, T)
        Multichannel input — C channels, T samples each.
    alpha : float
        Bandwidth penalty (larger → tighter band per mode).
    tau : float
        Dual ascent step (0 → noise-removing variant; equality
        constraint not enforced).
    K : int
        Number of modes.
    DC : 0 or 1
        If 1, first mode is locked at ω = 0 (use for signals with trend).
    init : 0, 1 or 2
        Initialisation of {ω_k}:  0 = all at 0,  1 = uniform on [0, fs/2],
        2 = random uniform on [0, fs/2].
    tol : float
        Convergence tolerance on Σ_k ‖û_k^{n+1} - û_k^n‖² / ‖û_k^n‖².
    max_iter : int
        Hard cap on ADMM iterations.

    Returns
    -------
    u : array of shape (K, C, T)
        Recovered modes — u[k, c, :] is mode k on channel c.
    u_hat : array of shape (K, C, T)
        Per-mode per-channel spectra (full-length, complex).
    omega : array of shape (n_iter, K)
        Centre frequencies (in cycles/sample) over iterations.
    """
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError("signal must be (C, T)")
    C, T = signal.shape
    if T < 4:
        raise ValueError("signal too short")

    # --- Make even length and mirror-extend to mitigate boundary effects ---
    save_T = T
    if T % 2 != 0:
        signal = signal[:, :-1]
        T = T - 1
    # Mirror-extension: build f of length 2T per channel
    f_mirror = np.empty((C, 2 * T))
    f_mirror[:, : T // 2] = signal[:, T // 2 - 1::-1]
    f_mirror[:, T // 2 : 3 * T // 2] = signal
    f_mirror[:, 3 * T // 2 :] = signal[:, : T // 2 - 1: -1][:, :T // 2]
    # The above is robust enough; if it mis-sizes for odd T edges,
    # just trim/pad after the loop.
    T_ext = f_mirror.shape[1]
    # Normalised frequency grid in [0, 1):
    fs = 1.0
    freqs = (np.arange(T_ext) / T_ext) - 0.5
    # Shift so DC is at index 0:
    # actually we'll keep symmetric grid centred on 0 and use fftshift later

    # Per-channel Fourier transform of the mirrored signal
    # We keep only the positive-frequency side (one-sided) to align with
    # Carvalho's convention.
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror, axis=1), axes=1)
    # Zero the negative-frequency half (we work with one-sided)
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[:, : T_ext // 2] = 0.0

    # --- Initialise omegas ---
    omega_plus = np.zeros((max_iter, K))
    if init == 0:
        omega_plus[0] = 0.0
    elif init == 1:
        omega_plus[0] = (np.arange(K) + 1) * (0.5 / (K + 1))
    elif init == 2:
        omega_plus[0] = np.sort(np.random.rand(K) * 0.5)
    else:
        raise ValueError("init must be 0, 1 or 2")
    if DC == 1:
        omega_plus[0, 0] = 0.0

    # --- Initialise dual lambda (per channel) ---
    lambda_hat = np.zeros((max_iter, C, T_ext), dtype=complex)

    # --- Mode spectra (per mode, per channel) ---
    u_hat_plus = np.zeros((max_iter, K, C, T_ext), dtype=complex)

    # --- Main ADMM loop ---
    n = 0
    uDiff = tol + np.finfo(float).eps
    eps = np.finfo(float).eps
    while uDiff > tol and n < max_iter - 1:
        # Sum over k of previous u_hat: shape (C, T_ext)
        sum_uk = u_hat_plus[n, :, :, :].sum(axis=0)
        # --- Update each mode (with sequential update, GS-style) ---
        for k in range(K):
            # Remove mode k from the residual
            # residual^c = f_hat^c - sum_{i != k} u_hat_i^c + lambda^c / 2
            # which is: f_hat - (sum_uk - u_hat_k_prev) + lambda/2
            residual = (f_hat_plus
                        - (sum_uk - u_hat_plus[n, k, :, :])
                        + lambda_hat[n, :, :] / 2.0)
            # Wiener filter: 1 / (1 + 2α (ω - ω_k)²)
            wiener = 1.0 / (1.0 + 2.0 * alpha
                            * (freqs - omega_plus[n, k]) ** 2)
            # Broadcast over channels
            u_new = residual * wiener[None, :]
            u_hat_plus[n + 1, k, :, :] = u_new
            # Update the running sum: replace old u_hat_k with new
            sum_uk = sum_uk - u_hat_plus[n, k, :, :] + u_new

            # --- Update ω_k AGGREGATED across channels ---
            if k == 0 and DC == 1:
                # Lock at 0
                omega_plus[n + 1, k] = 0.0
            else:
                # Use one-sided positive-freq slice
                # Indices for positive freq
                # In our shifted grid, positive freq is indices > T_ext//2 - 1
                pos = freqs > 0
                pos_freqs = freqs[pos]
                # Aggregate spectral power across channels
                power = np.sum(np.abs(u_new[:, pos]) ** 2, axis=0)  # over C
                num = np.dot(pos_freqs, power)
                den = np.sum(power)
                omega_plus[n + 1, k] = num / (den + eps)

        # --- Dual update (per channel) ---
        # λ^{n+1} = λ^n + τ (f - Σ_k u_k)
        sum_uk_new = u_hat_plus[n + 1, :, :, :].sum(axis=0)
        lambda_hat[n + 1, :, :] = (lambda_hat[n, :, :]
                                    + tau * (f_hat_plus - sum_uk_new))

        # --- Convergence test ---
        n += 1
        uDiff = 0.0
        for k in range(K):
            diff = u_hat_plus[n, k] - u_hat_plus[n - 1, k]
            uDiff += np.sum(np.abs(diff) ** 2) / (
                np.sum(np.abs(u_hat_plus[n - 1, k]) ** 2) + eps)
        uDiff = abs(uDiff)

    n_iter = n + 1
    omega = omega_plus[:n_iter]
    # Sort modes by ascending centre frequency
    order = np.argsort(omega[-1])
    u_hat_final = u_hat_plus[n - 1, order]  # last iterate, sorted
    omega = omega[:, order]

    # --- Reconstruct symmetric spectrum and inverse-FFT to time domain ---
    # u_hat_final shape: (K, C, T_ext) — but only positive freq filled
    # Need to mirror the negative side for real-valued reconstruction
    u_hat_full = np.zeros_like(u_hat_final)
    u_hat_full[:, :, T_ext // 2:] = u_hat_final[:, :, T_ext // 2:]
    u_hat_full[:, :, 1: T_ext // 2 + 1] = np.conj(
        u_hat_final[:, :, T_ext // 2:][:, :, ::-1])
    # Inverse FFT
    u_time_ext = np.real(np.fft.ifft(
        np.fft.ifftshift(u_hat_full, axes=2), axis=2))
    # Trim mirror extension
    u_time = u_time_ext[:, :, T // 2 : 3 * T // 2]

    # If we had to drop the last sample, pad it back
    if save_T != T:
        last = u_time[:, :, -1:]
        u_time = np.concatenate([u_time, last], axis=2)

    return u_time, u_hat_full, omega


if __name__ == "__main__":
    # Quick sanity check: 2-channel signal with two shared tones
    np.random.seed(0)
    fs = 500.0
    t = np.arange(int(fs * 1.0)) / fs
    s1 = np.cos(2 * np.pi * 25 * t)
    s2 = np.cos(2 * np.pi * 60 * t)
    # Two channels — both contain both tones but with different amplitudes
    x1 = 1.0 * s1 + 0.5 * s2 + 0.05 * np.random.randn(len(t))
    x2 = 0.5 * s1 + 1.0 * s2 + 0.05 * np.random.randn(len(t))
    x = np.stack([x1, x2])
    print(f"Input: shape {x.shape}, fs={fs}")

    u, _, omega = MVMD(x, alpha=2000.0, tau=0., K=2, DC=0, init=1,
                       tol=1e-7)
    print(f"Output u: shape {u.shape}")
    print(f"Final ω_k = {omega[-1] * fs} Hz  (expected: 25, 60)")
    for k in range(u.shape[0]):
        for c in range(u.shape[1]):
            amp = np.std(u[k, c]) * np.sqrt(2)
            print(f"  mode {k+1} channel {c+1}:  amp ≈ {amp:.3f}")
