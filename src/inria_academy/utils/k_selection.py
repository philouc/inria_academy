"""K selection for VMD and MVMD: information criteria, knee detection,
permutation entropy, and multi-init averaging.

Three classical information criteria

.. math::

   \\mathrm{AIC}(K) &= N \\log \\widehat{\\sigma}^2 + 2 K \\\\
   \\mathrm{BIC}(K) &= N \\log \\widehat{\\sigma}^2 + K \\log N \\\\
   \\mathrm{HQ}(K)  &= N \\log \\widehat{\\sigma}^2 + 2 K \\log \\log N

(Akaike 1974, Schwarz 1978, Hannan & Quinn 1979) tend to over-select K
for VMD: their argmin keeps drifting to K_max because the linear K
penalty is too weak relative to the residual term. This module
therefore complements them with four practical alternatives:

- a simple **log-MSE elbow** detector (largest single-step drop),
- the **Kneedle** algorithm (Satopaa et al. 2011) — industry-standard
  knee finder using normalised curve geometry,
- a **modified BIC** with quadratic penalty :math:`K^2 \\log N`,
- a **permutation-entropy** rule (Bandt & Pompe 2002) that flags
  noise-like modes on a per-mode basis.

Multi-init averaging is also available via the ``n_init`` parameter:
running VMD/MVMD with several initialisation strategies and averaging
the residual variance (using a robust median) stabilises the criteria
against optimisation local minima.

References

- Akaike, H. (1974). IEEE Trans. Aut. Control 19(6), 716–723.
- Schwarz, G. (1978). Annals of Statistics 6(2), 461–464.
- Hannan, E. J. & Quinn, B. G. (1979). JRSS B 41(2), 190–195.
- Bandt, C. & Pompe, B. (2002). Phys. Rev. Lett. 88(17), 174102.
- Satopaa, V. et al. (2011). *Finding a kneedle in a haystack*. ICDCSW.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np
from vmdpy import VMD

from inria_academy.utils.mvmd import MVMD


__all__ = [
    "select_k_vmd",
    "select_k_mvmd",
    "elbow_k",
    "kneedle",
    "permutation_entropy",
    "mode_pe",
    "plot_criteria",
    "plot_pe",
]


# ─── Information criteria ───────────────────────────────────────────────────

def _info_criteria(mse: float, n_eff: int, k: int) -> dict:
    """AIC, BIC, HQ and a modified BIC with quadratic penalty."""
    base = n_eff * np.log(max(mse, 1e-300))
    return {
        "aic":  float(base + 2.0 * k),
        "bic":  float(base + k * np.log(n_eff)),
        "hq":   float(base + 2.0 * k * np.log(np.log(max(n_eff, 3)))),
        "bic2": float(base + (k ** 2) * np.log(n_eff)),
    }


# ─── Elbow detection (simple) ───────────────────────────────────────────────

def elbow_k(K: np.ndarray, mse: np.ndarray) -> int:
    """K that maximises the marginal log-MSE drop.

    .. math::
       K^* = \\arg\\max_{K \\geq K_\\min + 1} \\big(
              \\log\\mathrm{MSE}(K-1) - \\log\\mathrm{MSE}(K) \\big)

    This is the strongest single-step "knee" — a parameter-free proxy
    for the K beyond which adding modes yields only marginal returns.
    """
    K = np.asarray(K)
    mse = np.asarray(mse)
    if len(K) < 2:
        return int(K[0])
    drops = np.log(mse[:-1]) - np.log(mse[1:])
    return int(K[1 + int(np.argmax(drops))])


# ─── Kneedle algorithm (Satopaa et al. 2011) ────────────────────────────────

def kneedle(
    x: np.ndarray,
    y: np.ndarray,
    curve: str = "convex",
    direction: str = "decreasing",
) -> int:
    """Find the knee of a curve using the Kneedle algorithm.

    Reference: Satopaa, V., Albrecht, J., Irwin, D. & Raghavan, B. (2011).
    *Finding a "Kneedle" in a Haystack: Detecting Knee Points in System
    Behavior.* ICDCSW.

    The algorithm normalises both axes to [0, 1] and computes the
    *difference curve* between the data and the diagonal that joins the
    two endpoints. The knee is the point of greatest distance from
    that diagonal — geometrically, the point of maximum curvature.

    Parameters
    ----------
    x : 1D array-like
        Strictly monotonic x-values (typically the K values).
    y : 1D array-like
        Corresponding y-values (e.g. ``log(MSE(K))``). Must have the
        same shape as ``x``.
    curve : {"convex", "concave"}
        Shape of the curve. For VMD's ``log(MSE)`` vs ``K``, the curve
        is **convex** (drops fast, then plateaus).
    direction : {"increasing", "decreasing"}
        Direction of y as x grows. For VMD's residual curve,
        ``"decreasing"``.

    Returns
    -------
    knee_x : int or float
        The x-value of the knee point (taken from the original ``x``
        array, not the normalised version).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.size < 3:
        return x[int(np.argmin(y) if direction == "decreasing" else np.argmax(y))]

    # Normalise to [0, 1]
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # Compute the difference between the curve and the appropriate
    # diagonal. The knee is at the maximum of this difference, with the
    # sign chosen so "larger = more curved" in every case.
    if direction == "decreasing":
        # Diagonal joins (0, 1) → (1, 0). Distance below the diagonal
        # is (1 - x_n) - y_n. Convex curve sits below diagonal; concave
        # curve sits above, so we flip the sign.
        d = (1 - x_n) - y_n
        if curve == "concave":
            d = -d
    else:  # increasing
        # Diagonal joins (0, 0) → (1, 1). Distance below diagonal is
        # x_n - y_n. Concave curve sits above (positive d_above = y - x).
        d = y_n - x_n
        if curve == "convex":
            d = -d

    return x[int(np.argmax(d))]


# ─── Permutation entropy (Bandt & Pompe 2002) ───────────────────────────────

def permutation_entropy(
    x: np.ndarray,
    m: int = 3,
    tau: int = 1,
    normalise: bool = True,
) -> float:
    """Compute the permutation entropy of a 1-D signal.

    Reference: Bandt, C. & Pompe, B. (2002). *Permutation Entropy: a
    Natural Complexity Measure for Time Series.* Physical Review
    Letters 88(17), 174102.

    The signal is embedded as overlapping vectors of length ``m`` (with
    step ``tau``). Each vector is replaced by the *ordinal pattern*
    (the permutation that sorts it). The entropy of the empirical
    distribution over the ``m!`` possible patterns is the permutation
    entropy.

    Parameters
    ----------
    x : 1D array-like
        Input signal.
    m : int, default 3
        Embedding (pattern) dimension. Typical values 3 – 7.
        With m = 3 there are 6 patterns; with m = 5 there are 120.
        Higher m gives finer discrimination but needs more samples.
    tau : int, default 1
        Embedding delay (step between consecutive samples of each
        pattern vector).
    normalise : bool, default True
        If True, divide by ``log(m!)`` so the result lies in [0, 1].
        0 = perfectly structured (e.g. a monotonic ramp);
        1 = maximally disordered (e.g. white noise).

    Returns
    -------
    H : float
        Permutation entropy, normalised to [0, 1] by default.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    L = N - (m - 1) * tau
    if L < 2:
        raise ValueError(f"Series too short for m={m}, tau={tau}: need "
                         f"at least {(m - 1) * tau + 2} samples, got {N}.")
    # Build the embedding matrix of shape (L, m)
    indices = np.arange(0, m * tau, tau)
    rows = np.arange(L)[:, None] + indices[None, :]
    emb = x[rows]
    # For each row, get the rank pattern (argsort gives permutation)
    patterns = np.argsort(emb, axis=1, kind="quicksort")
    # Count unique patterns
    counts = Counter(map(tuple, patterns))
    probs = np.fromiter(counts.values(), dtype=float, count=len(counts))
    probs /= probs.sum()
    H = -float(np.sum(probs * np.log(probs)))
    if normalise:
        H /= math.log(math.factorial(m))
    return H


def mode_pe(u: np.ndarray, m: int = 3, tau: int = 1) -> np.ndarray:
    """Permutation entropy of each mode in a VMD/MVMD decomposition.

    Parameters
    ----------
    u : ndarray
        VMD output of shape ``(K, N)`` *or* MVMD output of shape
        ``(K, C, N)``. For the multivariate case the PE is computed
        per channel and averaged across channels for each mode.
    m, tau
        Forwarded to :func:`permutation_entropy`.

    Returns
    -------
    pe : ndarray of shape (K,)
        Permutation entropy of each mode, in [0, 1].
    """
    if u.ndim == 2:
        K = u.shape[0]
        return np.array([permutation_entropy(u[k], m=m, tau=tau) for k in range(K)])
    elif u.ndim == 3:
        K, C, _ = u.shape
        out = np.empty(K)
        for k in range(K):
            out[k] = np.mean([permutation_entropy(u[k, c], m=m, tau=tau)
                              for c in range(C)])
        return out
    else:
        raise ValueError(f"u must be 2D or 3D, got shape {u.shape}")


def _k_pe_threshold(K_list, pe_max_per_K, threshold: float) -> int | None:
    """Largest K such that every K' <= K has max-mode PE below threshold.

    Returns None if no such K exists (first K violates already).
    """
    chosen = None
    for K, pe_max in zip(K_list, pe_max_per_K):
        if pe_max < threshold:
            chosen = K
        else:
            break
    return chosen


# ─── Multi-init helper ──────────────────────────────────────────────────────

def _vmd_inits(n_init: int) -> list[int]:
    """Distinct VMD init values to try when averaging over inits."""
    # vmdpy supports init in {0, 1, 2}; we cycle through them
    base = [1, 2, 0]
    return [base[i % 3] for i in range(n_init)]


def _run_vmd_multi(x, K, alpha, tau, DC, tol, n_init):
    """Run VMD n_init times with varying init; return the run whose MSE
    is closest to the median, plus the median MSE itself."""
    mses, runs = [], []
    for init_val in _vmd_inits(n_init):
        u, _, om = VMD(x, alpha=alpha, tau=tau, K=K, DC=DC,
                       init=init_val, tol=tol)
        recon = u.sum(axis=0)
        m = min(recon.size, x.size)
        residual = x[:m] - recon[:m]
        mses.append(float(np.mean(residual ** 2)))
        runs.append((u, om, m))
    mses_arr = np.array(mses)
    median_mse = float(np.median(mses_arr))
    # Pick the run closest to the median to use for downstream PE computation
    idx = int(np.argmin(np.abs(mses_arr - median_mse)))
    u, om, m = runs[idx]
    return median_mse, m, u, om


def _run_mvmd_multi(X, K, alpha, tau, DC, tol, n_init):
    """Same idea for MVMD."""
    mses, runs = [], []
    for init_val in _vmd_inits(n_init):
        u, _, om = MVMD(X, alpha=alpha, tau=tau, K=K, DC=DC,
                        init=init_val, tol=tol)
        recon = u.sum(axis=0)
        m = min(recon.shape[1], X.shape[1])
        residual = X[:, :m] - recon[:, :m]
        mses.append(float(np.mean(residual ** 2)))
        runs.append((u, om, m))
    mses_arr = np.array(mses)
    median_mse = float(np.median(mses_arr))
    idx = int(np.argmin(np.abs(mses_arr - median_mse)))
    u, om, m = runs[idx]
    return median_mse, m, u, om


# ─── univariate (VMD) ───────────────────────────────────────────────────────

def select_k_vmd(
    x: np.ndarray,
    K_range: Iterable[int],
    alpha: float = 2000.0,
    tau: float = 0.0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
    n_init: int = 1,
    pe_m: int = 3,
    pe_tau: int = 1,
    pe_threshold: float = 0.6,
    verbose: bool = False,
) -> dict:
    """Run VMD for each K in ``K_range`` and compute selection diagnostics.

    Parameters
    ----------
    x : ndarray of shape (N,)
        Univariate signal.
    K_range : iterable of int
        Candidate values of K to evaluate.
    alpha, tau, DC, tol
        Forwarded to :func:`vmdpy.VMD`.
    init : int, default 1
        Initialisation flag used when ``n_init = 1``. Ignored otherwise.
    n_init : int, default 1
        If > 1, runs VMD with several distinct ``init`` values and uses
        the median residual MSE across runs. Stabilises the criteria
        against ADMM local minima but multiplies runtime by ``n_init``.
    pe_m, pe_tau : int
        Embedding parameters for the permutation entropy of each mode
        (Bandt & Pompe 2002). Defaults ``m = 3``, ``tau = 1``.
    pe_threshold : float, default 0.6
        Threshold above which a mode is considered noise-like. The
        suggested ``k_pe`` is the largest K such that *every* mode at
        that K has PE below the threshold.
    verbose : bool, default False
        Print one progress line per K.

    Returns
    -------
    results : dict with keys
        - ``"K"`` : ndarray of K values tried
        - ``"mse"`` : ndarray, residual MSE for each K
        - ``"aic"``, ``"bic"``, ``"hq"``, ``"bic2"`` : criteria
        - ``"pe_max"`` : ndarray, max permutation entropy across modes per K
        - ``"pe_per_K"`` : list of arrays — PE of every mode at each K
        - ``"k_aic"``, ``"k_bic"``, ``"k_hq"``, ``"k_bic2"`` : argmin of each
        - ``"k_elbow"`` : K from simple log-MSE marginal drop
        - ``"k_kneedle"`` : K from Kneedle (Satopaa 2011)
        - ``"k_pe"`` : K from PE threshold rule (or None)
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    K_list = list(K_range)
    rows = []
    pe_per_K = []
    for K in K_list:
        if n_init > 1:
            mse, m, u, _ = _run_vmd_multi(x, K, alpha, tau, DC, tol, n_init)
        else:
            u, _, _ = VMD(x, alpha=alpha, tau=tau, K=K, DC=DC,
                          init=init, tol=tol)
            recon = u.sum(axis=0)
            m = min(recon.size, N)
            residual = x[:m] - recon[:m]
            mse = float(np.mean(residual ** 2))
        # Permutation entropy of every mode at this K
        pe = mode_pe(u, m=pe_m, tau=pe_tau)
        pe_per_K.append(pe)
        crit = _info_criteria(mse, m, K)
        crit["K"] = K
        crit["mse"] = mse
        crit["pe_max"] = float(pe.max())
        rows.append(crit)
        if verbose:
            print(f"  K={K:2d}   MSE={mse:.4e}   "
                  f"AIC={crit['aic']:8.1f}   BIC={crit['bic']:8.1f}   "
                  f"HQ={crit['hq']:8.1f}   BIC*={crit['bic2']:8.1f}   "
                  f"PE_max={crit['pe_max']:.3f}")
    return _pack(rows, pe_per_K, pe_threshold)


# ─── multivariate (MVMD) ────────────────────────────────────────────────────

def select_k_mvmd(
    X: np.ndarray,
    K_range: Iterable[int],
    alpha: float = 2000.0,
    tau: float = 0.0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
    n_init: int = 1,
    pe_m: int = 3,
    pe_tau: int = 1,
    pe_threshold: float = 0.6,
    verbose: bool = False,
) -> dict:
    """Same as :func:`select_k_vmd` for multichannel signals.

    Parameters
    ----------
    X : ndarray of shape (C, N)
        Multichannel signal.

    Notes
    -----
    The effective sample size used in the criteria is ``N_eff = C * N``.
    PE is computed per channel and averaged across channels for each
    mode before applying the threshold rule.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (C, N); got shape {X.shape}")
    C, N = X.shape
    K_list = list(K_range)
    rows = []
    pe_per_K = []
    for K in K_list:
        if n_init > 1:
            mse, m, u, _ = _run_mvmd_multi(X, K, alpha, tau, DC, tol, n_init)
        else:
            u, _, _ = MVMD(X, alpha=alpha, tau=tau, K=K, DC=DC,
                           init=init, tol=tol)
            recon = u.sum(axis=0)
            m = min(recon.shape[1], N)
            residual = X[:, :m] - recon[:, :m]
            mse = float(np.mean(residual ** 2))
        n_eff = C * m
        pe = mode_pe(u, m=pe_m, tau=pe_tau)
        pe_per_K.append(pe)
        crit = _info_criteria(mse, n_eff, K)
        crit["K"] = K
        crit["mse"] = mse
        crit["pe_max"] = float(pe.max())
        rows.append(crit)
        if verbose:
            print(f"  K={K:2d}   MSE={mse:.4e}   "
                  f"AIC={crit['aic']:8.1f}   BIC={crit['bic']:8.1f}   "
                  f"HQ={crit['hq']:8.1f}   BIC*={crit['bic2']:8.1f}   "
                  f"PE_max={crit['pe_max']:.3f}")
    return _pack(rows, pe_per_K, pe_threshold)


# ─── packaging helper ───────────────────────────────────────────────────────

def _pack(rows, pe_per_K, pe_threshold):
    K_arr   = np.array([r["K"]      for r in rows])
    mse_arr = np.array([r["mse"]    for r in rows])
    aic     = np.array([r["aic"]    for r in rows])
    bic     = np.array([r["bic"]    for r in rows])
    hq      = np.array([r["hq"]     for r in rows])
    bic2    = np.array([r["bic2"]   for r in rows])
    pe_max  = np.array([r["pe_max"] for r in rows])
    return {
        "K":         K_arr,
        "mse":       mse_arr,
        "aic":       aic,
        "bic":       bic,
        "hq":        hq,
        "bic2":      bic2,
        "pe_max":    pe_max,
        "pe_per_K":  pe_per_K,
        "k_aic":     int(K_arr[np.argmin(aic)]),
        "k_bic":     int(K_arr[np.argmin(bic)]),
        "k_hq":      int(K_arr[np.argmin(hq)]),
        "k_bic2":    int(K_arr[np.argmin(bic2)]),
        "k_elbow":   elbow_k(K_arr, mse_arr),
        "k_kneedle": int(kneedle(K_arr.astype(float), np.log(mse_arr),
                                  curve="convex", direction="decreasing")),
        "k_pe":      _k_pe_threshold(K_arr.tolist(), pe_max, pe_threshold),
        "pe_threshold": float(pe_threshold),
    }


# ─── plotting helpers ───────────────────────────────────────────────────────

def plot_criteria(results: dict, axes=None, title: str = ""):
    """Plot the diagnostic curves for K selection (3 panels).

    Panels:
    1. Residual MSE on log axis, with elbow and Kneedle marked
    2. Classical AIC / BIC / HQ
    3. Modified BIC (quadratic penalty)
    """
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=150)
    else:
        fig = axes[0].figure

    K = results["K"]

    # Panel 1: MSE + elbow + kneedle
    ax = axes[0]
    ax.semilogy(K, results["mse"], "o-", color="#0b1f3a", lw=1.4, ms=5)
    ax.axvline(results["k_elbow"], color="#7e3a93", ls="--", lw=1.3,
               label=f"elbow  (K* = {results['k_elbow']})")
    ax.axvline(results["k_kneedle"], color="#e08a1f", ls=":", lw=1.3,
               label=f"Kneedle  (K* = {results['k_kneedle']})")
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("residual MSE  (log)", fontsize=9)
    ax.set_xticks(K); ax.tick_params(labelsize=8)
    ax.grid(True, which="both", ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_title("(a)  Residual variance  +  knee detectors",
                 loc="left", fontsize=9.5, color="#0b1f3a",
                 fontweight="bold", pad=4)

    # Panel 2: Classical AIC / BIC / HQ
    ax = axes[1]
    for name, key, k_key, color, marker in (
        ("AIC", "aic", "k_aic", "#c9191e", "o"),
        ("BIC", "bic", "k_bic", "#0b1f3a", "s"),
        ("HQ",  "hq",  "k_hq",  "#1f8a4c", "^"),
    ):
        ax.plot(K, results[key], marker=marker, color=color, lw=1.3, ms=5,
                label=f"{name}   (K* = {results[k_key]})")
        ax.axvline(results[k_key], color=color, ls=":", lw=0.7, alpha=0.5)
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("info. criterion", fontsize=9)
    ax.set_xticks(K); ax.tick_params(labelsize=8)
    ax.grid(True, ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_title("(b)  Classical AIC / BIC / HQ\n"
                 "       (over-select K for VMD)",
                 loc="left", fontsize=9.5, color="#0b1f3a",
                 fontweight="bold", pad=4)

    # Panel 3: Modified BIC
    ax = axes[2]
    ax.plot(K, results["bic2"], "D-", color="#e08a1f", lw=1.4, ms=5,
            label=f"BIC*   (K* = {results['k_bic2']})")
    ax.axvline(results["k_bic2"], color="#e08a1f", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel(r"$N \log\, \mathrm{MSE} + K^2 \log N$", fontsize=9)
    ax.set_xticks(K); ax.tick_params(labelsize=8)
    ax.grid(True, ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    ax.set_title("(c)  Modified BIC  (quadratic penalty)",
                 loc="left", fontsize=9.5, color="#0b1f3a",
                 fontweight="bold", pad=4)

    if title:
        fig.suptitle(title, fontsize=10.5, color="#0b1f3a",
                     fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


def plot_pe(results: dict, ax=None, title: str = ""):
    """Plot the per-mode permutation entropy as a function of K.

    Shows a small bar per mode at each K, plus the PE-threshold line.
    Modes with PE above the threshold are considered noise-like.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6), dpi=150)

    K = results["K"]
    pe_per_K = results["pe_per_K"]
    threshold = results["pe_threshold"]

    # For each K, scatter the PE values of every mode
    for ki, k in enumerate(K):
        pe = pe_per_K[ki]
        # x-offsets so dots are spread within a small interval around k
        offsets = np.linspace(-0.18, 0.18, len(pe))
        colors = ["#1f8a4c" if p < threshold else "#c9191e" for p in pe]
        ax.scatter(np.full_like(pe, k, dtype=float) + offsets, pe,
                   c=colors, s=22, zorder=3, edgecolor="white", linewidth=0.5)

    ax.axhline(threshold, color="#0b1f3a", ls="--", lw=1.0,
               label=f"PE threshold = {threshold}")
    ax.set_xticks(K)
    ax.set_xlabel("K", fontsize=9)
    ax.set_ylabel("permutation entropy  (per mode)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    ax.grid(True, ls="-", lw=0.2, color="#dddddd")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    sub = (f"k_pe = {results['k_pe']}"
           if results["k_pe"] is not None
           else "k_pe = none (all K violate threshold)")
    ax.set_title(f"Permutation entropy per mode   ·   {sub}",
                 loc="left", fontsize=9.5, color="#0b1f3a",
                 fontweight="bold", pad=4)

    if title:
        ax.figure.suptitle(title, fontsize=10.5, color="#0b1f3a",
                           fontweight="bold", y=1.02)
    return ax
