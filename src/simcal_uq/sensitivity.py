from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    variance: float
    n_base: int


def sobol_jansen(
    func: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    *,
    n_base: int = 256,
    seed: int = 321,
) -> SobolResult:
    """Estimate first/total Sobol indices with Jansen estimators and scrambled QMC.

    Inputs are assumed mutually independent and uniformly distributed over `bounds`.
    `n_base` must be a power of two so SciPy's Sobol balance properties are retained.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (d, 2)")
    if n_base < 8 or n_base & (n_base - 1):
        raise ValueError("n_base must be a power of two and >= 8")
    d = bounds.shape[0]
    m = int(np.log2(n_base))
    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    unit = sampler.random_base2(m)
    a_u, b_u = unit[:, :d], unit[:, d:]
    lower, upper = bounds[:, 0], bounds[:, 1]
    a = qmc.scale(a_u, lower, upper)
    b = qmc.scale(b_u, lower, upper)

    f_a = np.array([float(func(row)) for row in a])
    f_b = np.array([float(func(row)) for row in b])
    variance = float(np.var(np.concatenate([f_a, f_b]), ddof=1))
    if variance <= 1e-14:
        raise ValueError("model output variance is too small for Sobol analysis")

    first = np.empty(d, dtype=float)
    total = np.empty(d, dtype=float)
    for i in range(d):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        f_ab = np.array([float(func(row)) for row in ab_i])
        total[i] = np.mean((f_a - f_ab) ** 2) / (2.0 * variance)
        first[i] = 1.0 - np.mean((f_b - f_ab) ** 2) / (2.0 * variance)

    first = np.clip(first, -0.05, 1.05)
    total = np.clip(total, -0.05, 1.05)
    return SobolResult(first_order=first, total_order=total, variance=variance, n_base=n_base)
