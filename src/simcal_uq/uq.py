from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .calibration import DEFAULT_BOUNDS, CalibrationResult, ObservationSet, calibrate
from .simulator import METRIC_NAMES, QueueParameters, simulate_queue, simulate_replications


@dataclass(frozen=True)
class BootstrapResult:
    parameter_samples: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray
    success_rate: float


@dataclass(frozen=True)
class PropagationResult:
    metric_samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    median: np.ndarray
    upper: np.ndarray


def bootstrap_calibration(
    observations: ObservationSet,
    base_fit: CalibrationResult,
    *,
    simulation_seeds: Iterable[int],
    n_bootstrap: int = 30,
    horizon: float = 500.0,
    bounds: np.ndarray = DEFAULT_BOUNDS,
    seed: int = 123,
    max_nfev: int = 45,
) -> BootstrapResult:
    """Nonparametric case bootstrap over observed simulation/measurement replications."""
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    successes = 0
    n = len(observations.replications)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_obs = ObservationSet(observations.replications[idx])
        fit = calibrate(
            boot_obs,
            simulation_seeds=simulation_seeds,
            horizon=horizon,
            bounds=bounds,
            start=base_fit.parameters.as_array(),
            max_nfev=max_nfev,
            global_search=True,
            seed=seed + 100 + b,
            global_maxiter=8,
            global_popsize=5,
        )
        samples.append(fit.parameters.as_array())
        successes += int(fit.success)
    arr = np.vstack(samples)
    q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
    return BootstrapResult(
        parameter_samples=arr,
        lower=q[0],
        median=q[1],
        upper=q[2],
        success_rate=successes / n_bootstrap,
    )


def _summarize_metric_samples(arr: np.ndarray) -> PropagationResult:
    q = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
    return PropagationResult(
        metric_samples=arr,
        mean=arr.mean(axis=0),
        std=arr.std(axis=0, ddof=1),
        lower=q[0],
        median=q[1],
        upper=q[2],
    )


def propagate_parameter_uncertainty(
    parameter_samples: np.ndarray,
    *,
    evaluation_seeds: Iterable[int] = (7001, 7002, 7003),
    horizon: float = 500.0,
) -> PropagationResult:
    """Propagate calibration uncertainty while controlling simulator randomness with CRN."""
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("parameter_samples must have shape (n, 4)")
    seeds = tuple(int(s) for s in evaluation_seeds)
    if not seeds:
        raise ValueError("evaluation_seeds cannot be empty")
    rows = []
    for row in samples:
        rows.append(
            simulate_replications(
                QueueParameters.from_array(row), seeds, horizon=horizon
            ).mean(axis=0)
        )
    return _summarize_metric_samples(np.vstack(rows))


def propagate_predictive_uncertainty(
    parameter_samples: np.ndarray,
    *,
    horizon: float = 500.0,
    seed: int = 987,
) -> PropagationResult:
    """Combine parameter uncertainty with fresh stochastic simulator variability."""
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("parameter_samples must have shape (n, 4)")
    rng = np.random.default_rng(seed)
    rows = []
    for row in samples:
        sim_seed = int(rng.integers(0, 2**31 - 1))
        rows.append(
            simulate_queue(
                QueueParameters.from_array(row), horizon=horizon, seed=sim_seed
            ).metric_vector()
        )
    return _summarize_metric_samples(np.vstack(rows))


def named_parameter_intervals(result: BootstrapResult) -> dict[str, dict[str, float]]:
    names = ("arrival_rate", "service_rate", "failure_rate", "repair_rate")
    return {
        name: {
            "lower_95": float(result.lower[i]),
            "median": float(result.median[i]),
            "upper_95": float(result.upper[i]),
        }
        for i, name in enumerate(names)
    }


def named_metric_intervals(result: PropagationResult) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(result.mean[i]),
            "std": float(result.std[i]),
            "lower_95": float(result.lower[i]),
            "median": float(result.median[i]),
            "upper_95": float(result.upper[i]),
        }
        for i, name in enumerate(METRIC_NAMES)
    }
