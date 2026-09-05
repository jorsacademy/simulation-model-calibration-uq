from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from .simulator import METRIC_NAMES, QueueParameters, simulate_replications


DEFAULT_BOUNDS = np.array(
    [
        [0.45, 1.35],
        [0.80, 1.90],
        [0.015, 0.14],
        [0.18, 0.95],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class ObservationSet:
    replications: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.replications, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(METRIC_NAMES):
            raise ValueError("replications must have shape (n, number_of_metrics)")
        if len(arr) < 2:
            raise ValueError("at least two observed replications are required")
        object.__setattr__(self, "replications", arr)

    @property
    def mean(self) -> np.ndarray:
        return self.replications.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.replications.std(axis=0, ddof=1)

    @property
    def standard_error(self) -> np.ndarray:
        floors = np.array([0.015, 0.35, 0.015, 0.01, 0.30, 0.008], dtype=float)
        return np.maximum(self.std / np.sqrt(len(self.replications)), floors)


@dataclass(frozen=True)
class CalibrationResult:
    parameters: QueueParameters
    cost: float
    optimality: float
    nfev: int
    residuals: np.ndarray
    predicted_metrics: np.ndarray
    observed_metrics: np.ndarray
    success: bool
    message: str


def generate_observations(
    params: QueueParameters,
    seeds: Iterable[int],
    *,
    horizon: float = 500.0,
) -> ObservationSet:
    return ObservationSet(simulate_replications(params, seeds, horizon=horizon))


def calibrate(
    observations: ObservationSet,
    *,
    simulation_seeds: Iterable[int],
    horizon: float = 500.0,
    bounds: np.ndarray = DEFAULT_BOUNDS,
    start: Iterable[float] | None = None,
    max_nfev: int = 80,
    global_search: bool = True,
    seed: int = 123,
    global_maxiter: int = 18,
    global_popsize: int = 6,
) -> CalibrationResult:
    """Calibrate rates by bounded nonlinear weighted least squares.

    Common random numbers are used across candidate parameter vectors by keeping the
    simulation seed set fixed during optimization. This makes the stochastic objective
    smoother without pretending that simulator noise disappeared.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")
    seeds = tuple(int(s) for s in simulation_seeds)
    if not seeds:
        raise ValueError("simulation_seeds cannot be empty")

    lower, upper = bounds[:, 0], bounds[:, 1]
    x0 = (lower + upper) / 2.0 if start is None else np.asarray(list(start), dtype=float)
    scale = observations.standard_error
    target = observations.mean

    def residual_fn(x: np.ndarray) -> np.ndarray:
        pred = simulate_replications(
            QueueParameters.from_array(x), seeds, horizon=horizon
        ).mean(axis=0)
        return (pred - target) / scale

    if global_search:
        def scalar_objective(x: np.ndarray) -> float:
            residual = residual_fn(x)
            return float(np.dot(residual, residual))

        global_fit = differential_evolution(
            scalar_objective,
            list(map(tuple, bounds)),
            strategy="best1bin",
            maxiter=global_maxiter,
            popsize=global_popsize,
            tol=0.02,
            polish=False,
            seed=seed,
            updating="immediate",
        )
        x0 = np.asarray(global_fit.x, dtype=float)

    fit = least_squares(
        residual_fn,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    predicted = simulate_replications(
        QueueParameters.from_array(fit.x), seeds, horizon=horizon
    ).mean(axis=0)
    return CalibrationResult(
        parameters=QueueParameters.from_array(fit.x),
        cost=float(fit.cost),
        optimality=float(fit.optimality),
        nfev=int(fit.nfev),
        residuals=np.asarray((predicted - target) / scale, dtype=float),
        predicted_metrics=predicted,
        observed_metrics=target,
        success=bool(fit.success),
        message=str(fit.message),
    )
