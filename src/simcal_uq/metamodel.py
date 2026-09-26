from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import ndtr


@dataclass(frozen=True)
class ReplicatedDesign:
    x: np.ndarray
    mean: np.ndarray
    sample_variance: np.ndarray
    mean_variance: np.ndarray
    replications: np.ndarray


def summarize_replications(
    x: np.ndarray,
    values: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    variance_floor: float = 1e-10,
) -> ReplicatedDesign:
    points = np.asarray(x, dtype=float)
    if points.ndim != 2:
        raise ValueError("x must be a two-dimensional design matrix")
    if len(values) != len(points):
        raise ValueError("values must contain one replication vector per design point")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")

    means: list[float] = []
    variances: list[float] = []
    mean_variances: list[float] = []
    counts: list[int] = []

    for raw in values:
        observations = np.asarray(raw, dtype=float)
        if observations.ndim != 1 or len(observations) < 2:
            raise ValueError("each design point needs at least two replications")
        if not np.all(np.isfinite(observations)):
            raise ValueError("replications must be finite")

        sample_variance = float(np.var(observations, ddof=1))
        sample_variance = max(sample_variance, variance_floor)
        count = int(len(observations))

        means.append(float(np.mean(observations)))
        variances.append(sample_variance)
        mean_variances.append(sample_variance / count)
        counts.append(count)

    return ReplicatedDesign(
        x=points,
        mean=np.asarray(means, dtype=float),
        sample_variance=np.asarray(variances, dtype=float),
        mean_variance=np.asarray(mean_variances, dtype=float),
        replications=np.asarray(counts, dtype=int),
    )


class StochasticKriging:
    """Small ordinary-kriging model with heteroskedastic replication noise.

    The observation-noise diagonal is the estimated variance of each replicated
    sample mean. Hyperparameters are estimated by Gaussian marginal likelihood.
    This is a transparent stochastic-kriging-style metamodel for small research
    benchmarks, not a replacement for a production GP library.
    """

    def __init__(self, *, jitter: float = 1e-9) -> None:
        if jitter <= 0:
            raise ValueError("jitter must be positive")
        self.jitter = float(jitter)
        self._fitted = False

    @staticmethod
    def _kernel(
        x1: np.ndarray,
        x2: np.ndarray,
        length_scale: np.ndarray,
        signal_variance: float,
    ) -> np.ndarray:
        scaled = (
            x1[:, None, :] - x2[None, :, :]
        ) / length_scale[None, None, :]
        squared_distance = np.sum(scaled * scaled, axis=2)
        return signal_variance * np.exp(-0.5 * squared_distance)

    def _standardize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_offset_) / self.x_scale_

    def fit(self, design: ReplicatedDesign) -> StochasticKriging:
        x = np.asarray(design.x, dtype=float)
        y = np.asarray(design.mean, dtype=float)
        noise = np.asarray(design.mean_variance, dtype=float)

        if x.ndim != 2 or len(x) < 3:
            raise ValueError("stochastic kriging requires at least three design points")
        if y.shape != (len(x),) or noise.shape != (len(x),):
            raise ValueError("incompatible design shapes")
        if np.any(noise <= 0) or not np.all(np.isfinite(noise)):
            raise ValueError("mean-variance estimates must be positive and finite")

        self.x_offset_ = np.min(x, axis=0)
        span = np.max(x, axis=0) - self.x_offset_
        self.x_scale_ = np.where(span > 1e-12, span, 1.0)
        xs = self._standardize_x(x)

        y_variance = max(float(np.var(y, ddof=1)), 1e-6)
        initial = np.concatenate(
            [
                np.full(x.shape[1], np.log(0.35)),
                np.array([np.log(y_variance)]),
            ]
        )
        bounds = [(-4.0, 2.5)] * x.shape[1] + [
            (np.log(y_variance) - 8.0, np.log(y_variance) + 8.0)
        ]

        def objective(theta: np.ndarray) -> float:
            length_scale = np.exp(theta[:-1])
            signal_variance = float(np.exp(theta[-1]))
            covariance = self._kernel(
                xs,
                xs,
                length_scale,
                signal_variance,
            )
            covariance = covariance + np.diag(noise + self.jitter)

            try:
                factor = cho_factor(covariance, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e50

            ones = np.ones(len(y), dtype=float)
            kinv_y = cho_solve(factor, y, check_finite=False)
            kinv_one = cho_solve(factor, ones, check_finite=False)
            denominator = float(ones @ kinv_one)
            if denominator <= 0:
                return 1e50

            trend = float((ones @ kinv_y) / denominator)
            residual = y - trend
            alpha = cho_solve(factor, residual, check_finite=False)
            log_determinant = 2.0 * float(np.log(np.diag(factor[0])).sum())
            return float(
                0.5 * residual @ alpha
                + 0.5 * log_determinant
                + 0.5 * len(y) * np.log(2.0 * np.pi)
            )

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
        )
        theta = result.x if np.all(np.isfinite(result.x)) else initial

        self.length_scale_ = np.exp(theta[:-1])
        self.signal_variance_ = float(np.exp(theta[-1]))
        self.x_train_ = xs
        self.y_train_ = y
        self.noise_variance_ = noise

        covariance = self._kernel(
            xs,
            xs,
            self.length_scale_,
            self.signal_variance_,
        )
        covariance = covariance + np.diag(noise + self.jitter)
        self.factor_ = cho_factor(covariance, lower=True, check_finite=False)

        ones = np.ones(len(y), dtype=float)
        kinv_y = cho_solve(self.factor_, y, check_finite=False)
        self.kinv_one_ = cho_solve(self.factor_, ones, check_finite=False)
        self.trend_denominator_ = float(ones @ self.kinv_one_)
        self.trend_ = float((ones @ kinv_y) / self.trend_denominator_)
        residual = y - self.trend_
        self.alpha_ = cho_solve(self.factor_, residual, check_finite=False)
        self._fitted = True
        return self

    def predict(
        self,
        x: np.ndarray,
        *,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("fit the stochastic-kriging model first")

        points = np.asarray(x, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != self.x_train_.shape[1]:
            raise ValueError("prediction points have incompatible shape")

        xs = self._standardize_x(points)
        cross = self._kernel(
            self.x_train_,
            xs,
            self.length_scale_,
            self.signal_variance_,
        )
        mean = self.trend_ + cross.T @ self.alpha_

        if not return_std:
            return np.asarray(mean, dtype=float)

        solved = cho_solve(self.factor_, cross, check_finite=False)
        base_variance = self.signal_variance_ - np.sum(cross * solved, axis=0)
        one_correction = 1.0 - self.kinv_one_ @ cross
        trend_variance = (one_correction * one_correction) / self.trend_denominator_
        variance = np.maximum(base_variance + trend_variance, 0.0)
        return np.asarray(mean, dtype=float), np.sqrt(variance)


def expected_improvement_minimize(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: float,
) -> np.ndarray:
    mu = np.asarray(mean, dtype=float)
    sigma = np.asarray(std, dtype=float)
    if mu.shape != sigma.shape:
        raise ValueError("mean and std must have equal shapes")
    if np.any(sigma < 0):
        raise ValueError("std must be non-negative")

    improvement = float(incumbent) - mu
    result = np.maximum(improvement, 0.0)
    positive = sigma > 1e-14
    if np.any(positive):
        z = improvement[positive] / sigma[positive]
        density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        result[positive] = (
            improvement[positive] * ndtr(z)
            + sigma[positive] * density
        )
    return result
