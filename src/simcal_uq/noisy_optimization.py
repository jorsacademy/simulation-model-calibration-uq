from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metamodel import (
    StochasticKriging,
    expected_improvement_minimize,
    summarize_replications,
)
from .simulator import QueueParameters, simulate_queue


@dataclass(frozen=True)
class QueueDesignProblem:
    arrival_rate: float = 0.80
    failure_rate: float = 0.04
    horizon: float = 300.0
    service_capacity_cost: float = 2.5
    repair_capacity_cost: float = 0.8
    queue_cost: float = 1.0
    cycle_time_cost: float = 0.15

    def evaluate(self, point: np.ndarray, *, seed: int) -> float:
        decision = np.asarray(point, dtype=float)
        if decision.shape != (2,):
            raise ValueError("queue design point must contain [service_rate, repair_rate]")
        service_rate, repair_rate = map(float, decision)
        if service_rate <= 0 or repair_rate <= 0:
            raise ValueError("service and repair rates must be positive")

        result = simulate_queue(
            QueueParameters(
                arrival_rate=self.arrival_rate,
                service_rate=service_rate,
                failure_rate=self.failure_rate,
                repair_rate=repair_rate,
            ),
            horizon=self.horizon,
            seed=int(seed),
        )
        return float(
            self.service_capacity_cost * service_rate
            + self.repair_capacity_cost * repair_rate
            + self.queue_cost * result.mean_queue
            + self.cycle_time_cost * result.mean_cycle_time
        )


@dataclass(frozen=True)
class NoisyOptimizationResult:
    method: str
    selected_index: int
    selected_point: np.ndarray
    budget_used: int
    counts: np.ndarray
    sample_means: np.ndarray
    sample_variances: np.ndarray


def queue_design_grid(
    *,
    service_rates: np.ndarray | None = None,
    repair_rates: np.ndarray | None = None,
) -> np.ndarray:
    services = (
        np.linspace(0.95, 1.55, 7)
        if service_rates is None
        else np.asarray(service_rates, dtype=float)
    )
    repairs = (
        np.linspace(0.12, 0.48, 7)
        if repair_rates is None
        else np.asarray(repair_rates, dtype=float)
    )
    if services.ndim != 1 or repairs.ndim != 1:
        raise ValueError("service and repair grids must be one-dimensional")
    if len(services) < 2 or len(repairs) < 2:
        raise ValueError("each grid needs at least two values")
    mesh = np.array(
        [[service, repair] for service in services for repair in repairs],
        dtype=float,
    )
    return mesh


def _draw_replications(
    problem: QueueDesignProblem,
    point: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("number of replications must be positive")
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n, dtype=np.int64)
    return np.asarray(
        [problem.evaluate(point, seed=int(seed)) for seed in seeds],
        dtype=float,
    )


def _summary_from_store(
    candidates: np.ndarray,
    store: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.full(len(candidates), np.nan, dtype=float)
    variances = np.full(len(candidates), np.nan, dtype=float)
    counts = np.zeros(len(candidates), dtype=int)
    for index, values in enumerate(store):
        if not values:
            continue
        observations = np.asarray(values, dtype=float)
        means[index] = float(np.mean(observations))
        variances[index] = (
            float(np.var(observations, ddof=1))
            if len(observations) > 1
            else 0.0
        )
        counts[index] = len(observations)
    return means, variances, counts


def _result(
    method: str,
    candidates: np.ndarray,
    store: list[list[float]],
    budget_used: int,
    selected_index: int,
) -> NoisyOptimizationResult:
    means, variances, counts = _summary_from_store(candidates, store)
    return NoisyOptimizationResult(
        method=method,
        selected_index=int(selected_index),
        selected_point=np.asarray(candidates[selected_index], dtype=float),
        budget_used=int(budget_used),
        counts=counts,
        sample_means=means,
        sample_variances=variances,
    )


def run_equal_allocation(
    problem: QueueDesignProblem,
    candidates: np.ndarray,
    *,
    total_budget: int,
    random_state: int = 42,
) -> NoisyOptimizationResult:
    candidates = np.asarray(candidates, dtype=float)
    n_candidates = len(candidates)
    if total_budget < 2 * n_candidates:
        raise ValueError("equal allocation needs at least two replications per candidate")

    base = total_budget // n_candidates
    remainder = total_budget % n_candidates
    rng = np.random.default_rng(random_state)
    store: list[list[float]] = [[] for _ in range(n_candidates)]

    for index, point in enumerate(candidates):
        reps = base + int(index < remainder)
        values = _draw_replications(problem, point, reps, rng)
        store[index].extend(map(float, values))

    means, _, _ = _summary_from_store(candidates, store)
    selected = int(np.nanargmin(means))
    return _result("equal_allocation", candidates, store, total_budget, selected)


def ocba_target_allocation(
    means: np.ndarray,
    sample_variances: np.ndarray,
    total_target: int,
    *,
    minimum_per_alternative: int = 2,
    gap_floor: float = 1e-8,
    variance_floor: float = 1e-10,
) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    variances = np.maximum(np.asarray(sample_variances, dtype=float), variance_floor)
    if means.ndim != 1 or variances.shape != means.shape:
        raise ValueError("means and variances must be equal-length vectors")
    if len(means) < 2:
        raise ValueError("OCBA needs at least two alternatives")
    if total_target < minimum_per_alternative * len(means):
        raise ValueError("total target is too small for the requested minimum allocation")

    best = int(np.argmin(means))
    gaps = np.abs(means - means[best])
    nonbest = np.arange(len(means)) != best
    gaps[nonbest] = np.maximum(gaps[nonbest], gap_floor)

    weights = np.zeros(len(means), dtype=float)
    weights[nonbest] = variances[nonbest] / (gaps[nonbest] ** 2)

    sigma_best = float(np.sqrt(variances[best]))
    ratio_term = np.sum(
        (weights[nonbest] ** 2) / variances[nonbest]
    )
    weights[best] = sigma_best * float(np.sqrt(max(ratio_term, variance_floor)))

    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0:
        weights[:] = 1.0

    free_budget = total_target - minimum_per_alternative * len(means)
    raw = weights / weights.sum() * free_budget
    allocation = np.full(len(means), minimum_per_alternative, dtype=int)
    allocation += np.floor(raw).astype(int)

    missing = total_target - int(allocation.sum())
    if missing > 0:
        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional)
        allocation[order[:missing]] += 1
    return allocation


def run_ocba(
    problem: QueueDesignProblem,
    candidates: np.ndarray,
    *,
    total_budget: int,
    initial_replications: int = 3,
    allocation_batch: int = 12,
    random_state: int = 42,
) -> NoisyOptimizationResult:
    candidates = np.asarray(candidates, dtype=float)
    n_candidates = len(candidates)
    initial_cost = initial_replications * n_candidates
    if initial_replications < 2:
        raise ValueError("OCBA needs at least two initial replications")
    if total_budget < initial_cost:
        raise ValueError("total budget is smaller than the initial OCBA design")
    if allocation_batch <= 0:
        raise ValueError("allocation_batch must be positive")

    rng = np.random.default_rng(random_state)
    store: list[list[float]] = [[] for _ in range(n_candidates)]
    for index, point in enumerate(candidates):
        values = _draw_replications(
            problem,
            point,
            initial_replications,
            rng,
        )
        store[index].extend(map(float, values))

    used = initial_cost
    while used < total_budget:
        means, variances, counts = _summary_from_store(candidates, store)
        target_total = min(total_budget, used + allocation_batch)
        target = ocba_target_allocation(
            means,
            variances,
            target_total,
            minimum_per_alternative=2,
        )
        incremental = np.maximum(target - counts, 0)
        if int(incremental.sum()) == 0:
            incremental[int(np.argmin(means))] = min(
                allocation_batch,
                total_budget - used,
            )

        for index, reps in enumerate(incremental):
            if used >= total_budget:
                break
            reps = min(int(reps), total_budget - used)
            if reps <= 0:
                continue
            values = _draw_replications(problem, candidates[index], reps, rng)
            store[index].extend(map(float, values))
            used += reps

    means, _, _ = _summary_from_store(candidates, store)
    selected = int(np.nanargmin(means))
    return _result("ocba", candidates, store, used, selected)


def run_stochastic_kriging_ei(
    problem: QueueDesignProblem,
    candidates: np.ndarray,
    *,
    total_budget: int,
    initial_points: int = 8,
    initial_replications: int = 3,
    sequential_replications: int = 2,
    random_state: int = 42,
) -> NoisyOptimizationResult:
    candidates = np.asarray(candidates, dtype=float)
    if candidates.ndim != 2:
        raise ValueError("candidates must be a two-dimensional array")
    n_candidates = len(candidates)
    if not 3 <= initial_points <= n_candidates:
        raise ValueError("initial_points must be between 3 and the number of candidates")
    if initial_replications < 2 or sequential_replications < 2:
        raise ValueError("stochastic kriging needs at least two replications per sampled point")

    initial_cost = initial_points * initial_replications
    if total_budget < initial_cost:
        raise ValueError("total budget is smaller than the initial metamodel design")

    rng = np.random.default_rng(random_state)
    initial_indices = np.sort(
        rng.choice(n_candidates, size=initial_points, replace=False)
    )
    store: list[list[float]] = [[] for _ in range(n_candidates)]

    for index in initial_indices:
        values = _draw_replications(
            problem,
            candidates[index],
            initial_replications,
            rng,
        )
        store[index].extend(map(float, values))
    used = initial_cost

    while used + sequential_replications <= total_budget:
        observed = np.asarray(
            [index for index, values in enumerate(store) if len(values) >= 2],
            dtype=int,
        )
        replicated = [np.asarray(store[index], dtype=float) for index in observed]
        design = summarize_replications(candidates[observed], replicated)
        model = StochasticKriging().fit(design)

        mean, std = model.predict(candidates, return_std=True)
        incumbent = float(np.min(design.mean))
        acquisition = expected_improvement_minimize(mean, std, incumbent)
        next_index = int(np.argmax(acquisition))

        values = _draw_replications(
            problem,
            candidates[next_index],
            sequential_replications,
            rng,
        )
        store[next_index].extend(map(float, values))
        used += sequential_replications

    if used < total_budget:
        means, _, _ = _summary_from_store(candidates, store)
        observed = np.flatnonzero(np.isfinite(means))
        best_observed = int(observed[np.argmin(means[observed])])
        values = _draw_replications(
            problem,
            candidates[best_observed],
            total_budget - used,
            rng,
        )
        store[best_observed].extend(map(float, values))
        used = total_budget

    observed = np.asarray(
        [index for index, values in enumerate(store) if len(values) >= 2],
        dtype=int,
    )
    replicated = [np.asarray(store[index], dtype=float) for index in observed]
    design = summarize_replications(candidates[observed], replicated)
    model = StochasticKriging().fit(design)
    posterior_mean = model.predict(candidates)
    selected = int(np.argmin(posterior_mean))
    return _result("stochastic_kriging_ei", candidates, store, used, selected)


def estimate_reference_means(
    problem: QueueDesignProblem,
    candidates: np.ndarray,
    *,
    replications: int = 100,
    random_state: int = 10000,
) -> np.ndarray:
    if replications <= 0:
        raise ValueError("replications must be positive")
    candidates = np.asarray(candidates, dtype=float)
    rng = np.random.default_rng(random_state)
    means = []
    for point in candidates:
        values = _draw_replications(problem, point, replications, rng)
        means.append(float(np.mean(values)))
    return np.asarray(means, dtype=float)


def simple_regret(
    result: NoisyOptimizationResult,
    reference_means: np.ndarray,
) -> float:
    reference = np.asarray(reference_means, dtype=float)
    if reference.shape != result.sample_means.shape:
        raise ValueError("reference means have incompatible shape")
    optimum = float(np.min(reference))
    return float(reference[result.selected_index] - optimum)
