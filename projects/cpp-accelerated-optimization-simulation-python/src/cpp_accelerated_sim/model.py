from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

try:
    from ._core import simulate_batch as _simulate_cpp
except ImportError:  # source-tree / unsupported-platform fallback
    _simulate_cpp = None


@dataclass(frozen=True, order=True)
class Policy:
    threshold_state: int
    crew_slots: int
    standby_spares: int

    def __post_init__(self) -> None:
        if self.threshold_state not in (1, 2, 3):
            raise ValueError("threshold_state must be 1, 2, or 3")
        if self.crew_slots not in (1, 2, 3):
            raise ValueError("crew_slots must be 1, 2, or 3")
        if self.standby_spares not in (0, 1, 2):
            raise ValueError("standby_spares must be 0, 1, or 2")


@dataclass(frozen=True)
class ScenarioDraws:
    degradation_u: np.ndarray
    maintenance_u: np.ndarray

    def __post_init__(self) -> None:
        if self.degradation_u.shape != self.maintenance_u.shape:
            raise ValueError("random arrays must have identical shape")
        if self.degradation_u.ndim != 3:
            raise ValueError("random arrays must be [scenario, day, asset]")
        if self.degradation_u.size == 0:
            raise ValueError("empty scenario draw tensor")
        if np.any((self.degradation_u < 0) | (self.degradation_u >= 1)):
            raise ValueError("degradation_u must be in [0,1)")
        if np.any((self.maintenance_u < 0) | (self.maintenance_u >= 1)):
            raise ValueError("maintenance_u must be in [0,1)")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(map(int, self.degradation_u.shape))


@dataclass(frozen=True)
class BatchResult:
    total_cost: np.ndarray
    lost_unit_days: np.ndarray
    failures: np.ndarray
    maintenance_events: np.ndarray


@dataclass(frozen=True)
class PolicyMetrics:
    mean_cost: float
    cvar95_cost: float
    objective: float
    mean_lost_unit_days: float
    mean_failures: float
    mean_maintenance_events: float


@dataclass(frozen=True)
class OptimizationResult:
    selected_policy: Policy
    selected_selection_metrics: PolicyMetrics
    selected_validation_metrics: PolicyMetrics
    baseline_policy: Policy
    baseline_validation_metrics: PolicyMetrics
    candidate_count: int
    backend: str


@dataclass(frozen=True)
class BenchmarkRow:
    backend: str
    seconds: float
    speedup_vs_python: float
    mean_cost: float


def cpp_available() -> bool:
    return _simulate_cpp is not None


def generate_draws(
    scenarios: int,
    *,
    days: int = 180,
    assets: int = 16,
    seed: int = 42,
) -> ScenarioDraws:
    if scenarios <= 0 or days <= 0 or assets <= 0:
        raise ValueError("scenarios, days and assets must be positive")
    rng = np.random.default_rng(seed)
    shape = (int(scenarios), int(days), int(assets))
    return ScenarioDraws(
        degradation_u=np.ascontiguousarray(rng.random(shape), dtype=np.float64),
        maintenance_u=np.ascontiguousarray(rng.random(shape), dtype=np.float64),
    )


def simulate_python(draws: ScenarioDraws, policy: Policy) -> BatchResult:
    """Scalar Python reference implementation; intentionally loop-heavy."""
    degradation_u = draws.degradation_u
    maintenance_u = draws.maintenance_u
    scenarios, days, assets = draws.shape

    total_cost = np.zeros(scenarios, dtype=np.float64)
    lost_unit_days = np.zeros(scenarios, dtype=np.float64)
    failures = np.zeros(scenarios, dtype=np.float64)
    maintenance_events = np.zeros(scenarios, dtype=np.float64)

    for s in range(scenarios):
        state = [0] * assets
        cost = 0.0
        lost_days = 0.0
        failed_count = 0.0
        maintenance_count = 0.0

        for day in range(days):
            candidates = [a for a in range(assets) if state[a] >= policy.threshold_state]
            candidates.sort(key=lambda a: (-state[a], a))
            selected = set(candidates[: policy.crew_slots])

            for a in selected:
                maintenance_count += 1.0
                u = float(maintenance_u[s, day, a])
                if state[a] == 3:
                    cost += 5000.0
                    state[a] = 0 if u < 0.98 else 1
                else:
                    cost += 900.0
                    state[a] = 0 if u < 0.92 else max(0, state[a] - 1)

            cost += policy.crew_slots * 140.0
            cost += policy.standby_spares * 90.0

            deficit = 0.0
            for a in range(assets):
                if a in selected:
                    deficit += 1.0
                elif state[a] == 1:
                    deficit += 0.10
                elif state[a] == 2:
                    deficit += 0.35
                elif state[a] == 3:
                    deficit += 1.0

            lost = max(0.0, deficit - policy.standby_spares)
            lost_days += lost
            cost += lost * 2200.0

            for a in range(assets):
                if a in selected or state[a] == 3:
                    continue
                u = float(degradation_u[s, day, a])
                if state[a] == 0 and u < 0.006:
                    state[a] = 1
                elif state[a] == 1 and u < 0.018:
                    state[a] = 2
                elif state[a] == 2 and u < 0.070:
                    state[a] = 3
                    failed_count += 1.0

        total_cost[s] = cost
        lost_unit_days[s] = lost_days
        failures[s] = failed_count
        maintenance_events[s] = maintenance_count

    return BatchResult(total_cost, lost_unit_days, failures, maintenance_events)


def simulate_cpp(draws: ScenarioDraws, policy: Policy) -> BatchResult:
    if _simulate_cpp is None:
        raise RuntimeError("C++ extension is not available; build/install the package first")
    arrays = _simulate_cpp(
        draws.degradation_u,
        draws.maintenance_u,
        policy.threshold_state,
        policy.crew_slots,
        policy.standby_spares,
    )
    return BatchResult(*(np.asarray(x) for x in arrays))


def simulate(
    draws: ScenarioDraws,
    policy: Policy,
    *,
    backend: Literal["auto", "python", "cpp"] = "auto",
) -> BatchResult:
    if backend == "auto":
        backend = "cpp" if cpp_available() else "python"
    if backend == "cpp":
        return simulate_cpp(draws, policy)
    if backend == "python":
        return simulate_python(draws, policy)
    raise ValueError("backend must be auto, python, or cpp")


def cvar95(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty vector")
    k = max(1, int(math.ceil(0.05 * len(values))))
    tail = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(tail))


def summarize(result: BatchResult, *, risk_weight: float = 0.25) -> PolicyMetrics:
    mean = float(np.mean(result.total_cost))
    tail = cvar95(result.total_cost)
    return PolicyMetrics(
        mean_cost=mean,
        cvar95_cost=tail,
        objective=(1.0 - risk_weight) * mean + risk_weight * tail,
        mean_lost_unit_days=float(np.mean(result.lost_unit_days)),
        mean_failures=float(np.mean(result.failures)),
        mean_maintenance_events=float(np.mean(result.maintenance_events)),
    )


def candidate_policies() -> tuple[Policy, ...]:
    return tuple(
        Policy(threshold, crew, spares)
        for threshold in (1, 2, 3)
        for crew in (1, 2, 3)
        for spares in (0, 1, 2)
    )


def optimize_policy(
    *,
    selection_scenarios: int = 1200,
    validation_scenarios: int = 3000,
    days: int = 180,
    assets: int = 16,
    seed: int = 42,
    risk_weight: float = 0.25,
    backend: Literal["auto", "python", "cpp"] = "auto",
) -> OptimizationResult:
    selection = generate_draws(selection_scenarios, days=days, assets=assets, seed=seed)
    validation = generate_draws(validation_scenarios, days=days, assets=assets, seed=seed + 100_000)

    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "cpp" if cpp_available() else "python"

    scored: list[tuple[float, float, Policy, PolicyMetrics]] = []
    for policy in candidate_policies():
        metrics = summarize(simulate(selection, policy, backend=resolved_backend), risk_weight=risk_weight)
        scored.append((metrics.objective, metrics.mean_cost, policy, metrics))
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    _, _, selected, selection_metrics = scored[0]

    baseline = Policy(threshold_state=3, crew_slots=1, standby_spares=0)
    selected_validation = summarize(
        simulate(validation, selected, backend=resolved_backend), risk_weight=risk_weight
    )
    baseline_validation = summarize(
        simulate(validation, baseline, backend=resolved_backend), risk_weight=risk_weight
    )

    return OptimizationResult(
        selected_policy=selected,
        selected_selection_metrics=selection_metrics,
        selected_validation_metrics=selected_validation,
        baseline_policy=baseline,
        baseline_validation_metrics=baseline_validation,
        candidate_count=len(candidate_policies()),
        backend=resolved_backend,
    )


def assert_equivalent(draws: ScenarioDraws, policy: Policy) -> None:
    if not cpp_available():
        raise RuntimeError("C++ extension is not available")
    py = simulate_python(draws, policy)
    cpp = simulate_cpp(draws, policy)
    for name in ("total_cost", "lost_unit_days", "failures", "maintenance_events"):
        np.testing.assert_allclose(
            getattr(py, name), getattr(cpp, name), rtol=0.0, atol=1e-9
        )


def benchmark(
    *,
    scenarios: int = 1500,
    days: int = 180,
    assets: int = 16,
    seed: int = 123,
    repeats: int = 2,
) -> tuple[BenchmarkRow, ...]:
    if not cpp_available():
        raise RuntimeError("C++ extension is required for benchmark")
    draws = generate_draws(scenarios, days=days, assets=assets, seed=seed)
    policy = Policy(2, 2, 1)
    assert_equivalent(draws, policy)

    def timed(fn):
        samples = []
        last = None
        for _ in range(repeats):
            start = time.perf_counter()
            last = fn()
            samples.append(time.perf_counter() - start)
        return min(samples), last

    py_seconds, py_result = timed(lambda: simulate_python(draws, policy))
    cpp_seconds, cpp_result = timed(lambda: simulate_cpp(draws, policy))
    np.testing.assert_allclose(py_result.total_cost, cpp_result.total_cost, rtol=0.0, atol=1e-9)

    mean_cost = float(np.mean(py_result.total_cost))
    return (
        BenchmarkRow("python_scalar", py_seconds, 1.0, mean_cost),
        BenchmarkRow("cpp17_pybind11", cpp_seconds, py_seconds / cpp_seconds, mean_cost),
    )
