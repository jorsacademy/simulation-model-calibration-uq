from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ReservePlan:
    grid_contract_mw: float
    backup_reserve_mw: float


@dataclass(frozen=True)
class ScenarioSet:
    net_load_mw: np.ndarray              # [scenario, period]
    grid_availability: np.ndarray        # [scenario, period], 0..1
    backup_availability: np.ndarray      # [scenario, period], 0..1
    grid_price_per_mwh: np.ndarray       # [scenario, period]
    period_hours: float

    def __post_init__(self):
        arrays = (
            self.net_load_mw,
            self.grid_availability,
            self.backup_availability,
            self.grid_price_per_mwh,
        )
        shape = np.asarray(arrays[0]).shape
        if len(shape) != 2:
            raise ValueError("scenario arrays must be two-dimensional")
        if any(np.asarray(a).shape != shape for a in arrays[1:]):
            raise ValueError("all scenario arrays must have identical shape")
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError("scenario set is empty")
        if self.period_hours <= 0:
            raise ValueError("period_hours must be positive")
        if np.any(self.net_load_mw < 0):
            raise ValueError("negative net load")
        if np.any((self.grid_availability < 0) | (self.grid_availability > 1)):
            raise ValueError("invalid grid availability")
        if np.any((self.backup_availability < 0) | (self.backup_availability > 1)):
            raise ValueError("invalid backup availability")
        if np.any(self.grid_price_per_mwh < 0):
            raise ValueError("negative grid price")

    @property
    def n_scenarios(self) -> int:
        return int(self.net_load_mw.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.net_load_mw.shape[1])

    def slice(self, start: int, stop: int) -> "ScenarioSet":
        return ScenarioSet(
            net_load_mw=self.net_load_mw[start:stop],
            grid_availability=self.grid_availability[start:stop],
            backup_availability=self.backup_availability[start:stop],
            grid_price_per_mwh=self.grid_price_per_mwh[start:stop],
            period_hours=self.period_hours,
        )


@dataclass(frozen=True)
class CostParameters:
    grid_capacity_charge_per_mw_day: float = 150.0
    backup_reserve_charge_per_mw_day: float = 250.0
    backup_energy_cost_per_mwh: float = 225.0
    shortfall_penalty_per_mwh: float = 1500.0

    def __post_init__(self):
        if min(
            self.grid_capacity_charge_per_mw_day,
            self.backup_reserve_charge_per_mw_day,
            self.backup_energy_cost_per_mwh,
            self.shortfall_penalty_per_mwh,
        ) < 0:
            raise ValueError("cost parameters must be nonnegative")


@dataclass(frozen=True)
class PlanMetrics:
    mean_cost: float
    cvar95_cost: float
    risk_objective: float
    probability_any_shortfall: float
    expected_unserved_energy_mwh: float


@dataclass(frozen=True)
class OptimizationResult:
    selected_plan: ReservePlan
    selected_metrics: PlanMetrics
    candidate_count: int
    baseline_plan: ReservePlan
    baseline_validation_metrics: PlanMetrics
    selected_validation_metrics: PlanMetrics


@dataclass(frozen=True)
class BenchmarkRow:
    method: str
    workers: int
    seconds: float
    speedup_vs_scalar: float
    mean_cost: float


def base_profiles(periods: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    load = np.array(
        [
            7.4, 7.1, 6.9, 6.8, 6.9, 7.2,
            8.1, 8.8, 9.5, 10.1, 10.4, 10.2,
            9.9, 9.6, 9.5, 9.8, 10.5, 10.8,
            10.3, 9.7, 9.1, 8.6, 8.2, 7.8,
        ],
        dtype=float,
    )
    solar = np.array(
        [
            0, 0, 0, 0, 0, 0,
            0.2, 0.7, 1.4, 2.0, 2.5, 2.8,
            2.9, 2.7, 2.2, 1.5, 0.8, 0.3,
            0, 0, 0, 0, 0, 0,
        ],
        dtype=float,
    )
    price = np.array(
        [
            78, 74, 72, 70, 72, 80,
            92, 105, 118, 126, 132, 128,
            122, 119, 124, 138, 151, 162,
            155, 139, 120, 105, 94, 85,
        ],
        dtype=float,
    )

    if periods == 24:
        return load, solar, price
    if periods <= 0:
        raise ValueError("periods must be positive")
    x_old = np.linspace(0.0, 1.0, 24)
    x_new = np.linspace(0.0, 1.0, periods)
    return (
        np.interp(x_new, x_old, load),
        np.interp(x_new, x_old, solar),
        np.interp(x_new, x_old, price),
    )


def generate_scenarios(
    n_scenarios: int,
    *,
    periods: int = 24,
    seed: int = 42,
    period_hours: float = 1.0,
) -> ScenarioSet:
    """
    Generate correlated industrial microgrid operating scenarios.

    Uncertainty includes:
    - daily and hourly load variation;
    - cloud-driven solar availability;
    - a two-state grid-derating Markov chain;
    - backup-generator availability/derating;
    - electricity-price shocks correlated with demand.

    Scenarios are generated once and reused across all candidate decisions
    (common random numbers).
    """
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be positive")

    base_load, base_solar, base_price = base_profiles(periods)
    rng = np.random.default_rng(seed)

    daily_load_shock = rng.normal(0.0, 0.055, size=(n_scenarios, 1))
    hourly_load_noise = rng.normal(0.0, 0.028, size=(n_scenarios, periods))
    load = base_load[None, :] * np.exp(
        daily_load_shock + hourly_load_noise - 0.5 * (0.055**2 + 0.028**2)
    )

    cloud_daily = rng.beta(5.0, 2.0, size=(n_scenarios, 1))
    cloud_hourly = np.clip(
        cloud_daily + rng.normal(0.0, 0.10, size=(n_scenarios, periods)),
        0.0,
        1.0,
    )
    solar = base_solar[None, :] * cloud_hourly
    net_load = np.maximum(load - solar, 0.0)

    # Grid state: normal=1.0, derated=0.20. Outages are persistent.
    grid_availability = np.ones((n_scenarios, periods), dtype=float)
    derated = np.zeros(n_scenarios, dtype=bool)
    for t in range(periods):
        u = rng.random(n_scenarios)
        start = (~derated) & (u < 0.006)
        recover = derated & (u < 0.48)
        derated = (derated | start) & (~recover)
        grid_availability[:, t] = np.where(derated, 0.20, 1.0)

    # Backup system can be unavailable or partially derated.
    backup_available = rng.random((n_scenarios, periods)) > 0.018
    backup_derating = 0.82 + 0.18 * rng.beta(
        8.0,
        2.0,
        size=(n_scenarios, periods),
    )
    backup_availability = backup_available * backup_derating

    # Shared demand shock enters price to create realistic positive dependence.
    price_noise = rng.normal(0.0, 0.055, size=(n_scenarios, periods))
    price_multiplier = np.exp(
        0.65 * daily_load_shock
        + price_noise
        - 0.5 * (0.65**2 * 0.055**2 + 0.055**2)
    )
    grid_price = base_price[None, :] * price_multiplier

    return ScenarioSet(
        net_load_mw=net_load.astype(np.float64),
        grid_availability=grid_availability,
        backup_availability=backup_availability.astype(np.float64),
        grid_price_per_mwh=grid_price.astype(np.float64),
        period_hours=float(period_hours),
    )


def _fixed_plan_cost(plan: ReservePlan, costs: CostParameters) -> float:
    if plan.grid_contract_mw < 0 or plan.backup_reserve_mw < 0:
        raise ValueError("plan capacities must be nonnegative")
    return (
        plan.grid_contract_mw * costs.grid_capacity_charge_per_mw_day
        + plan.backup_reserve_mw * costs.backup_reserve_charge_per_mw_day
    )


def scenario_costs_vectorized(
    scenarios: ScenarioSet,
    plan: ReservePlan,
    costs: CostParameters = CostParameters(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized NumPy Monte Carlo kernel."""
    fixed = _fixed_plan_cost(plan, costs)
    dt = scenarios.period_hours

    grid_capacity = (
        plan.grid_contract_mw * scenarios.grid_availability
    )
    grid_supply = np.minimum(
        scenarios.net_load_mw,
        grid_capacity,
    )
    residual = scenarios.net_load_mw - grid_supply

    backup_capacity = (
        plan.backup_reserve_mw * scenarios.backup_availability
    )
    backup_supply = np.minimum(residual, backup_capacity)
    shortfall = residual - backup_supply

    variable_cost = (
        grid_supply * scenarios.grid_price_per_mwh
        + backup_supply * costs.backup_energy_cost_per_mwh
        + shortfall * costs.shortfall_penalty_per_mwh
    ) * dt

    scenario_cost = fixed + np.sum(variable_cost, axis=1)
    unserved_energy = np.sum(shortfall * dt, axis=1)

    return scenario_cost, unserved_energy


def _scenario_costs_scalar_arrays(
    net_load: np.ndarray,
    grid_availability: np.ndarray,
    backup_availability: np.ndarray,
    grid_price: np.ndarray,
    period_hours: float,
    plan: ReservePlan,
    costs: CostParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    fixed = _fixed_plan_cost(plan, costs)
    n, T = net_load.shape

    output = np.empty(n, dtype=float)
    unserved = np.empty(n, dtype=float)

    for scenario_index in range(n):
        total = fixed
        total_unserved = 0.0
        for t in range(T):
            load = float(net_load[scenario_index, t])
            grid_cap = (
                plan.grid_contract_mw
                * float(grid_availability[scenario_index, t])
            )
            grid_supply = min(load, grid_cap)
            remaining = load - grid_supply

            backup_cap = (
                plan.backup_reserve_mw
                * float(backup_availability[scenario_index, t])
            )
            backup_supply = min(remaining, backup_cap)
            shortfall = remaining - backup_supply

            total += period_hours * (
                grid_supply * float(grid_price[scenario_index, t])
                + backup_supply * costs.backup_energy_cost_per_mwh
                + shortfall * costs.shortfall_penalty_per_mwh
            )
            total_unserved += period_hours * shortfall

        output[scenario_index] = total
        unserved[scenario_index] = total_unserved

    return output, unserved


def scenario_costs_scalar(
    scenarios: ScenarioSet,
    plan: ReservePlan,
    costs: CostParameters = CostParameters(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Scalar Python reference kernel used as the correctness oracle."""
    return _scenario_costs_scalar_arrays(
        scenarios.net_load_mw,
        scenarios.grid_availability,
        scenarios.backup_availability,
        scenarios.grid_price_per_mwh,
        scenarios.period_hours,
        plan,
        costs,
    )


def _shared_scalar_chunk_worker(args):
    metadata, start, stop, period_hours, plan, costs = args
    handles = []
    arrays = []
    try:
        for name, shape, dtype_str in metadata:
            shm = shared_memory.SharedMemory(name=name)
            handles.append(shm)
            arrays.append(
                np.ndarray(
                    tuple(shape),
                    dtype=np.dtype(dtype_str),
                    buffer=shm.buf,
                )
            )

        return _scenario_costs_scalar_arrays(
            arrays[0][start:stop],
            arrays[1][start:stop],
            arrays[2][start:stop],
            arrays[3][start:stop],
            period_hours,
            plan,
            costs,
        )
    finally:
        for shm in handles:
            shm.close()


def scenario_costs_multiprocessing(
    scenarios: ScenarioSet,
    plan: ReservePlan,
    costs: CostParameters = CostParameters(),
    *,
    workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel scalar Monte Carlo with shared-memory scenario arrays.

    The parent copies the four scenario tensors into shared memory once.
    Worker processes receive only metadata and index ranges, avoiding repeated
    pickling/copying of large NumPy chunks.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        return scenario_costs_scalar(scenarios, plan, costs)

    source_arrays = (
        scenarios.net_load_mw,
        scenarios.grid_availability,
        scenarios.backup_availability,
        scenarios.grid_price_per_mwh,
    )

    blocks = []
    metadata = []
    try:
        for array in source_arrays:
            contiguous = np.ascontiguousarray(array)
            shm = shared_memory.SharedMemory(
                create=True,
                size=contiguous.nbytes,
            )
            blocks.append(shm)
            shared_view = np.ndarray(
                contiguous.shape,
                dtype=contiguous.dtype,
                buffer=shm.buf,
            )
            shared_view[:] = contiguous
            metadata.append(
                (
                    shm.name,
                    contiguous.shape,
                    contiguous.dtype.str,
                )
            )

        n = scenarios.n_scenarios
        boundaries = np.linspace(
            0,
            n,
            min(workers, n) + 1,
            dtype=int,
        )
        tasks = [
            (
                tuple(metadata),
                int(boundaries[w]),
                int(boundaries[w + 1]),
                scenarios.period_hours,
                plan,
                costs,
            )
            for w in range(len(boundaries) - 1)
            if boundaries[w] < boundaries[w + 1]
        ]

        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
        ) as executor:
            parts = list(
                executor.map(
                    _shared_scalar_chunk_worker,
                    tasks,
                )
            )

        return (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
        )
    finally:
        for shm in blocks:
            try:
                shm.close()
            finally:
                shm.unlink()


def cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a nonempty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")

    k = max(1, int(math.ceil((1.0 - alpha) * len(values))))
    tail = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(tail))


def summarize_plan(
    scenario_cost: np.ndarray,
    unserved_energy: np.ndarray,
    *,
    risk_weight: float = 0.35,
    alpha: float = 0.95,
) -> PlanMetrics:
    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must be in [0,1]")

    mean = float(np.mean(scenario_cost))
    tail = cvar(scenario_cost, alpha=alpha)
    # Convex combination of mean and tail risk.
    objective = (1.0 - risk_weight) * mean + risk_weight * tail

    return PlanMetrics(
        mean_cost=mean,
        cvar95_cost=tail,
        risk_objective=float(objective),
        probability_any_shortfall=float(
            np.mean(unserved_energy > 1e-12)
        ),
        expected_unserved_energy_mwh=float(
            np.mean(unserved_energy)
        ),
    )


def candidate_plans() -> Tuple[ReservePlan, ...]:
    grid_values = np.arange(6.0, 13.0001, 0.5)
    backup_values = np.arange(0.0, 10.0001, 0.5)
    return tuple(
        ReservePlan(float(g), float(b))
        for g in grid_values
        for b in backup_values
    )


def optimize_candidate_grid(
    scenarios: ScenarioSet,
    *,
    costs: CostParameters = CostParameters(),
    risk_weight: float = 0.35,
) -> Tuple[ReservePlan, PlanMetrics]:
    """
    Exact enumeration over the declared finite first-stage decision grid.
    """
    best_plan = None
    best_metrics = None

    for plan in candidate_plans():
        scenario_cost, unserved = scenario_costs_vectorized(
            scenarios,
            plan,
            costs,
        )
        metrics = summarize_plan(
            scenario_cost,
            unserved,
            risk_weight=risk_weight,
        )
        if best_metrics is None or (
            metrics.risk_objective,
            metrics.mean_cost,
            plan.grid_contract_mw + plan.backup_reserve_mw,
        ) < (
            best_metrics.risk_objective,
            best_metrics.mean_cost,
            best_plan.grid_contract_mw + best_plan.backup_reserve_mw,
        ):
            best_plan = plan
            best_metrics = metrics

    assert best_plan is not None and best_metrics is not None
    return best_plan, best_metrics


def mean_scenario_baseline(
    selection_scenarios: ScenarioSet,
    *,
    costs: CostParameters = CostParameters(),
) -> ReservePlan:
    """
    Expected-value baseline: replace all random arrays with their period-wise
    means, then optimize the same finite candidate grid without tail risk.
    """
    mean_set = ScenarioSet(
        net_load_mw=np.mean(
            selection_scenarios.net_load_mw,
            axis=0,
            keepdims=True,
        ),
        grid_availability=np.mean(
            selection_scenarios.grid_availability,
            axis=0,
            keepdims=True,
        ),
        backup_availability=np.mean(
            selection_scenarios.backup_availability,
            axis=0,
            keepdims=True,
        ),
        grid_price_per_mwh=np.mean(
            selection_scenarios.grid_price_per_mwh,
            axis=0,
            keepdims=True,
        ),
        period_hours=selection_scenarios.period_hours,
    )
    plan, _ = optimize_candidate_grid(
        mean_set,
        costs=costs,
        risk_weight=0.0,
    )
    return plan


def run_stochastic_optimization(
    *,
    selection_scenarios: int = 12_000,
    validation_scenarios: int = 40_000,
    seed: int = 42,
    risk_weight: float = 0.35,
) -> OptimizationResult:
    costs = CostParameters()

    selection = generate_scenarios(
        selection_scenarios,
        seed=seed,
    )
    validation = generate_scenarios(
        validation_scenarios,
        seed=seed + 100_000,
    )

    selected_plan, selected_metrics = optimize_candidate_grid(
        selection,
        costs=costs,
        risk_weight=risk_weight,
    )
    baseline_plan = mean_scenario_baseline(
        selection,
        costs=costs,
    )

    selected_cost, selected_unserved = scenario_costs_vectorized(
        validation,
        selected_plan,
        costs,
    )
    baseline_cost, baseline_unserved = scenario_costs_vectorized(
        validation,
        baseline_plan,
        costs,
    )

    selected_validation = summarize_plan(
        selected_cost,
        selected_unserved,
        risk_weight=risk_weight,
    )
    baseline_validation = summarize_plan(
        baseline_cost,
        baseline_unserved,
        risk_weight=risk_weight,
    )

    return OptimizationResult(
        selected_plan=selected_plan,
        selected_metrics=selected_metrics,
        candidate_count=len(candidate_plans()),
        baseline_plan=baseline_plan,
        baseline_validation_metrics=baseline_validation,
        selected_validation_metrics=selected_validation,
    )


def benchmark_kernels(
    *,
    n_scenarios: int = 40_000,
    seed: int = 123,
    worker_counts: Sequence[int] = (1, 2, 4),
    repeats: int = 2,
) -> Tuple[BenchmarkRow, ...]:
    """
    Benchmark one fixed plan on one fixed scenario set.

    Timing includes shared-memory allocation/copy, process creation and cleanup,
    making the reported wall-clock speedup an end-to-end measurement rather
    than an idealized kernel-only number.
    """
    scenarios = generate_scenarios(n_scenarios, seed=seed)
    plan = ReservePlan(8.5, 3.0)
    costs = CostParameters()

    # Correctness reference.
    scalar_cost, scalar_unserved = scenario_costs_scalar(
        scenarios,
        plan,
        costs,
    )
    vector_cost, vector_unserved = scenario_costs_vectorized(
        scenarios,
        plan,
        costs,
    )
    if not np.allclose(scalar_cost, vector_cost, atol=1e-9, rtol=1e-12):
        raise RuntimeError("scalar and vectorized costs disagree")
    if not np.allclose(
        scalar_unserved,
        vector_unserved,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise RuntimeError("scalar and vectorized unserved energy disagree")

    def timed(fn):
        values = []
        last = None
        for _ in range(repeats):
            start = time.perf_counter()
            last = fn()
            values.append(time.perf_counter() - start)
        return float(min(values)), last

    scalar_seconds, _ = timed(
        lambda: scenario_costs_scalar(scenarios, plan, costs)
    )
    vector_seconds, _ = timed(
        lambda: scenario_costs_vectorized(scenarios, plan, costs)
    )

    rows = [
        BenchmarkRow(
            method="scalar_python",
            workers=1,
            seconds=scalar_seconds,
            speedup_vs_scalar=1.0,
            mean_cost=float(np.mean(scalar_cost)),
        ),
        BenchmarkRow(
            method="vectorized_numpy",
            workers=1,
            seconds=vector_seconds,
            speedup_vs_scalar=scalar_seconds / vector_seconds,
            mean_cost=float(np.mean(vector_cost)),
        ),
    ]

    for workers in worker_counts:
        if workers < 1:
            continue
        seconds, result = timed(
            lambda w=workers: scenario_costs_multiprocessing(
                scenarios,
                plan,
                costs,
                workers=w,
            )
        )
        parallel_cost, parallel_unserved = result
        if not np.allclose(
            scalar_cost,
            parallel_cost,
            atol=1e-9,
            rtol=1e-12,
        ):
            raise RuntimeError(
                f"multiprocessing({workers}) costs disagree"
            )
        if not np.allclose(
            scalar_unserved,
            parallel_unserved,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(
                f"multiprocessing({workers}) unserved energy disagrees"
            )

        rows.append(
            BenchmarkRow(
                method="multiprocessing_shared",
                workers=int(workers),
                seconds=seconds,
                speedup_vs_scalar=scalar_seconds / seconds,
                mean_cost=float(np.mean(parallel_cost)),
            )
        )

    return tuple(rows)


def hand_oracle() -> None:
    """One-scenario/two-period hand-check of the recourse equations."""
    scenarios = ScenarioSet(
        net_load_mw=np.array([[5.0, 8.0]]),
        grid_availability=np.array([[1.0, 0.5]]),
        backup_availability=np.array([[1.0, 1.0]]),
        grid_price_per_mwh=np.array([[100.0, 200.0]]),
        period_hours=1.0,
    )
    plan = ReservePlan(grid_contract_mw=6.0, backup_reserve_mw=2.0)
    costs = CostParameters(
        grid_capacity_charge_per_mw_day=10.0,
        backup_reserve_charge_per_mw_day=5.0,
        backup_energy_cost_per_mwh=300.0,
        shortfall_penalty_per_mwh=1000.0,
    )

    # Fixed = 6*10 + 2*5 = 70.
    # t0: load 5, grid 5 => 500.
    # t1: grid cap 3, backup 2, shortfall 3 =>
    #     3*200 + 2*300 + 3*1000 = 4200.
    # Total = 4770; unserved = 3 MWh.
    scenario_cost, unserved = scenario_costs_scalar(
        scenarios,
        plan,
        costs,
    )
    assert math.isclose(scenario_cost[0], 4770.0, abs_tol=1e-12)
    assert math.isclose(unserved[0], 3.0, abs_tol=1e-12)

    vector_cost, vector_unserved = scenario_costs_vectorized(
        scenarios,
        plan,
        costs,
    )
    assert np.array_equal(scenario_cost, vector_cost)
    assert np.array_equal(unserved, vector_unserved)


def self_test() -> None:
    hand_oracle()

    scenarios = generate_scenarios(500, periods=8, seed=7)
    plan = ReservePlan(8.0, 2.5)

    scalar_cost, scalar_unserved = scenario_costs_scalar(
        scenarios,
        plan,
    )
    vector_cost, vector_unserved = scenario_costs_vectorized(
        scenarios,
        plan,
    )
    assert np.allclose(scalar_cost, vector_cost, atol=1e-9, rtol=1e-12)
    assert np.allclose(
        scalar_unserved,
        vector_unserved,
        atol=1e-12,
        rtol=1e-12,
    )

    # Multiprocessing correctness on a small portable smoke set.
    parallel_cost, parallel_unserved = scenario_costs_multiprocessing(
        scenarios,
        plan,
        workers=2,
    )
    assert np.allclose(scalar_cost, parallel_cost, atol=1e-9, rtol=1e-12)
    assert np.allclose(
        scalar_unserved,
        parallel_unserved,
        atol=1e-12,
        rtol=1e-12,
    )

    # CVaR hand check: worst two of 20 values at alpha=0.90.
    values = np.arange(1.0, 21.0)
    assert math.isclose(cvar(values, alpha=0.90), 19.5, abs_tol=1e-12)

    # Candidate grid is finite, unique and exhaustively enumerable.
    plans = candidate_plans()
    assert len(plans) == 315
    assert len(set(plans)) == 315

    # Small optimization smoke run.
    result = run_stochastic_optimization(
        selection_scenarios=500,
        validation_scenarios=1000,
        seed=9,
    )
    assert result.candidate_count == 315
    assert result.selected_plan in plans
    assert result.baseline_plan in plans
    assert result.selected_validation_metrics.mean_cost > 0.0

    print("Parallel Monte Carlo stochastic optimization self-test: OK")


def print_optimization(result: OptimizationResult) -> None:
    s = result.selected_validation_metrics
    b = result.baseline_validation_metrics

    print("=" * 88)
    print("PARALLEL MONTE CARLO STOCHASTIC INDUSTRIAL ENERGY RESERVE PLANNING")
    print("=" * 88)
    print(f"Candidate plans exhaustively checked : {result.candidate_count}")
    print()
    print("Risk-aware selected plan")
    print(f"  grid contract              : {result.selected_plan.grid_contract_mw:.2f} MW")
    print(f"  backup reserve             : {result.selected_plan.backup_reserve_mw:.2f} MW")
    print(f"  validation mean cost       : ${s.mean_cost:,.2f}/day")
    print(f"  validation CVaR95          : ${s.cvar95_cost:,.2f}/day")
    print(f"  P(any shortfall)           : {100*s.probability_any_shortfall:.3f}%")
    print(f"  expected unserved energy   : {s.expected_unserved_energy_mwh:.5f} MWh/day")
    print()
    print("Expected-value baseline")
    print(f"  grid contract              : {result.baseline_plan.grid_contract_mw:.2f} MW")
    print(f"  backup reserve             : {result.baseline_plan.backup_reserve_mw:.2f} MW")
    print(f"  validation mean cost       : ${b.mean_cost:,.2f}/day")
    print(f"  validation CVaR95          : ${b.cvar95_cost:,.2f}/day")
    print(f"  P(any shortfall)           : {100*b.probability_any_shortfall:.3f}%")
    print(f"  expected unserved energy   : {b.expected_unserved_energy_mwh:.5f} MWh/day")
    print()
    print(
        "The selected plan is the exact minimizer over the declared 315-plan "
        "candidate grid for the selection sample. It is not a proof of the "
        "continuous first-stage optimum."
    )


def print_benchmark(rows: Sequence[BenchmarkRow]) -> None:
    print("=" * 88)
    print("MONTE CARLO KERNEL BENCHMARK")
    print("=" * 88)
    for row in rows:
        print(
            f"{row.method:<24} workers={row.workers:<2} "
            f"time={row.seconds:8.4f}s "
            f"speedup={row.speedup_vs_scalar:7.2f}x "
            f"mean=${row.mean_cost:,.3f}"
        )
    print()
    print(
        "Multiprocessing timings include shared-memory setup and process startup. "
        "Speedups are measurements for this machine/run, not portable claims."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--selection-scenarios", type=int, default=12_000)
    parser.add_argument("--validation-scenarios", type=int, default=40_000)
    parser.add_argument("--benchmark-scenarios", type=int, default=40_000)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.self_test:
        self_test()
    elif args.benchmark:
        rows = benchmark_kernels(
            n_scenarios=args.benchmark_scenarios,
            seed=args.seed,
            worker_counts=(1, 2, 4),
            repeats=args.benchmark_repeats,
        )
        print_benchmark(rows)
    else:
        result = run_stochastic_optimization(
            selection_scenarios=args.selection_scenarios,
            validation_scenarios=args.validation_scenarios,
            seed=args.seed,
        )
        print_optimization(result)
