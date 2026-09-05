from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


METRIC_NAMES = (
    "throughput_rate",
    "mean_cycle_time",
    "utilization",
    "availability",
    "mean_queue",
    "breakdown_rate",
)


@dataclass(frozen=True)
class QueueParameters:
    arrival_rate: float
    service_rate: float
    failure_rate: float
    repair_rate: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.arrival_rate, self.service_rate, self.failure_rate, self.repair_rate],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: Iterable[float]) -> QueueParameters:
        vals = np.asarray(list(values), dtype=float)
        if vals.shape != (4,):
            raise ValueError("expected four parameters")
        return cls(*map(float, vals))

    def validate(self) -> None:
        if min(self.arrival_rate, self.service_rate, self.failure_rate, self.repair_rate) <= 0:
            raise ValueError("all rates must be positive")


@dataclass(frozen=True)
class SimulationResult:
    throughput_rate: float
    mean_cycle_time: float
    utilization: float
    availability: float
    mean_queue: float
    breakdown_rate: float
    completed_jobs: int
    breakdowns: int

    def metric_vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in METRIC_NAMES], dtype=float)


def simulate_queue(
    params: QueueParameters,
    *,
    horizon: float = 500.0,
    seed: int = 0,
) -> SimulationResult:
    """Simulate a single-server queue with preempt-resume random breakdowns.

    Arrivals, service requirements, time-to-failure and repair times are exponential.
    Failures occur only while the machine is up; an interrupted service resumes after repair.
    """
    params.validate()
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(seed)
    inf = float("inf")

    t = 0.0
    last_t = 0.0
    next_arrival = rng.exponential(1.0 / params.arrival_rate)
    next_failure = rng.exponential(1.0 / params.failure_rate)
    next_repair = inf
    next_completion = inf

    machine_up = True
    busy = False
    queue: list[float] = []
    current_arrival = 0.0
    remaining_service = 0.0

    completed = 0
    breakdowns = 0
    cycle_times: list[float] = []
    queue_area = 0.0
    busy_up_time = 0.0
    up_time = 0.0

    def start_service(now: float) -> tuple[bool, float, float, float]:
        if not queue:
            return False, 0.0, 0.0, inf
        arrival = queue.pop(0)
        service = rng.exponential(1.0 / params.service_rate)
        return True, arrival, service, now + service

    while True:
        event_time = min(next_arrival, next_failure, next_repair, next_completion, horizon)
        dt = event_time - last_t
        queue_area += len(queue) * dt
        if machine_up:
            up_time += dt
        if machine_up and busy:
            busy_up_time += dt
        t = event_time
        last_t = event_time

        if t >= horizon - 1e-12:
            break

        if next_arrival <= min(next_failure, next_repair, next_completion):
            queue.append(t)
            next_arrival = t + rng.exponential(1.0 / params.arrival_rate)
            if machine_up and not busy:
                busy, current_arrival, remaining_service, next_completion = start_service(t)
            continue

        if next_failure <= min(next_repair, next_completion):
            machine_up = False
            breakdowns += 1
            next_failure = inf
            next_repair = t + rng.exponential(1.0 / params.repair_rate)
            if busy:
                remaining_service = max(0.0, next_completion - t)
                next_completion = inf
            continue

        if next_repair <= next_completion:
            machine_up = True
            next_repair = inf
            next_failure = t + rng.exponential(1.0 / params.failure_rate)
            if busy:
                next_completion = t + remaining_service
            elif queue:
                busy, current_arrival, remaining_service, next_completion = start_service(t)
            continue

        completed += 1
        cycle_times.append(t - current_arrival)
        busy = False
        remaining_service = 0.0
        next_completion = inf
        if machine_up and queue:
            busy, current_arrival, remaining_service, next_completion = start_service(t)

    mean_cycle = float(np.mean(cycle_times)) if cycle_times else float(horizon)
    return SimulationResult(
        throughput_rate=completed / horizon,
        mean_cycle_time=mean_cycle,
        utilization=busy_up_time / horizon,
        availability=up_time / horizon,
        mean_queue=queue_area / horizon,
        breakdown_rate=breakdowns / horizon,
        completed_jobs=completed,
        breakdowns=breakdowns,
    )


def simulate_replications(
    params: QueueParameters,
    seeds: Iterable[int],
    *,
    horizon: float = 500.0,
) -> np.ndarray:
    rows = [
        simulate_queue(params, horizon=horizon, seed=int(seed)).metric_vector()
        for seed in seeds
    ]
    if not rows:
        raise ValueError("at least one seed is required")
    return np.vstack(rows)
