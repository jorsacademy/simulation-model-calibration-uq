from __future__ import annotations
import argparse
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple
import numpy as np
from scipy.stats import t as student_t

@dataclass(frozen=True)
class LineDesign:
    buffer_machining_assembly: int
    buffer_assembly_test: int
    repair_technicians: int

    def __post_init__(self):
        if self.buffer_machining_assembly < 1:
            raise ValueError('buffer_machining_assembly must be >= 1')
        if self.buffer_assembly_test < 1:
            raise ValueError('buffer_assembly_test must be >= 1')
        if self.repair_technicians < 1:
            raise ValueError('repair_technicians must be >= 1')

@dataclass(frozen=True)
class SimulationScenario:
    arrival_times: np.ndarray
    process_times: np.ndarray
    requires_rework: np.ndarray
    rework_success: np.ndarray
    failure_intervals: Tuple[np.ndarray, ...]
    repair_durations: Tuple[np.ndarray, ...]

@dataclass
class ReplicationResult:
    profit_per_hour: float
    good_throughput_per_hour: float
    scrap_rate: float
    mean_cycle_time_hours: float
    mean_wip: float
    machine_utilization: Tuple[float, ...]
    machine_downtime_fraction: Tuple[float, ...]
    machine_blocked_fraction: Tuple[float, ...]
    good_units: int
    scrapped_units: int

@dataclass
class DesignEstimate:
    design: LineDesign
    mean_profit_per_hour: float
    std_profit_per_hour: float
    ci95_low: float
    ci95_high: float
    mean_throughput_per_hour: float
    mean_cycle_time_hours: float
    mean_wip: float
    mean_scrap_rate: float
    replications: int

@dataclass
class OptimizationResult:
    selected: DesignEstimate
    baseline_validation: DesignEstimate
    selected_validation: DesignEstimate
    selection_table: List[DesignEstimate]
    paired_profit_difference_mean: float
    paired_profit_difference_ci95_low: float
    paired_profit_difference_ci95_high: float

@dataclass
class MachineState:
    index: int
    name: str
    queue_index: int
    state: str = 'idle'
    current_job: int | None = None
    remaining_process: float = 0.0
    blocked_job: int | None = None
    failure_budget: float = math.inf
    failure_index: int = 0
    repair_index: int = 0

class ManufacturingLineDES:
    MACHINE_NAMES = ('Machining', 'Assembly', 'Test', 'Rework')
    NUM_MACHINES = 4
    GOOD_UNIT_CONTRIBUTION = 210.0
    SCRAP_COST = 85.0
    WIP_HOLDING_COST_PER_JOB_H = 2.5
    REPAIR_TECHNICIAN_COST_PER_H = 42.0
    BUFFER_SLOT_COST_PER_H = 7.5

    def __init__(self, design: LineDesign, scenario: SimulationScenario, *, warmup_hours: float=8.0, measurement_hours: float=40.0):
        if warmup_hours < 0 or measurement_hours <= 0:
            raise ValueError('invalid simulation horizon')
        self.design = design
        self.scenario = scenario
        self.warmup = float(warmup_hours)
        self.measurement_hours = float(measurement_hours)
        self.horizon = self.warmup + self.measurement_hours
        self.queues = [deque(), deque(), deque(), deque()]
        self.queue_caps = [math.inf, design.buffer_machining_assembly, design.buffer_assembly_test, math.inf]
        self.machines = [MachineState(i, self.MACHINE_NAMES[i], i) for i in range(self.NUM_MACHINES)]
        for m in self.machines:
            m.failure_budget = float(scenario.failure_intervals[m.index][0])
        self.events: List[Tuple[float, int, str, int]] = []
        self.event_counter = 0
        self.tech_busy = 0
        self.repair_wait_queue = deque()
        self.arrival_time_by_job: Dict[int, float] = {}
        self.good_completion_times: Dict[int, float] = {}
        self.scrap_completion_times: Dict[int, float] = {}
        self.wip = 0
        self.last_event_time = 0.0
        self.wip_area = 0.0
        self.machine_state_area = [{'busy': 0.0, 'down': 0.0, 'waiting_repair': 0.0, 'blocked': 0.0} for _ in range(self.NUM_MACHINES)]

    def _push_event(self, time: float, kind: str, item: int) -> None:
        self.event_counter += 1
        heapq.heappush(self.events, (float(time), self.event_counter, kind, int(item)))

    def _accumulate_state_area(self, new_time: float) -> None:
        start = max(self.last_event_time, self.warmup)
        end = min(new_time, self.horizon)
        dt = max(0.0, end - start)
        if dt > 0:
            self.wip_area += self.wip * dt
            for m in self.machines:
                if m.state in self.machine_state_area[m.index]:
                    self.machine_state_area[m.index][m.state] += dt
        self.last_event_time = new_time

    def _next_failure_budget(self, machine: MachineState) -> float:
        machine.failure_index += 1
        values = self.scenario.failure_intervals[machine.index]
        if machine.failure_index >= len(values):
            raise RuntimeError('scenario failure stream exhausted')
        return float(values[machine.failure_index])

    def _next_repair_duration(self, machine: MachineState) -> float:
        values = self.scenario.repair_durations[machine.index]
        if machine.repair_index >= len(values):
            raise RuntimeError('scenario repair stream exhausted')
        value = float(values[machine.repair_index])
        machine.repair_index += 1
        return value

    def _job_process_time(self, job: int, machine_index: int) -> float:
        return float(self.scenario.process_times[job, machine_index])

    def _schedule_processing(self, machine: MachineState, now: float) -> None:
        if machine.current_job is None or machine.remaining_process <= 0:
            raise RuntimeError('invalid busy-machine state')
        machine.state = 'busy'
        if machine.remaining_process <= machine.failure_budget + 1e-12:
            self._push_event(now + machine.remaining_process, 'process_complete', machine.index)
        else:
            self._push_event(now + machine.failure_budget, 'machine_failure', machine.index)

    def _release_blocked_upstream(self, downstream_machine: int, now: float) -> None:
        upstream = downstream_machine - 1
        if upstream not in (0, 1):
            return
        machine = self.machines[upstream]
        downstream_queue = self.queues[downstream_machine]
        if machine.state == 'blocked' and machine.blocked_job is not None and (len(downstream_queue) < self.queue_caps[downstream_machine]):
            downstream_queue.append(machine.blocked_job)
            machine.blocked_job = None
            machine.state = 'idle'
            machine.current_job = None
            machine.remaining_process = 0.0
            self._try_start_machine(upstream, now)

    def _try_start_machine(self, machine_index: int, now: float) -> None:
        machine = self.machines[machine_index]
        if machine.state != 'idle':
            return
        queue = self.queues[machine.queue_index]
        if not queue:
            return
        job = int(queue.popleft())
        if machine_index in (1, 2):
            self._release_blocked_upstream(machine_index, now)
        machine.current_job = job
        machine.remaining_process = self._job_process_time(job, machine_index)
        self._schedule_processing(machine, now)

    def _start_repair_if_possible(self, now: float) -> None:
        while self.repair_wait_queue and self.tech_busy < self.design.repair_technicians:
            machine_index = int(self.repair_wait_queue.popleft())
            machine = self.machines[machine_index]
            if machine.state != 'waiting_repair':
                continue
            self.tech_busy += 1
            machine.state = 'down'
            duration = self._next_repair_duration(machine)
            self._push_event(now + duration, 'repair_complete', machine_index)

    def _finish_job(self, job: int, good: bool, now: float) -> None:
        self.wip -= 1
        if self.wip < 0:
            raise RuntimeError('negative WIP')
        if good:
            self.good_completion_times[job] = now
        else:
            self.scrap_completion_times[job] = now

    def _route_completed_job(self, machine_index: int, job: int, now: float) -> bool:
        if machine_index == 0:
            target_queue = 1
        elif machine_index == 1:
            target_queue = 2
        elif machine_index == 2:
            if bool(self.scenario.requires_rework[job]):
                self.queues[3].append(job)
                self._try_start_machine(3, now)
            else:
                self._finish_job(job, True, now)
            return True
        else:
            self._finish_job(job, bool(self.scenario.rework_success[job]), now)
            return True
        if len(self.queues[target_queue]) < self.queue_caps[target_queue]:
            self.queues[target_queue].append(job)
            self._try_start_machine(target_queue, now)
            return True
        machine = self.machines[machine_index]
        machine.state = 'blocked'
        machine.blocked_job = job
        return False

    def _handle_process_complete(self, machine_index: int, now: float) -> None:
        machine = self.machines[machine_index]
        if machine.state != 'busy' or machine.current_job is None:
            return
        machine.failure_budget -= machine.remaining_process
        if machine.failure_budget < 0 and machine.failure_budget > -1e-09:
            machine.failure_budget = 0.0
        job = int(machine.current_job)
        machine.remaining_process = 0.0
        free = self._route_completed_job(machine_index, job, now)
        if free:
            machine.current_job = None
            machine.state = 'idle'
            self._try_start_machine(machine_index, now)

    def _handle_failure(self, machine_index: int, now: float) -> None:
        machine = self.machines[machine_index]
        if machine.state != 'busy' or machine.current_job is None:
            return
        operated = machine.failure_budget
        machine.remaining_process -= operated
        machine.failure_budget = 0.0
        machine.state = 'waiting_repair'
        self.repair_wait_queue.append(machine_index)
        self._start_repair_if_possible(now)

    def _handle_repair_complete(self, machine_index: int, now: float) -> None:
        machine = self.machines[machine_index]
        if machine.state != 'down':
            return
        self.tech_busy -= 1
        if self.tech_busy < 0:
            raise RuntimeError('negative technician utilization')
        machine.failure_budget = self._next_failure_budget(machine)
        if machine.current_job is None or machine.remaining_process <= 0:
            raise RuntimeError('repair completed without interrupted job')
        self._schedule_processing(machine, now)
        self._start_repair_if_possible(now)

    def run(self) -> ReplicationResult:
        for job, t in enumerate(self.scenario.arrival_times):
            if t > self.horizon:
                break
            self._push_event(float(t), 'arrival', int(job))
        while self.events:
            now, _, kind, item = heapq.heappop(self.events)
            if now > self.horizon + 1e-12:
                break
            self._accumulate_state_area(now)
            if kind == 'arrival':
                job = item
                self.arrival_time_by_job[job] = now
                self.wip += 1
                self.queues[0].append(job)
                self._try_start_machine(0, now)
            elif kind == 'process_complete':
                self._handle_process_complete(item, now)
            elif kind == 'machine_failure':
                self._handle_failure(item, now)
            elif kind == 'repair_complete':
                self._handle_repair_complete(item, now)
        self._accumulate_state_area(self.horizon)
        arrived = set(self.arrival_time_by_job)
        exited = set(self.good_completion_times) | set(self.scrap_completion_times)
        if set(self.good_completion_times) & set(self.scrap_completion_times):
            raise RuntimeError('job recorded as both good and scrapped')
        active_locations = []
        for q in self.queues:
            active_locations.extend(q)
        for machine in self.machines:
            if machine.current_job is not None:
                active_locations.append(machine.current_job)
        if len(active_locations) != len(set(active_locations)):
            raise RuntimeError('job appears in multiple active locations')
        active = set(active_locations)
        if active & exited:
            raise RuntimeError('exited job still present in line')
        if arrived != active | exited:
            raise RuntimeError(f'job conservation violated: missing={sorted(arrived - (active | exited))[:10]}, extra={sorted((active | exited) - arrived)[:10]}')
        if self.wip != len(active):
            raise RuntimeError('WIP state is inconsistent with active jobs')
        measured_good = [j for j, completion in self.good_completion_times.items() if self.warmup <= completion <= self.horizon]
        measured_scrap = [j for j, completion in self.scrap_completion_times.items() if self.warmup <= completion <= self.horizon]
        good = len(measured_good)
        scrap = len(measured_scrap)
        throughput = good / self.measurement_hours
        scrap_rate = scrap / max(good + scrap, 1)
        cycle_times = [completion - self.arrival_time_by_job[j] for j, completion in self.good_completion_times.items() if self.arrival_time_by_job[j] >= self.warmup and completion <= self.horizon] + [completion - self.arrival_time_by_job[j] for j, completion in self.scrap_completion_times.items() if self.arrival_time_by_job[j] >= self.warmup and completion <= self.horizon]
        mean_cycle = float(np.mean(cycle_times)) if cycle_times else math.inf
        mean_wip = self.wip_area / self.measurement_hours
        utilization = tuple((area['busy'] / self.measurement_hours for area in self.machine_state_area))
        downtime = tuple(((area['down'] + area['waiting_repair']) / self.measurement_hours for area in self.machine_state_area))
        blocked = tuple((area['blocked'] / self.measurement_hours for area in self.machine_state_area))
        gross_contribution_per_h = (good * self.GOOD_UNIT_CONTRIBUTION - scrap * self.SCRAP_COST) / self.measurement_hours
        operating_cost_per_h = self.WIP_HOLDING_COST_PER_JOB_H * mean_wip + self.REPAIR_TECHNICIAN_COST_PER_H * self.design.repair_technicians + self.BUFFER_SLOT_COST_PER_H * (self.design.buffer_machining_assembly + self.design.buffer_assembly_test)
        profit = gross_contribution_per_h - operating_cost_per_h
        return ReplicationResult(profit_per_hour=float(profit), good_throughput_per_hour=float(throughput), scrap_rate=float(scrap_rate), mean_cycle_time_hours=float(mean_cycle), mean_wip=float(mean_wip), machine_utilization=utilization, machine_downtime_fraction=downtime, machine_blocked_fraction=blocked, good_units=good, scrapped_units=scrap)

def _lognormal_parameters(mean: float, cv: float) -> Tuple[float, float]:
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - 0.5 * sigma2
    return (mu, sigma)

def generate_scenario(seed: int, *, horizon_hours: float=48.0, arrival_rate_per_hour: float=6.8, max_jobs: int=1000, failure_stream_length: int=256) -> SimulationScenario:
    rng = np.random.default_rng(seed)
    interarrivals = rng.exponential(1.0 / arrival_rate_per_hour, size=max_jobs)
    arrival_times = np.cumsum(interarrivals)
    arrival_times = arrival_times[arrival_times <= horizon_hours]
    n_jobs = len(arrival_times)
    if n_jobs == 0:
        raise RuntimeError('scenario generated no jobs')
    means = np.array([0.105, 0.125, 0.06, 0.155])
    cvs = np.array([0.18, 0.22, 0.15, 0.25])
    process = np.empty((n_jobs, 4), dtype=float)
    for m in range(4):
        mu, sigma = _lognormal_parameters(means[m], cvs[m])
        process[:, m] = rng.lognormal(mu, sigma, size=n_jobs)
    requires_rework = rng.random(n_jobs) < 0.085
    rework_success = rng.random(n_jobs) < 0.9
    mtbf = np.array([7.0, 6.0, 9.0, 12.0])
    mttr_mean = np.array([0.55, 0.7, 0.45, 0.5])
    mttr_cv = np.array([0.35, 0.4, 0.3, 0.35])
    failure_intervals = []
    repair_durations = []
    for m in range(4):
        failure_intervals.append(rng.exponential(mtbf[m], size=failure_stream_length))
        mu, sigma = _lognormal_parameters(mttr_mean[m], mttr_cv[m])
        repair_durations.append(rng.lognormal(mu, sigma, size=failure_stream_length))
    return SimulationScenario(arrival_times=np.asarray(arrival_times, dtype=float), process_times=process, requires_rework=requires_rework, rework_success=rework_success, failure_intervals=tuple(failure_intervals), repair_durations=tuple(repair_durations))

def make_scenarios(seeds: Iterable[int], *, warmup_hours: float=8.0, measurement_hours: float=40.0) -> List[SimulationScenario]:
    horizon = warmup_hours + measurement_hours
    return [generate_scenario(seed, horizon_hours=horizon) for seed in seeds]

def estimate_design(design: LineDesign, scenarios: List[SimulationScenario], *, warmup_hours: float=8.0, measurement_hours: float=40.0) -> DesignEstimate:
    results = [ManufacturingLineDES(design, scenario, warmup_hours=warmup_hours, measurement_hours=measurement_hours).run() for scenario in scenarios]
    profit = np.array([r.profit_per_hour for r in results], dtype=float)
    throughput = np.array([r.good_throughput_per_hour for r in results], dtype=float)
    cycle = np.array([r.mean_cycle_time_hours for r in results], dtype=float)
    wip = np.array([r.mean_wip for r in results], dtype=float)
    scrap = np.array([r.scrap_rate for r in results], dtype=float)
    n = len(results)
    if n < 2:
        raise ValueError('at least two replications required for CI')
    std = float(np.std(profit, ddof=1))
    half = float(student_t.ppf(0.975, df=n - 1) * std / math.sqrt(n))
    return DesignEstimate(design=design, mean_profit_per_hour=float(np.mean(profit)), std_profit_per_hour=std, ci95_low=float(np.mean(profit) - half), ci95_high=float(np.mean(profit) + half), mean_throughput_per_hour=float(np.mean(throughput)), mean_cycle_time_hours=float(np.mean(cycle)), mean_wip=float(np.mean(wip)), mean_scrap_rate=float(np.mean(scrap)), replications=n)

def candidate_designs() -> List[LineDesign]:
    buffer_values = (2, 4, 6, 8, 10)
    technicians = (1, 2)
    return [LineDesign(b1, b2, tech) for b1 in buffer_values for b2 in buffer_values for tech in technicians]

def optimize_design(*, selection_replications: int=12, validation_replications: int=30, seed: int=42, baseline: LineDesign=LineDesign(4, 4, 1)) -> OptimizationResult:
    if selection_replications < 2 or validation_replications < 2:
        raise ValueError('need at least two selection/validation replications')
    selection_scenarios = make_scenarios((seed + i for i in range(selection_replications)))
    estimates = [estimate_design(d, selection_scenarios) for d in candidate_designs()]
    estimates.sort(key=lambda e: (-e.mean_profit_per_hour, e.design.repair_technicians, e.design.buffer_machining_assembly + e.design.buffer_assembly_test))
    selected_design = estimates[0].design
    validation_scenarios = make_scenarios((seed + 100000 + i for i in range(validation_replications)))
    selected_validation = estimate_design(selected_design, validation_scenarios)
    baseline_validation = estimate_design(baseline, validation_scenarios)
    diff_mean, diff_low, diff_high = paired_profit_difference(selected_design, baseline, validation_scenarios)
    return OptimizationResult(selected=estimates[0], baseline_validation=baseline_validation, selected_validation=selected_validation, selection_table=estimates, paired_profit_difference_mean=diff_mean, paired_profit_difference_ci95_low=diff_low, paired_profit_difference_ci95_high=diff_high)

def paired_profit_difference(design_a: LineDesign, design_b: LineDesign, scenarios: List[SimulationScenario]) -> Tuple[float, float, float]:
    diff = []
    for scenario in scenarios:
        a = ManufacturingLineDES(design_a, scenario).run().profit_per_hour
        b = ManufacturingLineDES(design_b, scenario).run().profit_per_hour
        diff.append(a - b)
    d = np.asarray(diff, dtype=float)
    n = len(d)
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return (mean, mean - half, mean + half)

def deterministic_blocking_scenario() -> SimulationScenario:
    return SimulationScenario(arrival_times=np.array([0.0, 0.0, 0.0]), process_times=np.array([[0.5, 2.0, 0.1, 1.0], [0.5, 2.0, 0.1, 1.0], [0.5, 2.0, 0.1, 1.0]], dtype=float), requires_rework=np.array([False, False, False]), rework_success=np.array([True, True, True]), failure_intervals=tuple((np.array([100.0, 100.0], dtype=float) for _ in range(4))), repair_durations=tuple((np.array([1.0, 1.0], dtype=float) for _ in range(4))))

def deterministic_single_job_scenario() -> SimulationScenario:
    return SimulationScenario(arrival_times=np.array([0.0]), process_times=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float), requires_rework=np.array([False]), rework_success=np.array([True]), failure_intervals=tuple((np.array([100.0, 100.0], dtype=float) for _ in range(4))), repair_durations=tuple((np.array([1.0, 1.0], dtype=float) for _ in range(4))))

def self_test() -> None:
    oracle = ManufacturingLineDES(LineDesign(1, 1, 1), deterministic_single_job_scenario(), warmup_hours=0.0, measurement_hours=4.0).run()
    assert oracle.good_units == 1
    assert oracle.scrapped_units == 0
    assert math.isclose(oracle.good_throughput_per_hour, 0.25, abs_tol=1e-12)
    assert math.isclose(oracle.mean_cycle_time_hours, 3.0, abs_tol=1e-12)
    assert math.isclose(oracle.mean_wip, 0.75, abs_tol=1e-12)
    assert np.allclose(oracle.machine_utilization[:3], [0.25, 0.25, 0.25])
    assert math.isclose(oracle.machine_utilization[3], 0.0, abs_tol=1e-12)
    blocking_oracle = ManufacturingLineDES(LineDesign(1, 1, 1), deterministic_blocking_scenario(), warmup_hours=0.0, measurement_hours=5.0).run()
    assert math.isclose(blocking_oracle.machine_blocked_fraction[0], 1.0 / 5.0, abs_tol=1e-12)
    scenario = generate_scenario(123, horizon_hours=16.0)
    design = LineDesign(4, 4, 1)
    a = ManufacturingLineDES(design, scenario, warmup_hours=4.0, measurement_hours=12.0).run()
    b = ManufacturingLineDES(design, scenario, warmup_hours=4.0, measurement_hours=12.0).run()
    assert a == b
    assert a.good_throughput_per_hour > 0.0
    assert 0.0 <= a.scrap_rate <= 1.0
    assert a.mean_wip >= 0.0
    for value in (*a.machine_utilization, *a.machine_downtime_fraction, *a.machine_blocked_fraction):
        assert 0.0 <= value <= 1.0 + 1e-12
    assert np.array_equal(scenario.arrival_times, scenario.arrival_times.copy())
    one = ManufacturingLineDES(LineDesign(4, 4, 1), scenario, warmup_hours=4, measurement_hours=12).run()
    two = ManufacturingLineDES(LineDesign(4, 4, 2), scenario, warmup_hours=4, measurement_hours=12).run()
    assert one.good_throughput_per_hour >= 0
    assert two.good_throughput_per_hour >= 0
    designs = candidate_designs()
    assert len(designs) == 50
    assert len(set(designs)) == 50
    scenarios = [generate_scenario(1000 + i, horizon_hours=16.0) for i in range(4)]
    est = estimate_design(design, scenarios, warmup_hours=4.0, measurement_hours=12.0)
    assert est.ci95_low <= est.mean_profit_per_hour <= est.ci95_high
    print('Manufacturing DES self-test: OK')

def print_result(result: OptimizationResult) -> None:
    s = result.selected
    v = result.selected_validation
    b = result.baseline_validation
    print('=' * 82)
    print('MANUFACTURING DISCRETE-EVENT SIMULATION + DESIGN OPTIMIZATION')
    print('=' * 82)
    print('Selected SAA design')
    print(f'  buffer Machining->Assembly : {s.design.buffer_machining_assembly}')
    print(f'  buffer Assembly->Test      : {s.design.buffer_assembly_test}')
    print(f'  repair technicians         : {s.design.repair_technicians}')
    print(f'  selection mean profit      : {s.mean_profit_per_hour:.2f} $/h')
    print()
    print('Independent validation')
    print(f'  selected profit            : {v.mean_profit_per_hour:.2f} $/h [95% CI {v.ci95_low:.2f}, {v.ci95_high:.2f}]')
    print(f'  baseline profit            : {b.mean_profit_per_hour:.2f} $/h [95% CI {b.ci95_low:.2f}, {b.ci95_high:.2f}]')
    print(f'  selected throughput        : {v.mean_throughput_per_hour:.3f} good units/h')
    print(f'  baseline throughput        : {b.mean_throughput_per_hour:.3f} good units/h')
    print(f'  selected mean cycle time   : {v.mean_cycle_time_hours:.3f} h')
    print(f'  selected mean WIP          : {v.mean_wip:.3f} jobs')
    print(f'  selected scrap rate        : {100 * v.mean_scrap_rate:.2f}%')
    print(f'  paired profit improvement  : {result.paired_profit_difference_mean:.2f} $/h [95% CI {result.paired_profit_difference_ci95_low:.2f}, {result.paired_profit_difference_ci95_high:.2f}]')
    print()
    print('Selection is the exact maximizer of the sample-average objective over the declared 50-design candidate set. It is not a global optimum over all possible factory designs.')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--selection-replications', type=int, default=12)
    parser.add_argument('--validation-replications', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        result = optimize_design(selection_replications=args.selection_replications, validation_replications=args.validation_replications, seed=args.seed)
        print_result(result)
