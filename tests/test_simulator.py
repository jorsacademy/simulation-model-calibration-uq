import numpy as np

from simcal_uq import QueueParameters, simulate_queue, simulate_replications


def test_simulator_is_reproducible_and_sane():
    params = QueueParameters(0.8, 1.3, 0.05, 0.5)
    a = simulate_queue(params, horizon=250.0, seed=7)
    b = simulate_queue(params, horizon=250.0, seed=7)
    assert a == b
    assert 0.0 < a.throughput_rate < params.arrival_rate * 1.2
    assert 0.0 <= a.utilization <= a.availability <= 1.0
    assert a.mean_cycle_time > 0.0
    assert a.mean_queue >= 0.0


def test_higher_service_rate_reduces_cycle_time_on_average():
    seeds = range(20)
    slow = simulate_replications(QueueParameters(0.78, 1.05, 0.04, 0.55), seeds, horizon=300.0)
    fast = simulate_replications(QueueParameters(0.78, 1.60, 0.04, 0.55), seeds, horizon=300.0)
    assert np.mean(fast[:, 1]) < np.mean(slow[:, 1])
