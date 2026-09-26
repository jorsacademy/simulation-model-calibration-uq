import numpy as np

from simcal_uq.noisy_optimization import (
    QueueDesignProblem,
    estimate_reference_means,
    ocba_target_allocation,
    queue_design_grid,
    run_equal_allocation,
    run_ocba,
    run_stochastic_kriging_ei,
    simple_regret,
)


def test_ocba_target_allocation_preserves_total_budget():
    means = np.array([1.0, 1.1, 1.4, 1.8])
    variances = np.array([0.4, 0.8, 0.3, 1.1])
    allocation = ocba_target_allocation(
        means,
        variances,
        total_target=40,
        minimum_per_alternative=2,
    )
    assert allocation.sum() == 40
    assert np.all(allocation >= 2)


def test_noisy_optimization_methods_respect_budget_and_return_valid_candidates():
    candidates = queue_design_grid(
        service_rates=np.array([1.0, 1.2, 1.4]),
        repair_rates=np.array([0.18, 0.30, 0.42]),
    )
    problem = QueueDesignProblem(
        arrival_rate=0.70,
        failure_rate=0.035,
        horizon=80.0,
    )
    budget = 45

    equal = run_equal_allocation(
        problem,
        candidates,
        total_budget=budget,
        random_state=1,
    )
    ocba = run_ocba(
        problem,
        candidates,
        total_budget=budget,
        initial_replications=3,
        allocation_batch=6,
        random_state=2,
    )
    kriging = run_stochastic_kriging_ei(
        problem,
        candidates,
        total_budget=budget,
        initial_points=5,
        initial_replications=3,
        sequential_replications=2,
        random_state=3,
    )

    for result in (equal, ocba, kriging):
        assert result.budget_used == budget
        assert 0 <= result.selected_index < len(candidates)
        assert result.counts.sum() == budget
        assert result.selected_point.shape == (2,)


def test_simple_regret_is_nonnegative_against_reference():
    candidates = queue_design_grid(
        service_rates=np.array([1.0, 1.3]),
        repair_rates=np.array([0.2, 0.4]),
    )
    problem = QueueDesignProblem(
        arrival_rate=0.65,
        failure_rate=0.03,
        horizon=60.0,
    )
    result = run_equal_allocation(
        problem,
        candidates,
        total_budget=16,
        random_state=10,
    )
    reference = estimate_reference_means(
        problem,
        candidates,
        replications=12,
        random_state=99,
    )
    assert simple_regret(result, reference) >= -1e-12
