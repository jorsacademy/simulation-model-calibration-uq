import math
import unittest

import numpy as np

from parallel_monte_carlo_stochastic_optimization import (
    CostParameters,
    ReservePlan,
    ScenarioSet,
    candidate_plans,
    cvar,
    generate_scenarios,
    optimize_candidate_grid,
    scenario_costs_multiprocessing,
    scenario_costs_scalar,
    scenario_costs_vectorized,
    summarize_plan,
)


class ParallelMonteCarloTests(unittest.TestCase):
    def test_hand_recourse_oracle(self):
        scenarios = ScenarioSet(
            net_load_mw=np.array([[5.0, 8.0]]),
            grid_availability=np.array([[1.0, 0.5]]),
            backup_availability=np.array([[1.0, 1.0]]),
            grid_price_per_mwh=np.array([[100.0, 200.0]]),
            period_hours=1.0,
        )
        plan = ReservePlan(6.0, 2.0)
        costs = CostParameters(
            grid_capacity_charge_per_mw_day=10.0,
            backup_reserve_charge_per_mw_day=5.0,
            backup_energy_cost_per_mwh=300.0,
            shortfall_penalty_per_mwh=1000.0,
        )
        cost, unserved = scenario_costs_scalar(scenarios, plan, costs)
        self.assertTrue(math.isclose(cost[0], 4770.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(unserved[0], 3.0, abs_tol=1e-12))

    def test_vectorized_matches_scalar_reference(self):
        scenarios = generate_scenarios(700, periods=8, seed=17)
        plan = ReservePlan(8.5, 3.0)
        scalar = scenario_costs_scalar(scenarios, plan)
        vector = scenario_costs_vectorized(scenarios, plan)
        np.testing.assert_allclose(scalar[0], vector[0], atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(scalar[1], vector[1], atol=1e-12, rtol=1e-12)

    def test_shared_memory_multiprocessing_matches_scalar(self):
        scenarios = generate_scenarios(600, periods=8, seed=23)
        plan = ReservePlan(8.0, 2.5)
        scalar = scenario_costs_scalar(scenarios, plan)
        parallel = scenario_costs_multiprocessing(scenarios, plan, workers=2)
        np.testing.assert_allclose(scalar[0], parallel[0], atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(scalar[1], parallel[1], atol=1e-12, rtol=1e-12)

    def test_scenario_generation_is_reproducible(self):
        a = generate_scenarios(100, periods=6, seed=31)
        b = generate_scenarios(100, periods=6, seed=31)
        np.testing.assert_array_equal(a.net_load_mw, b.net_load_mw)
        np.testing.assert_array_equal(a.grid_availability, b.grid_availability)
        np.testing.assert_array_equal(a.backup_availability, b.backup_availability)
        np.testing.assert_array_equal(a.grid_price_per_mwh, b.grid_price_per_mwh)

    def test_cvar_hand_oracle(self):
        values = np.arange(1.0, 21.0)
        self.assertTrue(math.isclose(cvar(values, alpha=0.90), 19.5, abs_tol=1e-12))

    def test_candidate_grid_is_complete_and_unique(self):
        plans = candidate_plans()
        self.assertEqual(len(plans), 315)
        self.assertEqual(len(set(plans)), 315)
        self.assertIn(ReservePlan(6.0, 0.0), plans)
        self.assertIn(ReservePlan(13.0, 10.0), plans)

    def test_grid_optimizer_minimizes_declared_sample_objective(self):
        scenarios = generate_scenarios(350, periods=8, seed=41)
        selected, metrics = optimize_candidate_grid(scenarios, risk_weight=0.35)
        self.assertIn(selected, candidate_plans())

        objectives = []
        for plan in candidate_plans():
            cost, unserved = scenario_costs_vectorized(scenarios, plan)
            objectives.append(
                summarize_plan(
                    cost,
                    unserved,
                    risk_weight=0.35,
                ).risk_objective
            )
        self.assertTrue(
            math.isclose(
                metrics.risk_objective,
                min(objectives),
                abs_tol=1e-9,
            )
        )


if __name__ == "__main__":
    unittest.main()
