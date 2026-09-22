import unittest

import numpy as np

from cpp_accelerated_sim import (
    Policy,
    candidate_policies,
    cpp_available,
    generate_draws,
    optimize_policy,
    simulate_cpp,
    simulate_python,
)
from cpp_accelerated_sim.model import ScenarioDraws, cvar95


class ModelTests(unittest.TestCase):
    def test_candidate_grid_is_complete(self):
        policies = candidate_policies()
        self.assertEqual(len(policies), 27)
        self.assertEqual(len(set(policies)), 27)

    def test_draw_generation_is_reproducible(self):
        a = generate_draws(5, days=4, assets=3, seed=9)
        b = generate_draws(5, days=4, assets=3, seed=9)
        np.testing.assert_array_equal(a.degradation_u, b.degradation_u)
        np.testing.assert_array_equal(a.maintenance_u, b.maintenance_u)

    def test_python_hand_oracle(self):
        degradation = np.zeros((1, 3, 1), dtype=np.float64)
        maintenance = np.zeros((1, 3, 1), dtype=np.float64)
        result = simulate_python(
            ScenarioDraws(degradation, maintenance),
            Policy(2, 1, 0),
        )
        self.assertAlmostEqual(result.total_cost[0], 3740.0, places=9)
        self.assertAlmostEqual(result.maintenance_events[0], 1.0, places=9)


    def test_cvar95_hand_check(self):
        values = np.arange(1.0, 21.0)
        self.assertAlmostEqual(cvar95(values), 20.0, places=12)

    def test_standby_spare_cannot_increase_lost_unit_days(self):
        draws = generate_draws(12, days=25, assets=6, seed=22)
        without_spare = simulate_python(draws, Policy(2, 1, 0))
        with_spare = simulate_python(draws, Policy(2, 1, 1))
        self.assertTrue(np.all(with_spare.lost_unit_days <= without_spare.lost_unit_days + 1e-12))

    def test_small_finite_grid_optimization_returns_declared_policy(self):
        result = optimize_policy(
            selection_scenarios=12,
            validation_scenarios=20,
            days=20,
            assets=5,
            seed=31,
            backend="python",
        )
        self.assertEqual(result.candidate_count, 27)
        self.assertIn(result.selected_policy, candidate_policies())
        self.assertEqual(result.backend, "python")

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            Policy(0, 1, 0)

    @unittest.skipUnless(cpp_available(), "C++ extension not built")
    def test_cpp_matches_python_trajectory_vectors(self):
        draws = generate_draws(40, days=20, assets=6, seed=17)
        for policy in (Policy(1, 1, 0), Policy(2, 2, 1), Policy(3, 3, 2)):
            py = simulate_python(draws, policy)
            cpp = simulate_cpp(draws, policy)
            np.testing.assert_allclose(py.total_cost, cpp.total_cost, rtol=0, atol=1e-9)
            np.testing.assert_allclose(py.lost_unit_days, cpp.lost_unit_days, rtol=0, atol=1e-9)
            np.testing.assert_array_equal(py.failures, cpp.failures)
            np.testing.assert_array_equal(py.maintenance_events, cpp.maintenance_events)


if __name__ == "__main__":
    unittest.main()
