import math
import unittest

import numpy as np

from manufacturing_des_optimization import (
    LineDesign,
    ManufacturingLineDES,
    candidate_designs,
    deterministic_blocking_scenario,
    deterministic_single_job_scenario,
    estimate_design,
    generate_scenario,
    optimize_design,
)


class ManufacturingDESTests(unittest.TestCase):
    def test_single_job_hand_oracle(self):
        result = ManufacturingLineDES(
            LineDesign(1, 1, 1),
            deterministic_single_job_scenario(),
            warmup_hours=0.0,
            measurement_hours=4.0,
        ).run()

        self.assertEqual(result.good_units, 1)
        self.assertEqual(result.scrapped_units, 0)
        self.assertTrue(math.isclose(
            result.good_throughput_per_hour,
            0.25,
            abs_tol=1e-12,
        ))
        self.assertTrue(math.isclose(
            result.mean_cycle_time_hours,
            3.0,
            abs_tol=1e-12,
        ))
        self.assertTrue(math.isclose(
            result.mean_wip,
            0.75,
            abs_tol=1e-12,
        ))
        np.testing.assert_allclose(
            result.machine_utilization[:3],
            [0.25, 0.25, 0.25],
        )

    def test_finite_buffer_blocking_hand_oracle(self):
        result = ManufacturingLineDES(
            LineDesign(1, 1, 1),
            deterministic_blocking_scenario(),
            warmup_hours=0.0,
            measurement_hours=5.0,
        ).run()

        self.assertTrue(math.isclose(
            result.machine_blocked_fraction[0],
            0.2,
            abs_tol=1e-12,
        ))

    def test_fixed_scenario_is_exactly_reproducible(self):
        scenario = generate_scenario(1234, horizon_hours=16.0)
        design = LineDesign(4, 2, 1)

        a = ManufacturingLineDES(
            design,
            scenario,
            warmup_hours=4.0,
            measurement_hours=12.0,
        ).run()
        b = ManufacturingLineDES(
            design,
            scenario,
            warmup_hours=4.0,
            measurement_hours=12.0,
        ).run()

        self.assertEqual(a, b)

    def test_candidate_grid_is_complete_and_unique(self):
        designs = candidate_designs()
        self.assertEqual(len(designs), 50)
        self.assertEqual(len(set(designs)), 50)
        self.assertIn(LineDesign(2, 2, 1), designs)
        self.assertIn(LineDesign(10, 10, 2), designs)

    def test_estimator_produces_valid_confidence_interval(self):
        scenarios = [
            generate_scenario(2000 + i, horizon_hours=16.0)
            for i in range(5)
        ]
        estimate = estimate_design(
            LineDesign(4, 4, 1),
            scenarios,
            warmup_hours=4.0,
            measurement_hours=12.0,
        )
        self.assertLessEqual(
            estimate.ci95_low,
            estimate.mean_profit_per_hour,
        )
        self.assertGreaterEqual(
            estimate.ci95_high,
            estimate.mean_profit_per_hour,
        )
        self.assertGreater(estimate.mean_throughput_per_hour, 0.0)

    def test_short_optimization_returns_candidate_and_independent_validation(self):
        result = optimize_design(
            selection_replications=4,
            validation_replications=6,
            seed=7,
        )
        self.assertIn(result.selected.design, candidate_designs())
        self.assertEqual(result.selected.replications, 4)
        self.assertEqual(result.selected_validation.replications, 6)
        self.assertEqual(result.baseline_validation.replications, 6)
        self.assertLessEqual(
            result.paired_profit_difference_ci95_low,
            result.paired_profit_difference_mean,
        )
        self.assertGreaterEqual(
            result.paired_profit_difference_ci95_high,
            result.paired_profit_difference_mean,
        )


if __name__ == "__main__":
    unittest.main()
