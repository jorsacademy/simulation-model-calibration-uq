from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simcal_uq import (
    DEFAULT_BOUNDS,
    METRIC_NAMES,
    QueueParameters,
    bootstrap_calibration,
    calibrate,
    generate_observations,
    named_metric_intervals,
    named_parameter_intervals,
    propagate_parameter_uncertainty,
    propagate_predictive_uncertainty,
    simulate_replications,
    sobol_jansen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible calibration + UQ study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=float, default=350.0)
    parser.add_argument("--observed-reps", type=int, default=18)
    parser.add_argument("--calibration-reps", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=16)
    parser.add_argument("--sobol-base", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("artifacts/study.json"))
    args = parser.parse_args()

    true_params = QueueParameters(0.82, 1.28, 0.055, 0.48)
    observed_seeds = range(args.seed, args.seed + args.observed_reps)
    calibration_seeds = tuple(
        range(args.seed + 1000, args.seed + 1000 + args.calibration_reps)
    )
    observations = generate_observations(true_params, observed_seeds, horizon=args.horizon)

    fit = calibrate(
        observations,
        simulation_seeds=calibration_seeds,
        horizon=args.horizon,
        max_nfev=70,
    )
    boot = bootstrap_calibration(
        observations,
        fit,
        simulation_seeds=calibration_seeds,
        n_bootstrap=args.bootstrap,
        horizon=args.horizon,
        seed=args.seed + 2000,
        max_nfev=35,
    )
    parameter_propagation = propagate_parameter_uncertainty(
        boot.parameter_samples,
        evaluation_seeds=range(args.seed + 3000, args.seed + 3003),
        horizon=args.horizon,
    )
    predictive_propagation = propagate_predictive_uncertainty(
        boot.parameter_samples,
        horizon=args.horizon,
        seed=args.seed + 3500,
    )

    sens_seeds = tuple(range(args.seed + 4000, args.seed + 4003))

    def throughput_model(x: np.ndarray) -> float:
        return float(
            simulate_replications(
                QueueParameters.from_array(x), sens_seeds, horizon=args.horizon * 0.6
            )[:, 0].mean()
        )

    sobol = sobol_jansen(
        throughput_model,
        DEFAULT_BOUNDS,
        n_base=args.sobol_base,
        seed=args.seed + 5000,
    )

    true_array = true_params.as_array()
    est_array = fit.parameters.as_array()
    relative_error = np.abs(est_array - true_array) / true_array
    coverage = (boot.lower <= true_array) & (true_array <= boot.upper)

    payload = {
        "true_parameters": true_array.tolist(),
        "estimated_parameters": est_array.tolist(),
        "relative_parameter_error": relative_error.tolist(),
        "max_relative_parameter_error": float(relative_error.max()),
        "calibration_success": fit.success,
        "calibration_cost": fit.cost,
        "calibration_nfev": fit.nfev,
        "observed_metrics": dict(
            zip(METRIC_NAMES, map(float, fit.observed_metrics), strict=True)
        ),
        "predicted_metrics": dict(
            zip(METRIC_NAMES, map(float, fit.predicted_metrics), strict=True)
        ),
        "bootstrap_success_rate": boot.success_rate,
        "bootstrap_contains_truth": coverage.tolist(),
        "bootstrap_coverage_fraction": float(np.mean(coverage)),
        "parameter_intervals": named_parameter_intervals(boot),
        "parameter_only_metric_intervals": named_metric_intervals(parameter_propagation),
        "predictive_metric_intervals": named_metric_intervals(predictive_propagation),
        "throughput_sobol_first_order": sobol.first_order.tolist(),
        "throughput_sobol_total_order": sobol.total_order.tolist(),
        "config": vars(args) | {"output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
