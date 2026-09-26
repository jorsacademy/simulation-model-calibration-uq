from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simcal_uq.noisy_optimization import (
    QueueDesignProblem,
    estimate_reference_means,
    queue_design_grid,
    run_equal_allocation,
    run_ocba,
    run_stochastic_kriging_ei,
    simple_regret,
)


def _serialize_result(result, reference: np.ndarray) -> dict:
    selected_mean = float(reference[result.selected_index])
    observed_mean = result.sample_means[result.selected_index]
    return {
        "method": result.method,
        "budget_used": int(result.budget_used),
        "selected_index": int(result.selected_index),
        "selected_point": result.selected_point.tolist(),
        "selected_reference_mean": selected_mean,
        "simple_regret": simple_regret(result, reference),
        "selected_sample_mean": (
            float(observed_mean) if np.isfinite(observed_mean) else None
        ),
        "sampled_alternatives": int(np.sum(result.counts > 0)),
        "max_replications_at_one_alternative": int(np.max(result.counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--reference-reps", type=int, default=60)
    parser.add_argument("--horizon", type=float, default=250.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/noisy_optimization.json"),
    )
    args = parser.parse_args()

    if args.grid_size < 3:
        raise ValueError("grid-size must be at least 3")

    service_rates = np.linspace(0.95, 1.55, args.grid_size)
    repair_rates = np.linspace(0.12, 0.48, args.grid_size)
    candidates = queue_design_grid(
        service_rates=service_rates,
        repair_rates=repair_rates,
    )
    if args.budget < 3 * len(candidates):
        raise ValueError(
            "budget must be at least three replications per candidate "
            "so equal allocation and OCBA are both defined"
        )

    problem = QueueDesignProblem(horizon=args.horizon)
    reference = estimate_reference_means(
        problem,
        candidates,
        replications=args.reference_reps,
        random_state=args.seed + 10000,
    )

    equal = run_equal_allocation(
        problem,
        candidates,
        total_budget=args.budget,
        random_state=args.seed + 1,
    )
    ocba = run_ocba(
        problem,
        candidates,
        total_budget=args.budget,
        initial_replications=3,
        allocation_batch=max(6, args.grid_size),
        random_state=args.seed + 2,
    )
    kriging = run_stochastic_kriging_ei(
        problem,
        candidates,
        total_budget=args.budget,
        initial_points=min(max(6, args.grid_size), len(candidates)),
        initial_replications=3,
        sequential_replications=2,
        random_state=args.seed + 3,
    )

    payload = {
        "problem": {
            "decision_variables": ["service_rate", "repair_rate"],
            "grid_size": int(args.grid_size),
            "alternatives": int(len(candidates)),
            "simulation_budget_per_method": int(args.budget),
            "reference_replications_per_alternative": int(args.reference_reps),
            "horizon": float(args.horizon),
        },
        "reference": {
            "best_index": int(np.argmin(reference)),
            "best_point": candidates[int(np.argmin(reference))].tolist(),
            "best_mean_cost": float(np.min(reference)),
            "interpretation": (
                "Reference means use an evaluation-only Monte Carlo budget and are not "
                "available to any search policy."
            ),
        },
        "methods": [
            _serialize_result(equal, reference),
            _serialize_result(ocba, reference),
            _serialize_result(kriging, reference),
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
