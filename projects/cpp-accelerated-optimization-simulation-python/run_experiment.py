from __future__ import annotations

import argparse

from cpp_accelerated_sim import benchmark, cpp_available, optimize_policy


def print_optimization(result) -> None:
    s = result.selected_validation_metrics
    b = result.baseline_validation_metrics
    p = result.selected_policy
    print("=" * 88)
    print("C++-ACCELERATED INDUSTRIAL FLEET RELIABILITY / MAINTENANCE OPTIMIZATION")
    print("=" * 88)
    print(f"Backend                         : {result.backend}")
    print(f"Candidate policies checked      : {result.candidate_count}")
    print("Selected policy")
    print(f"  maintenance threshold state   : {p.threshold_state}")
    print(f"  staffed crew slots/day        : {p.crew_slots}")
    print(f"  standby spare units           : {p.standby_spares}")
    print(f"  validation mean cost          : {s.mean_cost:,.2f}")
    print(f"  validation CVaR95             : {s.cvar95_cost:,.2f}")
    print(f"  mean lost unit-days           : {s.mean_lost_unit_days:.3f}")
    print(f"  mean failures                 : {s.mean_failures:.3f}")
    print(f"  mean maintenance events       : {s.mean_maintenance_events:.3f}")
    print("Baseline: repair only after failure, one crew slot, no standby spare")
    print(f"  validation mean cost          : {b.mean_cost:,.2f}")
    print(f"  validation CVaR95             : {b.cvar95_cost:,.2f}")
    print(f"  mean lost unit-days           : {b.mean_lost_unit_days:.3f}")
    print(f"  mean failures                 : {b.mean_failures:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--selection-scenarios", type=int, default=1200)
    parser.add_argument("--validation-scenarios", type=int, default=3000)
    parser.add_argument("--benchmark-scenarios", type=int, default=1500)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--assets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.benchmark:
        if not cpp_available():
            raise SystemExit("C++ extension is not available. Install/build the package first.")
        for row in benchmark(
            scenarios=args.benchmark_scenarios,
            days=args.days,
            assets=args.assets,
            seed=args.seed,
            repeats=args.repeats,
        ):
            print(
                f"{row.backend:<18} time={row.seconds:.6f}s "
                f"speedup={row.speedup_vs_python:.2f}x mean_cost={row.mean_cost:,.3f}"
            )
        return

    result = optimize_policy(
        selection_scenarios=args.selection_scenarios,
        validation_scenarios=args.validation_scenarios,
        days=args.days,
        assets=args.assets,
        seed=args.seed,
    )
    print_optimization(result)


if __name__ == "__main__":
    main()
