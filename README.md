# Simulation Model Calibration & Uncertainty Quantification

A reproducible research implementation for calibrating a stochastic discrete-event simulation, quantifying parameter uncertainty, propagating that uncertainty to operational KPIs, and measuring global sensitivity with Sobol indices.

The project deliberately separates four questions that are often conflated in simulation studies:

1. **Calibration:** which simulator parameters best reproduce observed system summaries?
2. **Parameter uncertainty:** how stable are those parameters to the observed replication sample?
3. **Predictive uncertainty:** how much uncertainty remains in future KPI predictions after combining parameter uncertainty with intrinsic simulator randomness?
4. **Sensitivity:** which uncertain inputs explain the variance of an output KPI?

The implementation uses NumPy and SciPy only, so the complete study runs in GitHub Actions without commercial solvers, external services, or hidden data.

## Research scope

The benchmark is a repairable single-machine production queue with Poisson job arrivals, exponential service requirements, stochastic failures, stochastic repairs, preempt-resume service, and explicit event-driven queue evolution.

The calibrated parameters are `arrival_rate`, `service_rate`, `failure_rate`, and `repair_rate`. Calibration uses six observed summaries: throughput rate, mean cycle time, utilization, availability, mean queue length, and breakdown rate.

## Methodological design

Modern UQ toolkits treat calibration, uncertainty propagation, and sensitivity as separate analysis layers. OpenTURNS exposes nonlinear least-squares calibration and bootstrap-based calibration uncertainty, while its sensitivity module exposes Saltelli/Jansen Sobol algorithms. This repository follows the same conceptual separation but implements the pipeline transparently with SciPy.

The calibration stack is:

```text
observed replication KPI matrix
            |
            v
weighted stochastic calibration objective
            |
            v
bounded differential evolution
            |
            v
robust bounded least-squares polish
            |
            v
calibrated parameter vector
        /            \
       /              \
bootstrap refits   fit diagnostics
       |
       +----------------------------+
       |                            |
       v                            v
parameter-only UQ            predictive UQ
(common random numbers)      (fresh randomness)
       |
       v
Sobol sensitivity analysis
```

Official references used to ground the implementation:

- OpenTURNS calibration: https://openturns.github.io/openturns/latest/user_manual/calibration.html
- OpenTURNS nonlinear least-squares calibration: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.NonLinearLeastSquaresCalibration.html
- OpenTURNS Sobol sensitivity: https://openturns.github.io/openturns/latest/theory/reliability_sensitivity/sensitivity_sobol.html
- SciPy `least_squares`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
- SciPy `differential_evolution`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
- SciPy Sobol QMC: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html

## Why global search precedes least squares

A stochastic discrete-event simulator is not generally smooth in its parameters. Under fixed random streams, event counts and event order can change discontinuously as rates change. During development, a pure local finite-difference least-squares fit stayed too close to its initialization on the synthetic recovery benchmark.

The final implementation therefore uses bounded derivative-free differential evolution first and robust `soft_l1` least squares second. The global stage is a practical guard against a poor local start; it is not claimed to prove a global optimum.

## Common random numbers

Every candidate parameter vector is evaluated using the same simulation seed set during calibration. This common-random-number design reduces irrelevant Monte Carlo variation between candidates without pretending simulator uncertainty has disappeared.

Observation seeds and calibration seeds are deliberately different in the synthetic validation study. The estimator therefore cannot recover the ground truth simply by replaying the exact random stream that generated the observations.

## Bootstrap uncertainty

Observed replication rows are resampled with replacement and the simulator is recalibrated for each bootstrap sample. Because the DES calibration surface is non-smooth, bootstrap refits receive a shorter global-search stage before local polishing.

The reported parameter intervals are empirical 2.5%, 50%, and 97.5% quantiles. They are percentile bootstrap intervals, not BCa intervals and not Bayesian posterior credible intervals.

## Epistemic vs aleatory propagation

Two propagation modes are intentionally separate:

- **Parameter-only propagation:** each bootstrap parameter vector is evaluated on the same common random seeds and replicate means are retained. This emphasizes calibration/parameter uncertainty.
- **Predictive propagation:** each parameter vector receives a fresh simulation seed. This combines parameter uncertainty with stochastic process variability.

## Sobol sensitivity

`sobol_jansen` estimates first-order and total-order Sobol indices using scrambled Sobol low-discrepancy designs and Jansen estimators. Inputs are assumed mutually independent and uniformly distributed over the supplied bounds.

For the stochastic queue study, every sensitivity parameter vector is evaluated using the same small seed set and the replication mean. The variance decomposition therefore focuses on uncertain parameters rather than being dominated by unrelated simulator noise.

The estimator itself is tested on the analytic additive model `f(x)=x1+2*x2`, where the known variance contributions are 0.2, 0.8, and 0 for an unused third input.

## Run

```bash
python -m pip install -e ".[dev]"
pytest
python scripts/run_study.py
```

A smaller deterministic end-to-end study runs on every push and pull request and uploads the resulting JSON as a GitHub Actions artifact.

## Validation gates

The repository checks:

- simulator reproducibility under a fixed seed;
- expected queueing directionality when service capacity changes;
- synthetic parameter recovery against a known ground truth;
- bootstrap and uncertainty-propagation bounds/shapes;
- Jansen Sobol estimates against a function with known analytic indices;
- Python 3.10, 3.11, and 3.12 in GitHub Actions;
- an end-to-end calibration, bootstrap, propagation, and sensitivity research smoke run.

## Limitations

- This is a synthetic benchmark, not a plant-calibrated MES/ERP/IoT digital twin.
- Exponential arrival, service, failure, and repair assumptions are controlled modeling choices, not universal manufacturing laws.
- Bootstrap coverage in one synthetic CI seed is an integration diagnostic, not proof of nominal repeated-sampling coverage.
- Sobol indices depend on the chosen parameter bounds/distributions.
- Common random numbers reduce comparison noise but do not remove aleatory uncertainty.
- The project is an independent research implementation; it does not reproduce OpenTURNS, DAKOTA, UQpy, or any single published package.

## Literature context

Useful methodological starting points include Kennedy & O'Hagan (2001) on computer-model calibration, Jansen (1999) on variance-based designs for model output, Saltelli et al. (2010) on variance-based sensitivity analysis, and Efron & Tibshirani (1993) on bootstrap methods.

See `RESEARCH_NOTES.md` for implementation boundaries and interpretation guidance.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
