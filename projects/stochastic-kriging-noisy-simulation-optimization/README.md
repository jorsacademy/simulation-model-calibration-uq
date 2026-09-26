# Stochastic Kriging and Noisy Simulation Optimization

Native benchmark extension for the
`simulation-optimization-and-uncertainty-quantification` umbrella repository.

The implementation lives in the root package:

- `src/simcal_uq/metamodel.py` — replicated-design summaries, heteroskedastic
  stochastic kriging, and expected improvement for minimization;
- `src/simcal_uq/noisy_optimization.py` — queue-design simulation objective,
  equal allocation, OCBA, stochastic-kriging sequential search, reference
  evaluation, and simple regret;
- `scripts/run_noisy_optimization.py` — matched-budget benchmark runner.

## Research question

Given a fixed number of expensive stochastic simulation replications, how should
that budget be allocated across candidate operating designs?

The benchmark compares three policies:

1. **Equal allocation** — spread the budget uniformly over every candidate.
2. **OCBA** — allocate more replications to statistically competitive
   alternatives using approximate optimal computing budget allocation ratios.
3. **Stochastic kriging + expected improvement** — fit a heteroskedastic
   Gaussian-process metamodel to replicated sample means and sequentially select
   the next design to simulate.

## Decision problem

The underlying simulator is the repository's repairable production queue.

Two simulator inputs are reinterpreted as controllable capacity decisions:

- service rate;
- repair rate.

One noisy simulation replication returns

`capacity cost + repair-capacity cost + queue cost + cycle-time cost`.

This is a synthetic methodology benchmark. The cost coefficients are controlled
experimental choices, not factory economics.

## Stochastic-kriging model

At each sampled design point the code stores repeated simulation outputs and
estimates:

- the sample mean;
- the sample variance;
- the variance of the sample mean.

The Gaussian-process covariance matrix therefore uses a different observation
noise term for each design point. A constant trend, RBF covariance
hyperparameters, and process variance are estimated by Gaussian marginal
likelihood.

The posterior variance is the latent-response uncertainty. It is not a future
single-replication predictive variance.

## OCBA

The finite-alternative OCBA implementation uses the classical allocation
structure:

- non-best alternatives receive weights proportional to
  `variance / squared estimated optimality gap`;
- the estimated best alternative receives the corresponding balancing weight;
- integer replication targets are obtained with a largest-remainder correction.

This is an approximate sequential OCBA benchmark, not a proof of asymptotic
optimal allocation for every finite sample.

## Matched-budget evaluation

Every search method receives the same simulation-replication budget.

A larger, evaluation-only Monte Carlo sample estimates reference mean costs after
all methods have selected their designs. That reference budget is not available
to the search policies.

Primary decision metric:

`simple regret = selected reference mean - best reference mean`.

The benchmark does not assume that stochastic kriging or OCBA must beat equal
allocation on every seed.

## Run

From the umbrella repository root:

~~~bash
python -m pip install -e ".[dev]"
python scripts/run_noisy_optimization.py \
  --grid-size 5 \
  --budget 150 \
  --reference-reps 60 \
  --horizon 250
~~~

Output:

`artifacts/noisy_optimization.json`

## Scope boundary

This first extension deliberately uses:

- a finite candidate grid;
- independent simulation streams across alternatives;
- a diagonal replication-noise model;
- single-fidelity simulation;
- expected improvement as the sequential acquisition function.

Natural next extensions are correlated common-random-number stochastic kriging,
multi-fidelity simulation, constrained noisy optimization, knowledge gradient,
and larger ranking-and-selection experiments.
