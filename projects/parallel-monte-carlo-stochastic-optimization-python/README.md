# Parallel Monte Carlo Stochastic Optimization in Python

A stochastic industrial energy-reserve planning project that separates mathematical correctness from computational performance. The same Monte Carlo recourse model is evaluated with a scalar Python reference kernel, vectorized NumPy, and shared-memory multiprocessing.

## Problem

An industrial facility must choose day-ahead capacity before uncertain operations are realized:

```text
first-stage decisions
  grid contract [MW]
  backup reserve [MW]

uncertainty
  industrial demand
  solar generation
  grid derating / outage
  backup availability
  electricity price
```

The realized operating cost includes contracted capacity, grid energy, backup energy, and a large penalty for unserved energy.

The objective combines expected cost and tail risk:

```text
(1 - lambda) * E[cost] + lambda * CVaR95(cost)
```

with default `lambda = 0.35`.

## Correlated stochastic scenarios

The scenario generator includes:

- daily and hourly load shocks;
- cloud-driven solar uncertainty;
- persistent grid derating represented by a two-state Markov process;
- backup-system availability and derating;
- price shocks positively coupled to daily load conditions.

All candidate plans use the same generated scenario set. This common-random-number design avoids comparing plans on unrelated noise realizations.

## Finite first-stage optimization

The declared design grid is:

```text
grid contract  : 6.0 .. 13.0 MW, step 0.5
backup reserve : 0.0 .. 10.0 MW, step 0.5
```

There are exactly:

```text
15 * 21 = 315 candidate plans
```

Every candidate is evaluated, so the selected plan is the exact minimizer of the sample objective over this finite grid.

This is not a proof of the continuous-capacity optimum.

## Independent validation

Selection and validation scenario sets use different seeds. The selected stochastic plan is compared on the independent validation set with an expected-value baseline that optimizes the same finite decision grid after replacing random inputs with their period-wise means.

A representative local development run with 3,000 selection and 8,000 validation scenarios produced:

```text
risk-aware selected plan
  grid contract             11.00 MW
  backup reserve             7.50 MW
  validation mean cost      $26,519.52/day
  validation CVaR95         $32,673.96/day
  P(any shortfall)            4.537%
  expected unserved energy    0.07084 MWh/day

expected-value baseline
  grid contract             11.00 MW
  backup reserve             0.00 MW
  validation mean cost      $26,913.22/day
  validation CVaR95         $55,210.17/day
  P(any shortfall)           38.025%
  expected unserved energy    1.85022 MWh/day
```

These are results for the declared synthetic stochastic model and seed, not real plant-cost or reliability claims.

## Three Monte Carlo kernels

### Scalar Python reference

A direct scenario/period loop implements the recourse equations in the clearest possible form. It is intentionally retained as a correctness oracle.

### Vectorized NumPy

The same equations are evaluated across all scenarios with array operations. No mathematical approximation is changed.

### Shared-memory multiprocessing

The scalar reference workload is partitioned among processes. Large scenario tensors are copied into `multiprocessing.shared_memory` once, and workers receive metadata plus row ranges rather than separate serialized copies of every chunk.

Multiprocessing timing therefore includes:

- shared-memory allocation/copy;
- process startup;
- worker execution;
- result collection;
- cleanup.

This is deliberately an end-to-end benchmark.

## Numerical equivalence

For one fixed scenario set and plan, all implementations must agree on:

```text
scenario cost vector
scenario unserved-energy vector
```

The regression suite uses the scalar implementation as the numerical oracle and checks NumPy and multiprocessing results to tight tolerances.

## Hand-checkable recourse oracle

A one-scenario, two-period test has an analytically known result:

```text
fixed capacity cost = 70
period 1 cost       = 500
period 2 cost       = 4200
total cost          = 4770
unserved energy     = 3 MWh
```

Both scalar and vectorized kernels must reproduce this exactly.

## Performance interpretation

This project does not assume that adding processes always makes a workload faster.

Array-friendly numerical work often benefits much more from NumPy vectorization than from Python multiprocessing. For small workloads, process and shared-memory overhead can make multiprocessing slower than one scalar process. For larger scalar workloads, multiple workers may amortize that overhead.

For that reason GitHub Actions records actual wall-clock timings but does **not** fail the build if a particular speedup is not achieved.

Only numerical correctness is a CI requirement.

## Run

Stochastic optimization:

```bash
python parallel_monte_carlo_stochastic_optimization.py
```

Offline self-test:

```bash
python parallel_monte_carlo_stochastic_optimization.py --self-test
```

Regression suite:

```bash
python -m unittest discover -s tests -v
```

Benchmark:

```bash
python parallel_monte_carlo_stochastic_optimization.py \
  --benchmark \
  --benchmark-scenarios 10000 \
  --benchmark-repeats 1 \
  --seed 123
```

## Validated GitHub Actions run

The complete workflow was executed on GitHub Actions with CPython 3.12.14. The self-test, all seven regression tests, the 3,000-selection / 8,000-validation stochastic optimization, and the 10,000-scenario kernel benchmark completed successfully.

GitHub-runner optimization result:

```text
risk-aware selected plan
  grid contract             11.00 MW
  backup reserve             7.50 MW
  validation mean cost      $26,519.52/day
  validation CVaR95         $32,673.96/day
  P(any shortfall)            4.537%
  expected unserved energy    0.07084 MWh/day

expected-value baseline
  grid contract             11.00 MW
  backup reserve             0.00 MW
  validation mean cost      $26,913.22/day
  validation CVaR95         $55,210.17/day
  P(any shortfall)           38.025%
  expected unserved energy    1.85022 MWh/day
```

GitHub-runner kernel benchmark:

```text
scalar Python            1 worker   0.1218 s    1.00x
vectorized NumPy         1 worker   0.0012 s  102.21x
shared multiprocessing   1 worker   0.1227 s    0.99x
shared multiprocessing   2 workers  0.0880 s    1.38x
shared multiprocessing   4 workers  0.0797 s    1.53x
```

All kernels produced the same reported mean scenario cost (`$27,446.093`) and passed vector-level numerical-equivalence checks.

These timings are measurements from one hosted GitHub runner. They are not portable speedup guarantees. In particular, the NumPy result shows that vectorization is much more effective for this array-friendly kernel than adding Python processes, while multiprocessing still showed modest scaling from two to four workers on this runner.

## Exactness and modeling scope

The exhaustive grid search is exact for the declared 315-plan sample-average/risk problem.

The Monte Carlo scenarios themselves are synthetic. Their distributions, outage dynamics, tariffs, penalties and equipment availability are modeling assumptions rather than estimates from a particular plant.

A production implementation would need calibrated forecasts and scenario models, dependence validation, nonstationary outage behavior, contractual tariff rules, operational constraints, and out-of-sample model monitoring.
