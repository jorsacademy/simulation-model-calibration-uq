# Simulation, Digital-Twin, and Black-Box Optimization Research Series

This repository belongs to a broader set of independent projects on simulation optimization, expensive black-box optimization, model calibration, and digital-twin decision systems.

## Simulation and digital-twin foundations

| Repository | Main focus | Role in the series |
|---|---|---|
| `manufacturing-discrete-event-simulation-optimization-python` | Discrete-event simulation linked to optimization | Simulation-optimization foundation |
| `airport-checkin-simulation-optimization` | Simulation-based service-system optimization | Queueing/service application |
| `agent-based-supply-chain-simulation-python` | Agent-based simulation of supply-chain dynamics | Agent-based simulation |
| `simulation-model-calibration-uq` | Calibration and uncertainty quantification for simulation models | Model credibility / UQ |
| `industrial-digital-twin-system-optimization-python` | Digital-twin architecture connected to optimization | Digital-twin foundation |
| `dynamic-manufacturing-digital-twin-rl` | Reinforcement learning inside a dynamic manufacturing twin | Closed-loop learning/control |
| `cpp-accelerated-optimization-simulation-python` | Accelerating simulation/optimization workloads with C++ integration | Computational acceleration |
| `parallel-monte-carlo-stochastic-optimization-python` | Parallel stochastic evaluation and Monte Carlo optimization | Parallel stochastic simulation |

## Bayesian and sequential model-based optimization

The following educational repositories are intentionally kept as separate language-specific learning resources:

- `bayesian-optimization-endustri-muhendisligi` — Turkish educational guide.
- `bayesian-optimization-industrial-engineering` — English educational guide.

Additional method/tool projects:

| Repository | Main focus |
|---|---|
| `constrained-bayesian-optimization-chemical-process-python` | Bayesian optimization with explicit constraints |
| `botorch-bayesian-policy-search` | BoTorch-based policy/decision search |
| `sambo-sequential-model-based-optimization` | Sequential model-based black-box optimization |
| `smac3-simulation-based-optimization` | SMAC3 for simulation-based optimization |
| `processoptimizer-industrial-process-optimization` | ProcessOptimizer for industrial parameter search |
| `hyperopt-inventory-policy-optimization` | Hyperopt for inventory-policy tuning |
| `nevergrad-black-box-policy-optimization` | Gradient-free black-box policy optimization |
| `ray-tune-distributed-policy-search` | Distributed parameter/policy search |

## Why the tool-oriented repositories remain separate

SMAC3, SAMBO, BoTorch, Hyperopt, Nevergrad, Ray Tune, and ProcessOptimizer are not treated as interchangeable wrappers. Keeping the projects separate makes their search spaces, surrogate assumptions, parallelism, acquisition/search strategies, and integration patterns easier to compare directly.

They can later be benchmarked under one common experimental protocol without merging the repositories.

## Suggested conceptual progression

1. `manufacturing-discrete-event-simulation-optimization-python`
2. `simulation-model-calibration-uq`
3. `bayesian-optimization-industrial-engineering` or `bayesian-optimization-endustri-muhendisligi`
4. `constrained-bayesian-optimization-chemical-process-python`
5. tool-specific optimization projects (`SMAC3`, `SAMBO`, `BoTorch`, `Hyperopt`, `Nevergrad`)
6. `industrial-digital-twin-system-optimization-python`
7. `dynamic-manufacturing-digital-twin-rl`

The ordering is pedagogical rather than a ranking of methods.