# Research Notes

## 1. Scientific question

The repository asks a narrow question: given noisy repeated observations from a stochastic operational system, can a simulator be calibrated in a reproducible way while keeping **fit**, **parameter uncertainty**, **predictive uncertainty**, and **sensitivity** conceptually separate?

That separation is the primary methodological point of the project.

## 2. Simulator

The benchmark is a single repairable production resource. Jobs arrive according to a Poisson process and require exponentially distributed processing time. The resource can fail while it is up. If a failure interrupts a job, service is paused and resumes after an exponentially distributed repair time.

The event engine explicitly tracks arrivals, service completions, failures, repairs, queue-area integrals, busy-up time, up-time, completed jobs, and cycle times.

This is intentionally simpler than a plant-scale digital twin. The purpose is to make the calibration and UQ mechanics auditable.

## 3. Calibration objective

Observed data are represented as a matrix of repeated KPI measurements. The calibration target is the mean KPI vector. Residuals are standardized using estimated standard errors with metric-specific numerical floors to prevent nearly-zero observed variance from dominating the fit.

The objective therefore measures discrepancies in approximately comparable uncertainty units rather than mixing raw hours, rates, and queue lengths directly.

## 4. Why differential evolution + least squares

A discrete-event simulator can create a non-smooth calibration surface. Event counts, queue transitions, and failure ordering can change as rates change. Pure local finite-difference least squares was tested during development and was observed to stay too close to the midpoint initialization in a deterministic synthetic recovery experiment.

The final design therefore uses:

1. bounded differential evolution for global derivative-free exploration;
2. robust `soft_l1` bounded least-squares polishing.

The global stage is not claimed to prove a global optimum. It is a practical robustness measure against poor local initialization on a rough simulation response surface.

## 5. Common random numbers

Calibration uses one fixed seed set for every candidate parameter vector. This is a standard variance-reduction idea for comparing stochastic alternatives: the goal is to make candidate differences reflect parameter changes rather than unrelated random draws.

The observation seed set is separate from the calibration seed set in the synthetic study. This prevents recovery from becoming trivial by fitting the exact same random stream that generated the observations.

## 6. Bootstrap

The bootstrap is performed over complete observed replication rows. Each bootstrap sample is recalibrated. Because the simulator calibration surface is non-smooth, bootstrap refits use a shorter global differential-evolution stage before local polishing instead of relying only on the original optimum as a local start.

Reported intervals are empirical 2.5%, 50%, and 97.5% quantiles. They are percentile intervals, not BCa intervals and not Bayesian credible intervals.

## 7. Epistemic vs aleatory propagation

Two propagation modes are intentionally implemented.

### Parameter-only propagation

Each bootstrap parameter vector is evaluated on the same common random seeds and the replicate mean is retained. This suppresses most seed-to-seed process noise so the distribution mainly reflects calibration/parameter uncertainty.

### Predictive propagation

Each bootstrap parameter vector is evaluated with a fresh random seed. The resulting distribution combines parameter uncertainty and stochastic operational variability. This is closer to a predictive distribution for a future realization, but it is wider and answers a different question.

## 8. Sobol indices

The repository implements Jansen estimators for first-order and total-order Sobol indices. Scrambled Sobol sequences are generated with SciPy QMC. The base sample size must be a power of two to preserve the balance properties recommended by SciPy's Sobol implementation.

The implementation is validated against the analytic additive function

`f(x) = x1 + 2*x2`, with independent U(0,1) inputs,

for which the variance contributions are proportional to 1²:2² = 1:4. Therefore the first-order and total-order indices are 0.2 and 0.8, while the unused third input has index 0.

## 9. Validation interpretation

Synthetic recovery is useful because the true parameter vector is known. A small recovery error demonstrates that the pipeline can identify this benchmark under the chosen observation design; it does not prove identifiability for arbitrary simulators or real plant data.

Bootstrap truth coverage in one CI seed is included as an integration diagnostic, not as an estimate of repeated-sampling coverage probability. Proper coverage assessment would require many independent synthetic datasets and substantially more computation.


## 10. Stochastic kriging and noisy optimization

The noisy-optimization extension asks a different question from calibration: if the simulator itself is the objective oracle and every evaluation is stochastic, where should a fixed replication budget be spent?

The controlled decision problem exposes two queue-system design variables:

- service capacity through the service-rate parameter;
- repair capacity through the repair-rate parameter.

Each replication combines capacity cost, repair-capacity cost, mean queue length, and mean cycle time into one synthetic operating-cost response. These coefficients are benchmark choices, not industrial cost estimates.

### Replicated-design noise model

For each sampled design point the implementation retains the full replication vector. The metamodel is fitted to the replication mean, while the diagonal observation-noise term is

`sample variance / number of replications`.

The noise level can therefore differ across design points. This is the principal distinction from fitting an ordinary deterministic GP with one global nugget.

### Stochastic-kriging metamodel

The implementation uses ordinary kriging with:

- a constant generalized-least-squares trend;
- anisotropic RBF length scales;
- process variance estimated by Gaussian marginal likelihood;
- known heteroskedastic replication-noise estimates;
- a small numerical jitter for Cholesky stability.

Posterior uncertainty is uncertainty about the latent mean response, not the variance of one future raw simulation replication.

### Expected improvement

The sequential surrogate policy minimizes the simulated cost. At each stage it computes expected improvement relative to the best currently observed replication mean and spends the next small replication batch at the candidate with maximum acquisition value.

Repeated sampling of an already-observed candidate is allowed. Under simulation noise that can be rational because additional replications reduce uncertainty in the sample mean.

### OCBA

The finite-alternative ranking-and-selection policy uses the classical approximate OCBA allocation ratios. For non-best alternatives the weight is proportional to sample variance divided by squared estimated optimality gap. The estimated best receives the balancing allocation implied by the competitor weights.

The implementation converts continuous ratios to integer target counts with a minimum-replication floor and largest-remainder correction. It should be interpreted as a transparent sequential OCBA benchmark rather than a complete reproduction of every finite-sample OCBA variant.

### Evaluation contract

Equal allocation, OCBA, and stochastic-kriging expected improvement receive the same search-stage simulation budget.

Reference mean costs are estimated only after every method has selected a candidate. That larger Monte Carlo budget is evaluation-only information. The primary decision metric is simple regret relative to the best reference mean.

No test asserts that a sophisticated method must dominate equal allocation. The regression suite checks mathematical and computational invariants, not a preferred research conclusion.


## 11. Extensions

Natural research extensions include:

- real manufacturing or call-center event logs;
- censored/incomplete observations;
- non-exponential service and repair distributions;
- likelihood-free Bayesian calibration / ABC;
- Gaussian-process emulator-assisted calibration with explicit model discrepancy;
- explicit model discrepancy terms;
- profile likelihood confidence regions;
- Morris screening before Sobol analysis;
- correlated uncertain inputs and Shapley effects;
- multi-fidelity simulation calibration and optimization;
- correlated common-random-number stochastic kriging;
- knowledge-gradient and constrained noisy acquisition policies;
- sequential experimental design to reduce parameter uncertainty.
