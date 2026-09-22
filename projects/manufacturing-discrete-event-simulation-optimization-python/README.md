# Manufacturing Discrete-Event Simulation and Design Optimization

Event-driven simulation and simulation optimization for a stylized manufacturing line with finite buffers, stochastic processing times, machine failures, shared repair technicians, quality inspection, and rework.

The project is designed as an operations-research / industrial-engineering example rather than a queueing toy.

## Production system

The simulated line is:

```text
Arrivals
   |
   v
Machining
   |
 finite buffer B1
   |
   v
Assembly
   |
 finite buffer B2
   |
   v
Test -------- Pass ------> Finished goods
   |
  Fail
   v
Rework ------ Recover ----> Finished goods
   |
  Scrap
```

The model contains four machines:

- Machining
- Assembly
- Test
- Rework

## Discrete-event mechanics

The simulator implements its own event calendar using `heapq`.

Events include:

- external job arrival;
- processing completion;
- machine failure;
- repair completion.

Machines fail according to exponentially distributed **operating-time** intervals. A failure interrupts the current job, which resumes after repair.

All machines share a configurable pool of repair technicians. If every technician is occupied, failed machines wait in an FCFS maintenance queue.

The two main intermediate buffers are finite. When a downstream buffer is full, the upstream machine remains blocked while holding its completed job.

## Common random numbers

A simulation replication is generated as a complete primitive-randomness scenario containing:

- arrival times;
- job/station processing times;
- inspection/rework outcomes;
- machine failure intervals;
- repair durations.

Every candidate factory design in the same replication receives exactly the same scenario. This implements **common random numbers (CRN)** and enables paired statistical comparisons between designs.

## Design decisions

The optimization searches:

```text
B1 capacity in {2, 4, 6, 8, 10}
B2 capacity in {2, 4, 6, 8, 10}
repair technicians in {1, 2}
```

There are exactly 50 candidate designs.

The sample-average optimization enumerates all 50 designs, so the selected design is the exact maximizer of the estimated objective over this declared finite design set.

It is **not** a global optimum over every possible factory configuration.

## Economic objective

The simulation reports an hourly operating contribution:

```text
good-unit contribution
- scrap cost
- WIP holding cost
- repair-technician cost
- buffer-capacity charge
```

The coefficients are internally consistent educational assumptions, not calibrated accounting data for a particular plant.

## KPIs

Each replication reports:

- profit per hour;
- good-unit throughput;
- scrap rate;
- mean cycle time;
- time-average WIP;
- machine utilization;
- machine downtime fraction;
- machine blocking fraction.

The default model uses an arrival rate below the effective bottleneck capacity, avoiding an intentionally unstable queueing regime.

## Warm-up and measurement

The default run uses:

```text
8 h warm-up
40 h measurement window
```

Throughput and scrap are measured from exits during the measurement window.

Cycle-time statistics include jobs that **arrive after warm-up**, avoiding contamination from the initial empty-system transient.

## Selection and validation

The optimization uses two independent scenario sets:

1. **selection scenarios** — shared by all 50 candidate designs;
2. **validation scenarios** — generated after design selection and never used to choose the design.

The selected design and a predefined baseline are compared on the same validation scenarios.

A paired Student-t confidence interval is reported for:

```text
profit(selected design) - profit(baseline)
```

This paired CRN analysis is more informative than comparing two unrelated confidence intervals.

## Reference run

With the default seed, 12 selection replications and 30 independent validation replications, the validated development run produced:

```text
Selected design
  B1 = 4
  B2 = 2
  repair technicians = 1

Independent validation
  selected profit  ~1227.17 $/h
  baseline profit  ~1215.24 $/h

Paired improvement
  +11.93 $/h
  95% CI [8.97, 14.88]

Selected throughput
  ~6.448 good units/h

Selected mean cycle time
  ~2.115 h

Selected mean WIP
  ~14.194 jobs
```

The selected design does not maximize throughput. It trades a small throughput difference against the cost of unnecessary buffer capacity.

These values are reproducible for the declared model and seed. They are not general manufacturing-performance claims.

## Hand-checkable DES oracles

The regression suite includes two deterministic event schedules.

### Single-job oracle

One job requires exactly one hour at Machining, Assembly, and Test and never fails or requires rework.

Expected results over a 4-hour measurement window:

```text
completion time = 3 h
cycle time = 3 h
throughput = 0.25 jobs/h
mean WIP = 0.75 jobs
utilization of first three machines = 0.25 each
```

### Finite-buffer blocking oracle

Three jobs with a one-slot Machining→Assembly buffer create an exactly known blocking interval.

Machining is blocked from `t=1.5` to `t=2.5`, therefore its blocked fraction over five hours is exactly:

```text
1 / 5 = 0.20
```

This test also guards against recursive queue/unblocking bugs that can silently lose entities in DES implementations.

## Job-conservation invariant

At every completed simulation run the code checks that every arrived job is exactly one of:

- active in one queue/machine; or
- completed good; or
- scrapped.

A job cannot appear in multiple active locations or both exit classes. The tracked WIP count must equal the number of active jobs.

## Run

```bash
python manufacturing_des_optimization.py
```

Fast checks:

```bash
python manufacturing_des_optimization.py --self-test
python -m unittest discover -s tests -v
```

A smaller optimization run:

```bash
python manufacturing_des_optimization.py \
  --selection-replications 6 \
  --validation-replications 10 \
  --seed 42
```

## Scope and limitations

This is a stylized manufacturing simulation, not a calibrated digital twin.

It assumes:

- stationary arrival and processing distributions;
- operating-time exponential failure clocks;
- lognormal repair durations;
- one product family;
- no setup/changeover sequence dependence;
- no shift calendar;
- no material shortages;
- fixed inspection and rework probabilities.

For real decision support, distribution parameters and economic coefficients would need to be estimated and validated from plant data.
