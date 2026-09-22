#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fleet_sim {

struct Parameters {
    double p_healthy_to_degraded = 0.006;
    double p_degraded_to_critical = 0.018;
    double p_critical_to_failed = 0.070;
    double preventive_success = 0.92;
    double replacement_success = 0.98;
    double productivity_degraded = 0.90;
    double productivity_critical = 0.65;
    double preventive_cost = 900.0;
    double replacement_cost = 5000.0;
    double lost_capacity_cost = 2200.0;
    double crew_slot_cost = 140.0;
    double standby_spare_cost = 90.0;
};

struct BatchOutput {
    std::vector<double> total_cost;
    std::vector<double> lost_unit_days;
    std::vector<double> failures;
    std::vector<double> maintenance_events;
};

inline void validate_policy(int threshold_state, int crew_slots, int standby_spares) {
    if (threshold_state < 1 || threshold_state > 3) {
        throw std::invalid_argument("threshold_state must be 1, 2, or 3");
    }
    if (crew_slots < 1 || crew_slots > 16) {
        throw std::invalid_argument("crew_slots must be in [1,16]");
    }
    if (standby_spares < 0 || standby_spares > 16) {
        throw std::invalid_argument("standby_spares must be in [0,16]");
    }
}

inline BatchOutput simulate_batch(
    const double* degradation_u,
    const double* maintenance_u,
    std::size_t scenarios,
    std::size_t days,
    std::size_t assets,
    int threshold_state,
    int crew_slots,
    int standby_spares,
    const Parameters& p = Parameters{}) {

    validate_policy(threshold_state, crew_slots, standby_spares);
    if (!degradation_u || !maintenance_u) {
        throw std::invalid_argument("random draw pointers must not be null");
    }
    if (scenarios == 0 || days == 0 || assets == 0) {
        throw std::invalid_argument("scenario dimensions must be positive");
    }

    BatchOutput out;
    out.total_cost.assign(scenarios, 0.0);
    out.lost_unit_days.assign(scenarios, 0.0);
    out.failures.assign(scenarios, 0.0);
    out.maintenance_events.assign(scenarios, 0.0);

    std::vector<std::uint8_t> state(assets, 0);
    std::vector<std::uint8_t> maintained(assets, 0);
    std::vector<std::size_t> candidates;
    candidates.reserve(assets);

    for (std::size_t s = 0; s < scenarios; ++s) {
        std::fill(state.begin(), state.end(), static_cast<std::uint8_t>(0));
        double scenario_cost = 0.0;
        double lost_days = 0.0;
        double failures = 0.0;
        double maintenance_events = 0.0;

        for (std::size_t d = 0; d < days; ++d) {
            std::fill(maintained.begin(), maintained.end(), static_cast<std::uint8_t>(0));
            candidates.clear();

            // Highest-severity assets are maintained first; asset id is a
            // deterministic tie breaker. Failed assets always qualify.
            for (std::size_t a = 0; a < assets; ++a) {
                if (state[a] >= static_cast<std::uint8_t>(threshold_state)) {
                    candidates.push_back(a);
                }
            }
            std::stable_sort(candidates.begin(), candidates.end(), [&](std::size_t lhs, std::size_t rhs) {
                if (state[lhs] != state[rhs]) return state[lhs] > state[rhs];
                return lhs < rhs;
            });

            const std::size_t selected = std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(crew_slots));
            for (std::size_t k = 0; k < selected; ++k) {
                const std::size_t a = candidates[k];
                maintained[a] = 1;
                ++maintenance_events;

                const std::size_t idx = (s * days + d) * assets + a;
                const double u = maintenance_u[idx];
                if (state[a] == 3) {
                    scenario_cost += p.replacement_cost;
                    state[a] = (u < p.replacement_success) ? 0 : 1;
                } else {
                    scenario_cost += p.preventive_cost;
                    if (u < p.preventive_success) {
                        state[a] = 0;
                    } else {
                        state[a] = static_cast<std::uint8_t>(state[a] > 0 ? state[a] - 1 : 0);
                    }
                }
            }

            // Daily fixed capacity charges are paid regardless of utilization.
            scenario_cost += static_cast<double>(crew_slots) * p.crew_slot_cost;
            scenario_cost += static_cast<double>(standby_spares) * p.standby_spare_cost;

            double capacity_deficit = 0.0;
            for (std::size_t a = 0; a < assets; ++a) {
                if (maintained[a]) {
                    capacity_deficit += 1.0;
                    continue;
                }
                switch (state[a]) {
                    case 0: break;
                    case 1: capacity_deficit += 1.0 - p.productivity_degraded; break;
                    case 2: capacity_deficit += 1.0 - p.productivity_critical; break;
                    default: capacity_deficit += 1.0; break;
                }
            }

            const double covered = std::min<double>(static_cast<double>(standby_spares), capacity_deficit);
            const double lost = capacity_deficit - covered;
            lost_days += lost;
            scenario_cost += lost * p.lost_capacity_cost;

            // End-of-day deterioration for assets not maintained today.
            for (std::size_t a = 0; a < assets; ++a) {
                if (maintained[a] || state[a] == 3) continue;
                const std::size_t idx = (s * days + d) * assets + a;
                const double u = degradation_u[idx];
                if (state[a] == 0 && u < p.p_healthy_to_degraded) {
                    state[a] = 1;
                } else if (state[a] == 1 && u < p.p_degraded_to_critical) {
                    state[a] = 2;
                } else if (state[a] == 2 && u < p.p_critical_to_failed) {
                    state[a] = 3;
                    ++failures;
                }
            }
        }

        out.total_cost[s] = scenario_cost;
        out.lost_unit_days[s] = lost_days;
        out.failures[s] = failures;
        out.maintenance_events[s] = maintenance_events;
    }

    return out;
}

}  // namespace fleet_sim
