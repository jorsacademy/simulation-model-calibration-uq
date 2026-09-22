#include <cmath>
#include <iostream>
#include <vector>
#include "../src/cpp_accelerated_sim/simulation_core.hpp"

int main() {
    // One scenario, three days, one asset. Uniform zero forces deterioration
    // H->D on day 1 and D->C on day 2. Threshold=2 triggers preventive
    // maintenance on day 3.
    const std::size_t S = 1, D = 3, A = 1;
    std::vector<double> degradation(S * D * A, 0.0);
    std::vector<double> maintenance(S * D * A, 0.0);
    auto out = fleet_sim::simulate_batch(
        degradation.data(), maintenance.data(), S, D, A,
        2, 1, 0);

    // Day 1: 140
    // Day 2: 140 + 0.1*2200 = 360
    // Day 3: 140 + 900 + 1.0*2200 = 3240
    // Total = 3740.
    if (std::abs(out.total_cost[0] - 3740.0) > 1e-9) {
        std::cerr << "unexpected total cost: " << out.total_cost[0] << "\n";
        return 1;
    }
    if (std::abs(out.maintenance_events[0] - 1.0) > 1e-9) return 2;
    std::cout << "C++ core smoke oracle: OK\n";
    return 0;
}
