#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "simulation_core.hpp"

namespace py = pybind11;

py::tuple simulate_batch_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> degradation_u,
    py::array_t<double, py::array::c_style | py::array::forcecast> maintenance_u,
    int threshold_state,
    int crew_slots,
    int standby_spares) {

    auto deg = degradation_u.request();
    auto maint = maintenance_u.request();
    if (deg.ndim != 3 || maint.ndim != 3) {
        throw py::value_error("random arrays must have shape [scenario, day, asset]");
    }
    if (deg.shape != maint.shape) {
        throw py::value_error("degradation_u and maintenance_u shapes must match");
    }

    const auto scenarios = static_cast<std::size_t>(deg.shape[0]);
    const auto days = static_cast<std::size_t>(deg.shape[1]);
    const auto assets = static_cast<std::size_t>(deg.shape[2]);

    py::array_t<double> total_cost(scenarios);
    py::array_t<double> lost_unit_days(scenarios);
    py::array_t<double> failures(scenarios);
    py::array_t<double> maintenance_events(scenarios);

    auto cost_buf = total_cost.request();
    auto lost_buf = lost_unit_days.request();
    auto fail_buf = failures.request();
    auto maint_buf = maintenance_events.request();

    fleet_sim::BatchOutput output;
    {
        py::gil_scoped_release release;
        output = fleet_sim::simulate_batch(
            static_cast<const double*>(deg.ptr),
            static_cast<const double*>(maint.ptr),
            scenarios,
            days,
            assets,
            threshold_state,
            crew_slots,
            standby_spares);
    }

    std::copy(output.total_cost.begin(), output.total_cost.end(), static_cast<double*>(cost_buf.ptr));
    std::copy(output.lost_unit_days.begin(), output.lost_unit_days.end(), static_cast<double*>(lost_buf.ptr));
    std::copy(output.failures.begin(), output.failures.end(), static_cast<double*>(fail_buf.ptr));
    std::copy(output.maintenance_events.begin(), output.maintenance_events.end(), static_cast<double*>(maint_buf.ptr));

    return py::make_tuple(total_cost, lost_unit_days, failures, maintenance_events);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++17 accelerated industrial fleet Monte Carlo kernel";
    m.def(
        "simulate_batch",
        &simulate_batch_py,
        py::arg("degradation_u"),
        py::arg("maintenance_u"),
        py::arg("threshold_state"),
        py::arg("crew_slots"),
        py::arg("standby_spares"));
}
