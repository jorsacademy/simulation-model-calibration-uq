import numpy as np

from simcal_uq import (
    QueueParameters,
    bootstrap_calibration,
    calibrate,
    generate_observations,
    propagate_parameter_uncertainty,
    propagate_predictive_uncertainty,
)


def test_bootstrap_and_propagation_shapes_and_bounds():
    truth = QueueParameters(0.82, 1.28, 0.055, 0.48)
    obs = generate_observations(truth, range(10, 20), horizon=220.0)
    seeds = range(300, 304)
    fit = calibrate(obs, simulation_seeds=seeds, horizon=220.0, max_nfev=45)
    boot = bootstrap_calibration(
        obs,
        fit,
        simulation_seeds=seeds,
        n_bootstrap=5,
        horizon=220.0,
        seed=9,
        max_nfev=20,
    )
    assert boot.parameter_samples.shape == (5, 4)
    assert np.all(boot.lower <= boot.median)
    assert np.all(boot.median <= boot.upper)
    assert 0.0 <= boot.success_rate <= 1.0

    param_prop = propagate_parameter_uncertainty(
        boot.parameter_samples, evaluation_seeds=(11, 12), horizon=180.0
    )
    pred_prop = propagate_predictive_uncertainty(boot.parameter_samples, horizon=180.0, seed=11)
    assert param_prop.metric_samples.shape == (5, 6)
    assert pred_prop.metric_samples.shape == (5, 6)
    assert np.all(np.isfinite(param_prop.mean))
    assert np.all(np.isfinite(pred_prop.mean))
    assert np.all(param_prop.lower <= param_prop.upper)
