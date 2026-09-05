import numpy as np

from simcal_uq import QueueParameters, calibrate, generate_observations


def test_synthetic_calibration_recovers_truth_reasonably():
    truth = QueueParameters(0.82, 1.28, 0.055, 0.48)
    obs = generate_observations(truth, range(30, 46), horizon=350.0)
    fit = calibrate(obs, simulation_seeds=range(200, 208), horizon=350.0, max_nfev=70)
    relative_error = np.abs(fit.parameters.as_array() - truth.as_array()) / truth.as_array()
    assert fit.success
    assert relative_error.max() < 0.40
    assert np.mean(np.abs(fit.residuals)) < 3.0
