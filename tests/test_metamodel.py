import numpy as np

from simcal_uq.metamodel import (
    StochasticKriging,
    expected_improvement_minimize,
    summarize_replications,
)


def test_replication_summary_uses_variance_of_the_mean():
    x = np.array([[0.0], [0.5], [1.0]])
    values = [
        np.array([1.0, 1.2, 0.8, 1.1]),
        np.array([0.5, 0.6, 0.4, 0.55]),
        np.array([1.4, 1.5, 1.3, 1.45]),
    ]
    design = summarize_replications(x, values)
    assert np.all(design.replications == 4)
    assert np.allclose(
        design.mean_variance,
        design.sample_variance / design.replications,
    )


def test_stochastic_kriging_tracks_smooth_noisy_response():
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 1.0, 7).reshape(-1, 1)
    values = []
    for point in x[:, 0]:
        truth = (point - 0.35) ** 2
        values.append(truth + rng.normal(0.0, 0.03, size=8))

    model = StochasticKriging().fit(summarize_replications(x, values))
    prediction, std = model.predict(
        np.array([[0.35], [0.9]]),
        return_std=True,
    )
    assert prediction[0] < prediction[1]
    assert np.all(std >= 0.0)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(std))


def test_expected_improvement_for_minimization_is_nonnegative():
    mean = np.array([0.8, 1.0, 1.2])
    std = np.array([0.1, 0.2, 0.0])
    ei = expected_improvement_minimize(mean, std, incumbent=0.9)
    assert np.all(ei >= 0.0)
    assert ei[0] > ei[2]
