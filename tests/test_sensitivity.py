import numpy as np

from simcal_uq import sobol_jansen


def test_jansen_sobol_matches_additive_function():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    def func(x):
        return float(x[0] + 2.0 * x[1])

    result = sobol_jansen(func, bounds, n_base=1024, seed=3)
    assert abs(result.first_order[0] - 0.2) < 0.07
    assert abs(result.first_order[1] - 0.8) < 0.07
    assert abs(result.total_order[0] - 0.2) < 0.07
    assert abs(result.total_order[1] - 0.8) < 0.07
    assert abs(result.total_order[2]) < 0.04


def test_sobol_requires_power_of_two():
    bounds = np.array([[0.0, 1.0]])
    try:
        sobol_jansen(lambda x: float(x[0]), bounds, n_base=30)
    except ValueError as exc:
        assert "power of two" in str(exc)
    else:
        raise AssertionError("expected ValueError")
