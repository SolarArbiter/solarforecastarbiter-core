import numpy as np


from solarforecastarbiter.metrics import cost


def test_band_masks(ferc890_cost_params):
    err = np.array([0, 0, -1, .1, 2.5, 2.1, 1.9, -3, 100])
    masks = cost._band_masks(ferc890_cost_params.parameters.bands,
                             err)
    expmasks = np.array([
        np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool),
        np.array([1, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool),
        np.array([1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=bool),
    ])
    assert (masks == expmasks).all()


def test_error_band_cost_wrapper(ferc890_cost_params):
    fnc = cost.error_band_cost_wrapper(ferc890_cost_params.parameters)
    obs = np.array([1, 1, 1, 2, 2.00, 2.0, 0, 4])
    fx = np.array([1., 2, 3, 0, -0.1, 1.9, 3, 5])
    err = fnc(obs, fx, lambda x, y: y - x)
    exp = ((0 + 1 + 2 + -2 + 0.0 + -0.1 + 0 + 1) +
           (0 + 0 + 0 + 0. + 0.0 + 0.00 + 3 + 0) * 1.1 +
           (0 + 0 + 0 + 0. + 2.1 + 0.00 + 0 + 0) * -0.9)
    assert err == exp
