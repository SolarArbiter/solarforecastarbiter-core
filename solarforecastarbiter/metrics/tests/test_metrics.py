import numpy as np
from solarforecastarbiter.metrics import deterministic


def test_deterministic_metrics():
    # generate synthetic data for testing
    n = 10
    y_true = np.random.randn(n)
    y_pred = np.random.randn(n)

    mae = deterministic.mean_absolute(y_true, y_pred)
    mbe = deterministic.mean_bias(y_true, y_pred)
    rmse = deterministic.root_mean_square(y_true, y_pred)

    assert isinstance(mae, float)
    assert isinstance(mbe, float)
    assert isinstance(rmse, float)

    assert mae >= 0
    assert rmse >= 0
