import numpy as np
from solarforecastarbiter.metrics import deterministic


def test_deterministic_metrics():

    # synthetic data for testing
    n = 10
    y_true = np.random.randn(n)
    y_pred = np.random.randn(n)
    y_ref = np.random.randn(n)
    y_norm = 1.0

    mae = deterministic.mean_absolute(y_true, y_pred)
    mbe = deterministic.mean_bias(y_true, y_pred)
    rmse = deterministic.root_mean_square(y_true, y_pred)
    crmse = deterministic.centered_root_mean_square(y_true, y_pred)
    nrmse = deterministic.normalized_root_mean_square(y_true, y_pred, y_norm)
    mape = deterministic.mean_absolute_percentage(y_true, y_pred)
    s = deterministic.forecast_skill(y_true, y_pred, y_ref)
    r = deterministic.pearson_correlation_coeff(y_true, y_pred)
    r2 = deterministic.coeff_determination(y_true, y_pred)

    # scalar metrics
    assert isinstance(mae, float)
    assert isinstance(mbe, float)
    assert isinstance(rmse, float)
    assert isinstance(crmse, float)
    assert isinstance(nrmse, float)
    assert isinstance(mape, float)
    assert isinstance(s, float)
    assert isinstance(r, float)
    assert isinstance(r2, float)

    # non-negative metrics
    assert mae >= 0.0
    assert rmse >= 0.0
    assert mape >= 0.0
    assert nrmse >= 0.0

    # bounded metrics
    assert r >= -1.0
    assert r <= 1.0
    assert r2 <= 1.0

    # known results
    s = deterministic.forecast_skill(y_true, y_pred, y_pred)
    assert s == 0.0

    r2 = deterministic.coeff_determination(y_true, y_true)
    assert r2 == 1.0
