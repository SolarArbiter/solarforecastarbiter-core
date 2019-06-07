import pytest

from solarforecastarbiter.metrics import context


def test_get_default_deterministic_context():

    default_context = context.get_default_deterministic_context()

    # Check default parameters
    assert default_context['is_pv_power'] is False
    assert default_context['include_night_hours'] is False
    assert default_context['timezone'] is None

    metrics = default_context['metrics']
    assert metrics['mean'] is False
    assert metrics['std'] is False
    assert metrics['mae'] is True
    assert metrics['mape'] is False
    assert metrics['mbe'] is True
    assert metrics['rmse'] is True
    assert metrics['nrmse'] is False
    assert metrics['crmse'] is False
    assert metrics['pearson_r'] is False
    assert metrics['r2_score'] is False
    assert metrics['ksi'] is False
    assert metrics['over_ksi'] is False
    assert metrics['cpi'] is False

    preproc = default_context['preprocessing']
    assert preproc['observations']['fill_method'] == 'exclude'
    assert preproc['observations']['fill_value'] is None
    assert preproc['observations']['max_fill_intervals'] is None
    assert preproc['forecasts']['fill_method'] == 'exclude'
    assert preproc['forecasts']['fill_value'] is None
    assert preproc['forecasts']['max_fill_intervals'] is None
    assert preproc['use_obs_interval_length'] is False

    results = default_context['results']
    assert results['timeseries']['observations'] is True
    assert results['timeseries']['forecasts'] is True
    assert results['groupings']['month'] is False
    assert results['groupings']['weekday'] is False
    assert results['groupings']['hour'] is True
    assert results['groupings']['date'] is False

    # Check setting arguments
    default_context_1 = context.get_default_deterministic_context(
        is_pv_power=True)
    assert default_context_1['is_pv_power'] is True
    assert default_context_1['metrics']['rmse'] is False
    assert default_context_1['metrics']['nrmse'] is True

    default_context_2 = context.get_default_deterministic_context(
        include_night_hours=True)
    assert default_context_2['include_night_hours'] is True


@pytest.mark.xfail(raises=NotImplementedError)
def test_get_default_event_context():
    pass


@pytest.mark.xfail(raises=NotImplementedError)
def test_get_default_probabilistic_context():
    pass
