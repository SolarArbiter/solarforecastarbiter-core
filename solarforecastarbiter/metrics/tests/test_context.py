import pytest

from solarforecastarbiter.metrics import context


def test_get_default_deterministic_context():
    
    default_context = context.get_default_deterministic_context()
    
    # Check default parameters
    assert default_context['is_pv_power'] == False
    assert default_context['include_night_hours'] == False
    
    metrics = default_context['metrics']
    assert metrics['mean']  == True
    assert metrics['std']   == True
    assert metrics['mae']   == True
    assert metrics['mape']  == False
    assert metrics['mbe']   == True
    assert metrics['rmse']  == True
    assert metrics['nrmse'] == False
    assert metrics['crmse'] == False
    assert metrics['pearson_r'] == False
    assert metrics['r2_score']  == False
    assert metrics['ksi']   == False
    assert metrics['over_ksi']  == False
    assert metrics['cpi']   == False
    
    preproc = default_context['preprocessing']
    assert preproc['observations']['fill_method'] == 'exclude'
    assert preproc['observations']['fill_value'] is None
    assert preproc['observations']['max_fill_intervals'] is None
    assert preproc['forecasts']['fill_method'] == 'exclude'
    assert preproc['forecasts']['fill_value'] is None
    assert preproc['forecasts']['max_fill_intervals'] is None
    assert preproc['use_obs_interval_length'] == False
    
    results = default_context['results']
    assert results['series']['observations'] == True
    assert results['series']['forecasts'] == True
    assert results['series']['errors'] == True
    assert results['groupings']['season'] == False
    assert results['groupings']['month'] == False
    assert results['groupings']['dow'] == False
    assert results['groupings']['hod'] == True
    
    # Check setting arguments
    default_context_1 = context.get_default_deterministic_context(is_pv_power=True)
    assert default_context_1['is_pv_power'] == True
    assert default_context_1['metrics']['rmse'] == False
    assert default_context_1['metrics']['nrmse'] == True
    
    default_context_2 = context.get_default_deterministic_context(include_night_hours=True)
    assert default_context_2['include_night_hours'] == True
    
    
@pytest.mark.skip(reason="not yet implemented")
def test_get_default_event_context():
    pass


@pytest.mark.skip(reason="not yet implemented")
def test_get_default_probabilistic_context():
    pass
 