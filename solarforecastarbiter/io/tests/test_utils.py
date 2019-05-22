import json
import pandas as pd
import pandas.testing as pdt
import pytest

from solarforecastarbiter.io import utils

# data for test Dataframe
TEST_DICT = {'value': [2.0, 43.9, 338.0, -199.7, 0.32],
             'quality_flag': [1, 1, 9, 5, 2]}

DF_INDEX = pd.DatetimeIndex(start=pd.Timestamp('2019-01-24T00:00'),
                            freq='1min',
                            periods=5,
                            tz='UTC', name='timestamp')
TEST_DATA = pd.DataFrame(TEST_DICT, index=DF_INDEX)


@pytest.mark.parametrize('dump_quality,default_flag,flag_value', [
    (False, None, 1),
    (True, 2, 2)
])
def test_obs_df_to_json(dump_quality, default_flag, flag_value):
    td = TEST_DATA.copy()
    if dump_quality:
        del td['quality_flag']
    converted = utils.observation_df_to_json_payload(td, default_flag)
    converted_dict = json.loads(converted)
    assert 'values' in converted_dict
    values = converted_dict['values']
    assert len(values) == 5
    assert values[0]['timestamp'] == '2019-01-24T00:00:00Z'
    assert values[0]['quality_flag'] == flag_value
    assert isinstance(values[0]['value'], float)


def test_obs_df_to_json_no_quality():
    td = TEST_DATA.copy()
    del td['quality_flag']
    with pytest.raises(KeyError):
        utils.observation_df_to_json_payload(td)


def test_obs_df_to_json_no_values():
    td = TEST_DATA.copy().rename(columns={'value': 'val1'})
    with pytest.raises(KeyError):
        utils.observation_df_to_json_payload(td)


def test_forecast_series_to_json():
    series = pd.Series([0, 1, 2, 3, 4], index=pd.date_range(
        start='2019-01-01T12:00Z', freq='5min', periods=5))
    expected = [{'value': 0.0, 'timestamp': '2019-01-01T12:00:00Z'},
                {'value': 1.0, 'timestamp': '2019-01-01T12:05:00Z'},
                {'value': 2.0, 'timestamp': '2019-01-01T12:10:00Z'},
                {'value': 3.0, 'timestamp': '2019-01-01T12:15:00Z'},
                {'value': 4.0, 'timestamp': '2019-01-01T12:20:00Z'}]
    json_out = utils.forecast_object_to_json(series)
    assert json.loads(json_out)['values'] == expected


def test_json_payload_to_observation_df(observation_values,
                                        observation_values_text):
    out = utils.json_payload_to_observation_df(
        json.loads(observation_values_text))
    pdt.assert_frame_equal(out, observation_values)


def test_json_payload_to_forecast_series(forecast_values,
                                         forecast_values_text):
    out = utils.json_payload_to_forecast_series(
        json.loads(forecast_values_text))
    pdt.assert_series_equal(out, forecast_values)


def test_empty_payload_to_obsevation_df():
    out = utils.json_payload_to_observation_df({'values': []})
    assert set(out.columns) == {'value', 'quality_flag'}
    assert isinstance(out.index, pd.DatetimeIndex)


def test_empty_payload_to_forecast_series():
    out = utils.json_payload_to_forecast_series({'values': []})
    assert isinstance(out.index, pd.DatetimeIndex)


def test_hidden_token():
    ht = utils.HiddenToken('THETOKEN')
    assert str(ht) != 'THETOKEN'
    assert ht.token == 'THETOKEN'
