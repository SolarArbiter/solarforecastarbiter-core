import json
import pandas as pd
import pandas.testing as pdt
import pytest

from solarforecastarbiter.io import utils

# data for test Dataframe
TEST_DICT = {'var_1': [2.0, 43.9, 338.0, -199.7, 0.32],
             'var_1_flag': [0, 0, 1, 3, 3],
             'var_2': [33.88, 3.22, 1e-2, 248.71, 48.7],
             'var_2_flag': [1, 1, 1, 1, 1]}

DF_INDEX = pd.DatetimeIndex(start=pd.Timestamp('2019-01-24T00:00'),
                            freq='1min',
                            periods=5,
                            tz='UTC', name='timestamp')
TEST_DATA = pd.DataFrame(TEST_DICT, index=DF_INDEX)


@pytest.mark.parametrize('df,var_label,flag_label,default_flag,flag_value', [
    (TEST_DATA, 'var_1', 'var_1_flag', None, 0),
    (TEST_DATA, 'var_1', None, 0, 0),
    (TEST_DATA, 'var_2', 'var_2_flag', None, 1),
])
def test_obs_df_to_json(df, var_label, flag_label, default_flag, flag_value):
    converted = utils.observation_df_to_json_payload(df, var_label, flag_label,
                                                     default_flag)
    converted_dict = json.loads(converted)
    assert 'values' in converted_dict
    values = converted_dict['values']
    assert len(values) == 5
    assert values[0]['timestamp'] == '2019-01-24T00:00:00Z'
    assert values[0]['quality_flag'] == flag_value
    assert isinstance(values[0]['value'], float)


def test_obs_df_to_json_no_quality():
    with pytest.raises(KeyError):
        utils.observation_df_to_json_payload(TEST_DATA, 'var_1')


def test_forecast_series_to_json():
    series = pd.Series([0, 1, 2, 3, 4], index=pd.DatetimeIndex(
        start='2019-01-01T12:00Z', freq='5min', periods=5))
    expected = [{'value': 0.0, 'timestamp': '2019-01-01T12:00:00Z'},
                {'value': 1.0, 'timestamp': '2019-01-01T12:05:00Z'},
                {'value': 2.0, 'timestamp': '2019-01-01T12:10:00Z'},
                {'value': 3.0, 'timestamp': '2019-01-01T12:15:00Z'},
                {'value': 4.0, 'timestamp': '2019-01-01T12:20:00Z'}]
    json_out = utils.forecast_object_to_json(series)
    assert json.loads(json_out)['values'] == expected


def test_forecast_obj_to_json_nonpandas():
    with pytest.raises(TypeError):
        utils.forecast_object_to_json({'values': [0, 1, 2]})


@pytest.mark.parametrize('colname', ['val1', 'fx'])
def test_forecast_frame_to_json(colname):
    df = pd.DataFrame({colname: [0, 1, 2, 3, 4]},
                      index=pd.DatetimeIndex(
                          start='2019-01-01T12:00Z', freq='5min',
                          periods=5))
    expected = [{'value': 0.0, 'timestamp': '2019-01-01T12:00:00Z'},
                {'value': 1.0, 'timestamp': '2019-01-01T12:05:00Z'},
                {'value': 2.0, 'timestamp': '2019-01-01T12:10:00Z'},
                {'value': 3.0, 'timestamp': '2019-01-01T12:15:00Z'},
                {'value': 4.0, 'timestamp': '2019-01-01T12:20:00Z'}]
    json_out = utils.forecast_object_to_json(df, colname)
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
