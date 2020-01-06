from functools import partial
import json
import pandas as pd
import pandas.testing as pdt
import pytest

from solarforecastarbiter.io import utils

# data for test Dataframe
TEST_DICT = {'value': [2.0, 43.9, 338.0, -199.7, 0.32],
             'quality_flag': [1, 1, 9, 5, 2]}

DF_INDEX = pd.date_range(start=pd.Timestamp('2019-01-24T00:00'),
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


@pytest.mark.parametrize('label,exp,start,end', [
    ('instant', TEST_DATA, None, None),
    (None, TEST_DATA, None, None),
    ('ending', TEST_DATA.iloc[1:], None, None),
    ('beginning', TEST_DATA.iloc[:-1], None, None),
    pytest.param('er', TEST_DATA, None, None,
                 marks=pytest.mark.xfail(raises=ValueError)),
    # start/end outside data
    ('ending', TEST_DATA, pd.Timestamp('20190123T2300Z'), None),
    ('beginning', TEST_DATA, None, pd.Timestamp('20190124T0100Z')),
    # more limited
    ('ending', TEST_DATA.iloc[2:], pd.Timestamp('20190124T0001Z'), None),
    ('beginning', TEST_DATA.iloc[:-2], None,
     pd.Timestamp('20190124T0003Z')),
    ('instant', TEST_DATA.iloc[1:-1], pd.Timestamp('20190124T0001Z'),
     pd.Timestamp('20190124T0003Z')),
])
def test_adjust_timeseries_for_interval_label(label, exp, start, end):
    start = start or pd.Timestamp('2019-01-24T00:00Z')
    end = end or pd.Timestamp('2019-01-24T00:04Z')
    out = utils.adjust_timeseries_for_interval_label(
        TEST_DATA, label, start, end)
    pdt.assert_frame_equal(exp, out)


def test_adjust_timeseries_for_interval_label_no_tz():
    test_data = TEST_DATA.tz_localize(None)
    label = None
    start = pd.Timestamp('2019-01-24T00:00Z')
    end = pd.Timestamp('2019-01-24T00:04Z')
    with pytest.raises(ValueError):
        utils.adjust_timeseries_for_interval_label(
            test_data, label, start, end)


def test_adjust_timeseries_for_interval_label_no_tz_empty():
    test_data = pd.DataFrame()
    label = None
    start = pd.Timestamp('2019-01-24T00:00Z')
    end = pd.Timestamp('2019-01-24T00:04Z')
    out = utils.adjust_timeseries_for_interval_label(
            test_data, label, start, end)
    pdt.assert_frame_equal(test_data, out)


@pytest.mark.parametrize('label,exp', [
    ('instant', TEST_DATA['value']),
    ('ending', TEST_DATA['value'].iloc[1:]),
    ('beginning', TEST_DATA['value'].iloc[:-1])
])
def test_adjust_timeseries_for_interval_label_series(label, exp):
    start = pd.Timestamp('2019-01-24T00:00Z')
    end = pd.Timestamp('2019-01-24T00:04Z')
    out = utils.adjust_timeseries_for_interval_label(
        TEST_DATA['value'], label, start, end)
    pdt.assert_series_equal(exp, out)


@pytest.mark.parametrize('ser', [
    TEST_DATA['value'],
    pd.Series([], index=pd.DatetimeIndex([], tz='UTC')),
    pytest.param(pd.Series(), marks=[
        pytest.mark.xfail(strict=True, type=TypeError)]),
    pytest.param(pd.Series([], index=pd.DatetimeIndex([])), marks=[
        pytest.mark.xfail(strict=True, type=TypeError)]),
    pytest.param(TEST_DATA, marks=[
        pytest.mark.xfail(strict=True, type=TypeError)]),
])
def test_serialize_timeseries(ser):
    out = utils.serialize_timeseries(ser)
    outd = json.loads(out)
    assert 'schema' in outd
    assert 'data' in outd


@pytest.mark.parametrize('inp,exp', [
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
     pd.Series([], name='value', index=pd.DatetimeIndex(
         [], tz='UTC', name='timestamp'))),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [], "other_stuff": {}}',  # NOQA
     pd.Series([], name='value', index=pd.DatetimeIndex(
         [], tz='UTC', name='timestamp'))),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "alue", "index": "timestamp", "dtype": "float64"}, "more": [], "data": []}',  # NOQA
     pd.Series([], name='alue', index=pd.DatetimeIndex(
         [], tz='UTC', name='timestamp'))),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "more": [], "data": [], "other": []}',  # NOQA
     pd.Series([], name='value', index=pd.DatetimeIndex(
         [], tz='UTC', name='timestamp'))),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [{"timestamp": "2019-01-01T00:00:00Z", "value": 1.0}], "other_stuff": {}}',  # NOQA
     pd.Series([1.0], index=pd.DatetimeIndex(["2019-01-01T00:00:00"],
                                             tz='UTC', name='timestamp'),
               name='value')),
    pytest.param(
        '{"data": [], "other_stuff": {}}',
        pd.Series(),
        marks=[pytest.mark.xfail(strict=True, type=ValueError)]),
    pytest.param(
        '{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "other_stuff": {}}',  # NOQA
        pd.Series(),
        marks=[pytest.mark.xfail(strict=True, type=ValueError)]),
    pytest.param(
        '{"schema": {"version": 0, "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
        pd.Series(),
        marks=[pytest.mark.xfail(strict=True, type=KeyError)]),
])
def test_deserialize_timeseries(inp, exp):
    out = utils.deserialize_timeseries(inp)
    pdt.assert_series_equal(out, exp)


def test_serialize_roundtrip():
    ser = utils.serialize_timeseries(TEST_DATA['value'])
    out = utils.deserialize_timeseries(ser)
    pdt.assert_series_equal(out, TEST_DATA['value'])


def test_hidden_token():
    ht = utils.HiddenToken('THETOKEN')
    assert str(ht) != 'THETOKEN'
    assert ht.token == 'THETOKEN'


@pytest.fixture(params=[0, 1, 2, 3])
def start_end(request):
    if request.param == 0:
        return (pd.Timestamp('2019-01-01T12:00:00Z'),
                pd.Timestamp('2019-01-12T17:47:22-0700'))
    elif request.param == 1:
        return ('2019-01-01T12:00:00Z',
                pd.Timestamp('2019-01-12T17:47:22-0700'))
    elif request.param == 2:
        return ('2019-01-01T12:00:00Z',
                '2019-01-12T17:47:22-0700')
    elif request.param == 3:
        return (pd.Timestamp('2019-01-01T12:00:00Z'),
                '2019-01-12T17:47:22-0700')


@pytest.fixture(params=[0, 1, 2, 3, 4])
def tfunc(request, start_end):
    start, end = start_end
    if request.param == 0:
        def f(start, end): return start, end
        return partial(f, start, end)
    elif request.param == 1:
        def f(start, end): return start, end
        return partial(f, start=start, end=end)
    elif request.param == 2:
        def f(start, *, end): return start, end
        return partial(f, start, end=end)
    elif request.param == 3:
        def f(te, start, more=None, *, end): return start, end
        return partial(f, '', start, end=end)
    elif request.param == 4:
        def f(te, startitat, end, more=None, start=None): return start, end
        return partial(f, '', 'now', start=start, end=end)


def test_ensure_timestamps_partials(tfunc):
    dec_f = utils.ensure_timestamps('start', 'end')(tfunc.func)
    s, e = dec_f(*tfunc.args, **tfunc.keywords)
    assert s == pd.Timestamp('2019-01-01T12:00:00Z')
    assert e == pd.Timestamp('2019-01-12T17:47:22-0700')


def test_ensure_timestamps_normal(start_end):
    @utils.ensure_timestamps('start', 'end')
    def f(other, start, end, x=None):
        """cool docstring"""
        return start, end
    s, e = start_end
    start, end = f('', s, e)
    assert f.__doc__ == 'cool docstring'
    assert start == pd.Timestamp('2019-01-01T12:00:00Z')
    assert end == pd.Timestamp('2019-01-12T17:47:22-0700')


def test_ensure_timestamps_other_args():
    @utils.ensure_timestamps('x')
    def f(other, start, end, x=None):
        """cool docstring"""
        return start, end, x
    start, end, x = f('', 'a', 'end', '2019-01-01T12:00:00Z')
    assert f.__doc__ == 'cool docstring'
    assert x == pd.Timestamp('2019-01-01T12:00:00Z')
    assert start == 'a'
    assert end == 'end'


def test_ensure_timestamps_err():
    @utils.ensure_timestamps('start', 'end', 'x')
    def f(other, start, end, x=None):  # pragma: no cover
        return start, end
    with pytest.raises(ValueError):
        start, end = f('', '2019-09-01T12:00Z', 'blah')
