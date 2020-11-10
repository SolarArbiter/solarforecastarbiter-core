from functools import partial
import json
import numpy as np
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
EMPTY_SERIES = pd.Series(dtype=float)
EMPTY_TIMESERIES = pd.Series([], name='value', index=pd.DatetimeIndex(
    [], name='timestamp', tz='UTC'), dtype=float)

EMPTY_DATAFRAME = pd.DataFrame(dtype=float)
EMPTY_TIME_DATAFRAME = pd.DataFrame([], index=pd.DatetimeIndex(
    [], name='timestamp', tz='UTC'), dtype=float)
TEST_DATAFRAME = pd.DataFrame({
    '25.0': [0.0, 1, 2, 3, 4, 5],
    '50.0': [1.0, 2, 3, 4, 5, 6],
    '75.0': [2.0, 3, 4, 5, 6, 7]},
    index=pd.date_range(start='20190101T0600',
                        end='20190101T1100',
                        freq='1h',
                        tz='America/Denver',
                        name='timestamp')).tz_convert('UTC')


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


def test_null_json_payload_to_observation_df():
    observation_values_text = b"""
{
  "_links": {
    "metadata": ""
  },
  "observation_id": "OBSID",
  "values": [
    {
      "quality_flag": 1,
      "timestamp": "2019-01-01T12:00:00-0700",
      "value": null
    },
    {
      "quality_flag": 1,
      "timestamp": "2019-01-01T12:05:00-0700",
      "value": null
    }
  ]
}"""
    ind = pd.DatetimeIndex([
        pd.Timestamp("2019-01-01T19:00:00Z"),
        pd.Timestamp("2019-01-01T19:05:00Z")
    ], name='timestamp')
    observation_values = pd.DataFrame({
        'value': pd.Series([None, None], index=ind, dtype=float),
        'quality_flag': pd.Series([1, 1], index=ind)
    })
    out = utils.json_payload_to_observation_df(
        json.loads(observation_values_text))
    pdt.assert_frame_equal(out, observation_values)


def test_null_json_payload_to_forecast_series():
    forecast_values_text = b"""
{
  "_links": {
    "metadata": ""
  },
  "forecast_id": "OBSID",
  "values": [
    {
      "timestamp": "2019-01-01T12:00:00-0700",
      "value": null
    },
    {
      "timestamp": "2019-01-01T12:05:00-0700",
      "value": null
    }
  ]
}"""
    ind = pd.DatetimeIndex([
        pd.Timestamp("2019-01-01T19:00:00Z"),
        pd.Timestamp("2019-01-01T19:05:00Z")
    ], name='timestamp')
    forecast_values = pd.Series([None, None], index=ind, dtype=float,
                                name='value')
    out = utils.json_payload_to_forecast_series(
        json.loads(forecast_values_text))
    pdt.assert_series_equal(out, forecast_values)


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


@pytest.mark.parametrize('inp,exp', [
    (TEST_DATA['value'], '{"schema":{"version": 1, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64", "objtype": "Series"},"data":[{"timestamp":"2019-01-24T00:00:00Z","value":2.0},{"timestamp":"2019-01-24T00:01:00Z","value":43.9},{"timestamp":"2019-01-24T00:02:00Z","value":338.0},{"timestamp":"2019-01-24T00:03:00Z","value":-199.7},{"timestamp":"2019-01-24T00:04:00Z","value":0.32}]}'),  # NOQA: E501
    (EMPTY_TIMESERIES, '{"schema":{"version": 1, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64", "objtype": "Series"},"data":[]}'),  # NOQA: E501
    pytest.param(EMPTY_SERIES, '', marks=[
        pytest.mark.xfail(strict=True, type=TypeError)]),
    pytest.param(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), '',
                 marks=[pytest.mark.xfail(strict=True, type=TypeError)]),
    (TEST_DATAFRAME, '{"schema":{"version": 1, "orient": "records", "timezone": "UTC", "column": ["25.0", "50.0", "75.0"], "index": "timestamp", "dtype": ["float64", "float64", "float64"], "objtype": "DataFrame"},"data":[{"timestamp":"2019-01-01T13:00:00Z","25.0":0.0,"50.0":1.0,"75.0":2.0},{"timestamp":"2019-01-01T14:00:00Z","25.0":1.0,"50.0":2.0,"75.0":3.0},{"timestamp":"2019-01-01T15:00:00Z","25.0":2.0,"50.0":3.0,"75.0":4.0},{"timestamp":"2019-01-01T16:00:00Z","25.0":3.0,"50.0":4.0,"75.0":5.0},{"timestamp":"2019-01-01T17:00:00Z","25.0":4.0,"50.0":5.0,"75.0":6.0},{"timestamp":"2019-01-01T18:00:00Z","25.0":5.0,"50.0":6.0,"75.0":7.0}]}'),  # NOQA: E501
    (EMPTY_TIME_DATAFRAME, '{"schema":{"version": 1, "orient": "records", "timezone": "UTC", "column": [], "index": "timestamp", "dtype": [], "objtype": "DataFrame"},"data":[]}'),  # NOQA: E501
    pytest.param(EMPTY_DATAFRAME, '', marks=[
        pytest.mark.xfail(strict=True, type=TypeError)]),
])
def test_serialize_timeseries(inp, exp):
    out = utils.serialize_timeseries(inp)
    outd = json.loads(out)
    assert 'schema' in outd
    assert 'data' in outd
    assert out == exp


@pytest.mark.parametrize('inp,exp', [
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
     EMPTY_TIMESERIES),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "US/Arizona", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
     pd.Series([], name='value', index=pd.DatetimeIndex(
         [], tz='US/Arizona', name='timestamp'), dtype=float)),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [], "other_stuff": {}}',  # NOQA
     EMPTY_TIMESERIES),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "alue", "index": "timestamp", "dtype": "float64"}, "more": [], "data": []}',  # NOQA
     pd.Series([], name='alue', index=pd.DatetimeIndex(
         [], tz='UTC', name='timestamp'), dtype=float)),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "more": [], "data": [], "other": []}',  # NOQA
     EMPTY_TIMESERIES),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [{"timestamp": "2019-01-01T00:00:00Z", "value": 1.0}], "other_stuff": {}}',  # NOQA
     pd.Series([1.0], index=pd.DatetimeIndex(["2019-01-01T00:00:00"],
                                             tz='UTC', name='timestamp'),
               name='value')),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [{"timestamp": "2019-01-01T00:00:00", "value": 1.0}], "other_stuff": {}}',  # NOQA
     pd.Series([1.0], index=pd.DatetimeIndex(["2019-01-01T00:00:00"],
                                             tz='UTC', name='timestamp'),
               name='value')),
    ('{"schema": {"version": 0, "orient": "records", "timezone": "Etc/GMT+8", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": [{"timestamp": "2019-01-01T00:00:00", "value": 1.0}], "other_stuff": {}}',  # NOQA
     pd.Series([1.0], index=pd.DatetimeIndex(["2019-01-01T00:00:00"],
                                             tz='Etc/GMT+8', name='timestamp'),
               name='value')),
    pytest.param(
        '{"data": [], "other_stuff": {}}',
        EMPTY_SERIES,
        marks=[pytest.mark.xfail(strict=True, type=ValueError)]),
    pytest.param(
        '{"schema": {"version": 0, "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "other_stuff": {}}',  # NOQA
        EMPTY_SERIES,
        marks=[pytest.mark.xfail(strict=True, type=ValueError)]),
    pytest.param(
        '{"schema": {"version": 0, "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
        EMPTY_SERIES,
        marks=[pytest.mark.xfail(strict=True, type=KeyError)]),
    ('{"schema": {"version": 1, "objtype": "Series", "orient": "records", "timezone": "UTC", "column": "value", "index": "timestamp", "dtype": "float64"}, "data": []}',  # NOQA
     EMPTY_TIMESERIES)
])
def test_deserialize_timeseries(inp, exp):
    out = utils.deserialize_timeseries(inp)
    pdt.assert_series_equal(out, exp)


@pytest.mark.parametrize('inp,exp', [
    ('{"schema":{"version": 1, "orient": "records", "timezone": "UTC", "column": [], "index": "timestamp", "dtype": [], "objtype": "DataFrame"},"data":[]}',  # NOQA
     EMPTY_TIME_DATAFRAME),
    ('{"schema": {"version": 1, "objtype": "DataFrame", "orient": "records", "timezone": "UTC", "column": [], "index": "timestamp", "dtype": ["float64"]}, "data": []}',  # NOQA
     EMPTY_TIME_DATAFRAME),
    ('{"schema": {"version": 1, "objtype": "DataFrame", "orient": "records", "timezone": "UTC", "column": ["25.0"], "index": "timestamp", "dtype": ["float64"]}, "data": [{"timestamp": "2019-01-01T00:00:00", "25.0": 1.0}], "other_stuff": {}}',  # NOQA
     pd.DataFrame({'25.0': 1.0}, index=pd.DatetimeIndex(
         ["2019-01-01T00:00:00"], tz='UTC', name='timestamp'))),
    ('{"schema": {"version": 1, "objtype": "DataFrame", "orient": "records", "timezone": "Etc/GMT+8", "column": ["25.0"], "index": "timestamp", "dtype": ["float64"]}, "data": [{"timestamp": "2019-01-01T00:00:00", "25.0": 1.0}], "other_stuff": {}}',  # NOQA
     pd.DataFrame({'25.0': 1.0}, index=pd.DatetimeIndex(
         ["2019-01-01T00:00:00"], tz='Etc/GMT+8', name='timestamp'))),
])
def test_deserialize_timeseries_frame(inp, exp):
    out = utils.deserialize_timeseries(inp)
    pdt.assert_frame_equal(out, exp)


def test_serialize_roundtrip():
    ser = utils.serialize_timeseries(TEST_DATA['value'])
    out = utils.deserialize_timeseries(ser)
    pdt.assert_series_equal(out, TEST_DATA['value'])


# use the conftest.py dataframe for security against refactoring
def test_serialize_roundtrip_frame(prob_forecast_values):
    ser = utils.serialize_timeseries(prob_forecast_values)
    out = utils.deserialize_timeseries(ser)
    pdt.assert_frame_equal(out, prob_forecast_values)


def test_serialize_roundtrip_frame_floats(prob_forecast_values):
    # check that roundtrip is not faithful if input columns is Float64Index
    prob_forecast_floats = prob_forecast_values.copy()
    prob_forecast_floats.columns = prob_forecast_floats.columns.astype(float)
    ser = utils.serialize_timeseries(prob_forecast_floats)
    out = utils.deserialize_timeseries(ser)
    pdt.assert_frame_equal(out, prob_forecast_values)


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


def test_ensure_timestamps_omitted_args():
    @utils.ensure_timestamps('start', 'end', 'other')
    def f(start, end, a, *, other):  # pragma: no cover
        return start, end

    with pytest.raises(
            TypeError, match="missing a required argument: 'end'"):
        start, end = f('2020-01-01T00:00Z')

    with pytest.raises(
            TypeError, match="missing a required argument: 'a'"):
        start, end = f('2020-01-01T00:00Z', '2020-01-02T00:00Z')

    with pytest.raises(
            TypeError, match="missing a required argument: 'other'"):
        start, end = f('2020-01-01T00:00Z', end='2020-01-02T00:00Z', a=0)

    f('2020-01-01T00:00Z', '2020-01-02T00:00Z', 0, other=None)
    with pytest.raises(ValueError):
        f('2020-01-01T00:00Z', '2020-01-02T00:00Z', 0, other='a')


def test_load_report_values(raw_report, report_objects):
    _, obs, fx0, fx1, agg, fxagg = report_objects
    ser = pd.Series(np.random.random(10),
                    name='value', index=pd.date_range(
                        start='20200101', freq='5min', periods=10,
                        tz='UTC', name='timestamp'))
    val = utils.serialize_timeseries(ser)
    vals = [{'id': id_, 'processed_values': val} for id_ in
            (fx0.forecast_id, fx1.forecast_id, obs.observation_id,
             agg.aggregate_id, fxagg.forecast_id)]
    inp = raw_report(False)
    out = utils.load_report_values(inp, vals)
    for fxo in out:
        pdt.assert_series_equal(fxo.forecast_values, ser)
        pdt.assert_series_equal(fxo.observation_values, ser)
