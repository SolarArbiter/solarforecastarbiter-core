"""Collection of Functions to convert API responses into python objects
and vice versa.
"""
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import json
import re


import pandas as pd


def _dataframe_to_json(payload_df):
    payload_df.index.name = 'timestamp'
    json_vals = payload_df.tz_convert("UTC").reset_index().to_json(
        orient="records", date_format='iso', date_unit='s')
    return '{"values":' + json_vals + '}'


def observation_df_to_json_payload(
        observation_df, default_quality_flag=None):
    """Extracts a variable from an observation DataFrame and formats it
    into a JSON payload for posting to the Solar Forecast Arbiter API.

    Parameters
    ----------
    observation_df : DataFrame
        Dataframe of observation data. Must contain a tz-aware DateTimeIndex
        and a 'value' column. May contain a column of data quality
        flags labeled 'quality_flag'.
    default_quality_flag : int
        If 'quality_flag' is not a column, the quality flag for each row is
        set to this value.

    Returns
    -------
    string
        SolarForecastArbiter API JSON payload for posting to the observation
        endpoint. See Notes section for example.

    Notes
    -----
    Function returns an object in the following format:

    .. code::

        {
          'values': [
            {
              “timestamp”: “2018-11-22T12:01:48Z”, # ISO 8601 datetime in UTC
              “value”: 10.23, # floating point value of observation
              “quality_flag”: 0
            },...
          ]
        }

    Raises
    ------
    KeyError
       When 'value' is missing from the columns or 'quality_flag'
       is missing and default_quality_flag is None
    """
    if default_quality_flag is None:
        payload_df = observation_df[['value', 'quality_flag']]
    else:
        payload_df = observation_df[['value']]
        payload_df['quality_flag'] = int(default_quality_flag)
    return _dataframe_to_json(payload_df)


def forecast_object_to_json(forecast_series):
    """
    Converts a forecast Series to JSON to post to the
    SolarForecastArbiter API.

    Parameters
    ----------
    forecast_series : pandas.Series
        The series that contains the forecast values with a
        datetime index.

    Returns
    -------
    string
        The JSON encoded forecast values dict
    """
    payload_df = forecast_series.to_frame('value')
    return _dataframe_to_json(payload_df)


def _json_to_dataframe(json_payload):
    # in the future, might worry about reading the response in chunks
    # to stream the data and avoid having it all in memory at once,
    # but 30 days of 1 minute data is probably ~4 MB of text. A better
    # approach would probably be to switch to a binary format.
    vals = json_payload['values']
    if len(vals) == 0:
        df = pd.DataFrame([], columns=['value', 'quality_flag'],
                          index=pd.DatetimeIndex([], name='timestamp',
                                                 tz='UTC'))
    else:
        df = pd.DataFrame.from_dict(json_payload['values'])
        df.index = pd.to_datetime(df['timestamp'], utc=True,
                                  infer_datetime_format=True)
    return df


def json_payload_to_observation_df(json_payload):
    """
    Convert the JSON payload dict as returned by the SolarForecastArbiter API
    observations/values endpoint into a DataFrame

    Parameters
    ----------
    json_payload : dict
        Dictionary as returned by the API with a "values" key which is a list
        of dicts like {'timestamp': <timestamp>, 'value': <float>,
        'quality_flag': <int>}

    Returns
    -------
    pandas.DataFrame
       With a tz-aware DatetimeIndex and ['value', 'quality_flag'] columns
       and dtypes {'value': float, 'quality_flag': int}
    """
    df = _json_to_dataframe(json_payload)
    return df[['value', 'quality_flag']].astype(
        {'value': float, 'quality_flag': int})


def json_payload_to_forecast_series(json_payload):
    """
    Convert the JSON payload dict as returned by the SolarForecastArbiter API
    forecasts/values endpoing into a Series

    Parameters
    ----------
    json_payload : dict
        Dictionary as returned by the API with a "values" key which is a list
        of dicts like {'timestamp': <timestamp>, 'value': <float>}

    Returns
    -------
    pandas.Series
       With a tz-aware DatetimeIndex and float dtype
    """

    df = _json_to_dataframe(json_payload)
    return df['value'].astype(float)


def adjust_start_end_for_interval_label(interval_label, start, end,
                                        limit_instant=False):
    """
    Adjusts the start and end times depending on the interval_label.

    Parameters
    ----------
    interval_label : str or None
       The interval label for the the object the data represents
    start : pandas.Timestamp
       Start time to restrict data to
    end : pandas.Timestamp
       End time to restrict data to
    limit_instant : boolean
       If true, an interval label of 'instant' will remove a nanosecond
       from end to ensure forecasts do not overlap. If False, instant
       returns start, end unmodified

    Returns
    -------
    start, end
       Return the adjusted start and end

    Raises
    ------
    ValueError
       If an invalid interval_label is given

    Examples
    --------
    .. testsetup::

       from solarforecastarbiter.io.utils import *

    Define input start/end:

    >>> start = pd.Timestamp('20190101 1200Z')
    >>> end = pd.Timestamp('20190101 1300Z')

    Beginning:

    >>> adjust_start_end_for_interval_label('beginning', start, end)
    (Timestamp('2019-01-01 12:00:00+0000', tz='UTC'), Timestamp('2019-01-01 12:59:59.999999999+0000', tz='UTC'))

    Ending:

    >>> adjust_start_end_for_interval_label('ending', start, end)
    (Timestamp('2019-01-01 12:00:00.000000001+0000', tz='UTC'), Timestamp('2019-01-01 13:00:00+0000', tz='UTC'))

    Instantaneous:

    >>> adjust_start_end_for_interval_label('instant', start, end)
    (Timestamp('2019-01-01 12:00:00+0000', tz='UTC'), Timestamp('2019-01-01 13:00:00+0000', tz='UTC'))

    >>> adjust_start_end_for_interval_label('instant', start, end,
    ...                                     limit_instant=True)
    (Timestamp('2019-01-01 12:00:00+0000', tz='UTC'), Timestamp('2019-01-01 12:59:59.999999999+0000', tz='UTC'))

    """ # NOQA

    if (
            interval_label is not None and
            interval_label not in ('instant', 'beginning', 'ending')
    ):
        raise ValueError('Invalid interval_label')

    if (
            interval_label == 'beginning' or
            (interval_label == 'instant' and limit_instant)
    ):
        end -= pd.Timedelta(1, unit='nano')
    elif interval_label == 'ending':
        start += pd.Timedelta(1, unit='nano')
    return start, end


def adjust_timeseries_for_interval_label(data, interval_label, start, end):
    """
    Adjusts the index of the data depending on the interval_label, start,
    and end. Will always return the data located between start, end.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
       The data with a localized DatetimeIndex
    interval_label : str or None
       The interval label for the the object the data represents
    start : pandas.Timestamp
       Start time to restrict data to
    end : pandas.Timestamp
       End time to restrict data to

    Returns
    -------
    pandas.Series or pandas.DataFrame
       Return data between start and end, in/excluding the endpoints
       depending on interval_label

    Raises
    ------
    ValueError
       If an invalid interval_label is given or data is not localized.
    """
    start, end = adjust_start_end_for_interval_label(interval_label, start,
                                                     end)
    data = data.sort_index(axis=0)
    # pandas >= 0.25.1 requires start, end to have same tzinfo.
    # unexpected behavior when data is not localized, so prevent that
    if data.empty:
        return data
    if data.index.tzinfo is None:
        raise ValueError('data must be localized')
    start = start.tz_convert(data.index.tzinfo)
    end = end.tz_convert(data.index.tzinfo)
    return data.loc[start:end]


def serialize_timeseries(ser):
    """
    Serialize a timeseries to JSON. Note that the microseconds
    portion of the index will be discarded.

    Parameters
    ----------
    ser : {pandas.Series, pandas.DataFrame}
       Must have a tz-localized datetime index

    Returns
    -------
    str
        The JSON serialized data along with a schema

    Raises
    ------
    TypeError
        If the input is invalid
    """
    if not (
            isinstance(ser, (pd.Series, pd.DataFrame)) and
            isinstance(ser.index, pd.DatetimeIndex) and
            ser.index.tzinfo is not None
    ):
        raise TypeError(
            'Only pandas Series or DataFrame with a localized DatetimeIndex '
            'is supported')
    v = ser.copy()
    v.index.name = 'timestamp'
    if isinstance(v, pd.Series):
        jsonvals = v.tz_convert('UTC').reset_index(name='value').to_json(
            orient='records', date_format='iso', date_unit='s')
        column = 'value'
        dtype = str(v.dtype)
        objtype = 'Series'
    else:
        v.index.name = 'timestamp'
        jsonvals = v.tz_convert('UTC').reset_index().to_json(
            orient='records', date_format='iso', date_unit='s')
        column = v.columns.astype(str).to_list()
        dtype = v.dtypes.astype(str).to_list()
        objtype = 'DataFrame'
    schema = {
        'version': 1,
        'orient': 'records',
        'timezone': 'UTC',
        'column': column,
        'index': 'timestamp',
        'dtype': dtype,
        'objtype': objtype
    }
    out = '{"schema":' + json.dumps(schema) + ',"data":' + jsonvals + '}'
    return out


def deserialize_timeseries(data):
    """
    Deserializes a timeseries from JSON

    Parameters
    ----------
    data : str
        JSON string to deserialize. Must have schema and data keys.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Deserialized timeseries

    Raises
    ------
    ValueError
       If "schema" or "data" keys are not found in the JSON string
    KeyError
       If the schema object does not contain the proper keys
    """
    schema_str = re.search('(?<="schema":)\\s*{[^{}]*}\\s*(?=(,|}))', data)
    if schema_str is None:
        raise ValueError('Could not locate schema in data string')
    schema = json.loads(schema_str.group(0))
    if schema['version'] == 0:
        # compatibility with data serialized and stored before the
        # objtype key was added to schema and DataFrames were suppored in v1
        objtype_str = 'Series'
    else:
        objtype_str = schema['objtype']
    # find between "data": and , or }, with only one set of []
    data_str = re.search('(?<="data":)\\s*\\[[^\\[\\]]*\\](?=\\s*(,|}))', data)
    if data_str is None:
        raise ValueError('Could not locate data key in data string')
    df = pd.read_json(data_str.group(0), orient=schema['orient'],
                      convert_dates=True)
    if objtype_str == 'Series':
        if df.empty:
            return pd.Series([], name=schema['column'], index=pd.DatetimeIndex(
                [], tz=schema['timezone'], name='timestamp'),
                            dtype=schema.get('dtype', float))
        out = df.set_index(schema['index'])[schema['column']].astype(
            schema['dtype'])
    elif objtype_str == 'DataFrame':
        if df.empty:
            try:
                dtype = schema['dtype'][0]
            except IndexError:
                dtype = float
            return pd.DataFrame(
                [], columns=schema['column'], index=pd.DatetimeIndex(
                    [], tz=schema['timezone'], name='timestamp'), dtype=dtype)
        out = df.set_index(schema['index'])
        # pd.read_json will set all column names to strings, so
        # columns originally specified with float names need to be
        # mapped back into the right name dtype. this code is not needed
        # if columns are always strings.
        # str_col_map = {str(col): col for col in schema['column']}
        # out = out.rename(columns=str_col_map)
        out = out.astype(dict(zip(schema['column'], schema['dtype'])))
    if out.index.tzinfo is None:
        out = out.tz_localize(schema['timezone'])
    return out


class HiddenToken:
    """
    Obscure the representation of the input string `token` to avoid saving
    or displaying access tokens in logs.
    """
    def __init__(self, token):
        self.token = str(token)  # make sure it isn't a localproxy

    def __repr__(self):
        return '****ACCESS*TOKEN****'


def ensure_timestamps(*time_args):
    """
    Decorator that converts the specified time arguments of the wrapped
    function to pandas.Timestamp objects

    Parameters
    ----------
    strings
       Function arguments to convert to pandas.Timestamp before
       executing function

    Raises
    ------
    ValueError
        If any of time_args cannot be converted to pandas.Timestamp

    Examples
    --------
    .. testsetup::

       import datetime as dt
       from solarforecastarbiter.io.utils import *

    >>> @ensure_timestamps('start', 'end')
    ... def get_values(start, end, other_arg):
    ...     # do stuff with start, end assumed to be pandas.Timestamps
    ...     if isinstance(start, pd.Timestamp):
    ...         return True

    >>> get_values('2019-01-01T00:00Z', dt.datetime(2019, 1, 2, 12), 'other')
    True
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            sig = signature(f)
            inds = {k: None for k in time_args}
            for i, k in enumerate(sig.parameters.keys()):
                if k in inds:
                    inds[k] = i
            nargs = list(args)
            for k, ind in inds.items():
                if k in kwargs:
                    kwargs[k] = pd.Timestamp(kwargs[k])
                elif ind is not None:
                    try:
                        nargs[ind] = pd.Timestamp(args[ind])
                    except IndexError:
                        pass
            sig.bind(*nargs, **kwargs)
            return f(*nargs, **kwargs)
        return wrapper
    return decorator


def load_report_values(raw_report, values):
    """
    Load the processed forecast/observation data into the
    datamodel.ProcessedForecastObservation objects of the raw_report.

    Parameters
    ----------
    raw_report : datamodel.RawReport
        The raw report with processed_forecasts_observations to
        be replaced
    values : list
        The report values dict as returned by the API.

    Returns
    -------
    tuple
       Of datamodel.ProcessedForecastObservation with values loaded into
       `forecast_values` and `observation_values`
    """
    val_dict = {v['id']: v['processed_values'] for v in values}
    out = []
    for fxobs in raw_report.processed_forecasts_observations:
        fx_vals = val_dict.get(fxobs.forecast_values, None)
        if fx_vals is not None:
            fx_vals = deserialize_timeseries(fx_vals)
        obs_vals = val_dict.get(fxobs.observation_values, None)
        if obs_vals is not None:
            obs_vals = deserialize_timeseries(obs_vals)
        ref_fx_vals = val_dict.get(fxobs.reference_forecast_values)
        if ref_fx_vals is not None:
            ref_fx_vals = deserialize_timeseries(ref_fx_vals)
        new_fxobs = fxobs.replace(forecast_values=fx_vals,
                                  observation_values=obs_vals,
                                  reference_forecast_values=ref_fx_vals)
        out.append(new_fxobs)
    return tuple(out)


@contextmanager
def mock_raw_report_endpoints(base_url):
    """
    Mock API report endpoints under base_url to enable testing
    of the report generation task run via the dashboard. Requires
    requests_mock>=1.8.0. Catches all report endpoints, but if
    report generation requires POSTing to other endpoints in the
    future, they will need to be added here.
    """
    import requests_mock
    value_dict = {}  # for raw processed values

    def post_value_callback(request, context):
        context.status_code = 200
        rjson = request.json()
        id_ = rjson['object_id']
        value_dict[id_] = rjson['processed_values']
        return id_

    raw_dict = {}  # for the raw reports

    def post_raw_callback(request, context):
        context.status_code = 200
        raw_dict.update(request.json())

    report_dict = {}  # for new and full reports

    def post_report_callback(request, context):
        context.status_code = 200
        report_dict.update(request.json())
        return 'no_id'

    def get_report_callback(request, context):
        context.status_code = 200
        out = report_dict.copy()
        if raw_dict:
            out['raw_report'] = raw_dict
            out['values'] = [
                {'id': k, 'processed_values': v}
                for k, v in value_dict.items()]
        out['status'] = 'complete'
        return out

    with requests_mock.Mocker(real_http=True) as m:
        value_re = re.compile(f'{base_url}/reports/.*/values')
        raw_re = re.compile(f'{base_url}/reports/.*/raw')
        m.register_uri('GET', re.compile(
            f'{base_url}/reports/.*'),
                       json=get_report_callback)
        m.register_uri('POST', re.compile(
            f'{base_url}/reports/*'),
                       text=post_report_callback)
        m.register_uri('POST', re.compile(
            f'{base_url}/reports/.*/status/.*'))
        m.register_uri('POST', value_re, text=post_value_callback)
        m.register_uri('POST', raw_re, json=post_raw_callback)
        yield
