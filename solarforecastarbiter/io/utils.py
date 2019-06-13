# coding: utf-8

"""Collection of Functions to convert API responses into python objects
and vice versa.
"""
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
        endpoint. An object in the following format:
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
                          index=pd.DatetimeIndex([], name='timestamp'))
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
    """
    df = _json_to_dataframe(json_payload)
    return df[['value', 'quality_flag']]


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
       With a tz-aware DatetimeIndex
    """

    df = _json_to_dataframe(json_payload)
    return df['value']


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
    """

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
       The data with a DatetimeIndex
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
       If an invalid interval_label is given
    """
    start, end = adjust_start_end_for_interval_label(interval_label, start,
                                                     end)
    data.sort_index(axis=0, inplace=True)
    return data.loc[start:end]


class HiddenToken:
    """
    Obscure the representation of the input string `token` to avoid saving
    or displaying access tokens in logs.
    """
    def __init__(self, token):
        self.token = str(token)  # make sure it isn't a localproxy

    def __repr__(self):
        return '****ACCESS*TOKEN****'
