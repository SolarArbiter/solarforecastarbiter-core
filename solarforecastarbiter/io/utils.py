# coding: utf-8

"""Collection of Functions to convert API responses into python objects
and vice versa.
"""
import pandas as pd


def _dataframe_to_json(payload_df):
    payload_df.index.name = 'timestamp'
    return (
        '{"values":' +
        payload_df.tz_convert("UTC").reset_index().to_json(
            orient="records", date_format='iso', date_unit='s')
        + '}')


def observation_df_to_json_payload(
        observation_df,
        variable_label,
        quality_flag_label=None,
        default_quality_flag=None):
    """Extracts a variable from an observation DataFrame and formats it
    into a JSON payload for posting to the Solar Forecast Arbiter API.

    Parameters
    ----------
    observation_df : DataFrame
        Dataframe of observation data. Must contain a tz-aware DateTimeIndex
        and a column of variable values. May contain a column of data quality
        flags.
    variable_label : string
        Label of column containing the observation data.
    quality_flag_label : string
        Label of column containing quality flag integer. Defaults to None,
        in which case a KeyError is raised unless default_quality_flag is set.
    default_quality_flag : int
        If quality_flag_label is not defined, the quality flag for each row is
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
              “quality_flag”: 0, # 0 or 1. 1 indicates data is questionable.
            },...
          ]
        }

    Raises
    ------
    KeyError
       When quality_flag_label and default_quality_flag are not set
    """
    if quality_flag_label is not None:
        payload_df = observation_df[
            [variable_label, quality_flag_label]].rename(
                columns={variable_label: 'value',
                         quality_flag_label: 'quality_flag'})
    elif default_quality_flag is not None:
        payload_df = observation_df[[variable_label]].rename(
            columns={variable_label: 'value'})
        payload_df['quality_flag'] = int(default_quality_flag)
    else:
        raise KeyError(
            'Either quality_flag_label or default_quality_flag need to be set')
    return _dataframe_to_json(payload_df)


def forecast_object_to_json(forecast_obj, column_label=None):
    """
    Converts a forecast Series or DataFrame to JSON to post to the
    SolarForecastArbiter API.

    Parameters
    ----------
    forecast_obj : pandas.Series or pandas.DataFrame
        The series or dataframe that contains the forecast values with a
        datetime index.
    column_label : string
        If forecast_obj is a DataFrame, use this column as the forecast values.

    Returns
    -------
    string
        The JSON encoded forecast values dict
    """
    if isinstance(forecast_obj, pd.Series):
        forecast_df = forecast_obj.to_frame('value')
        column_label = 'value'
    elif isinstance(forecast_obj, pd.DataFrame):
        forecast_df = forecast_obj
    else:
        raise TypeError('forecast_obj must be a pandas Series or DataFrame')

    payload_df = forecast_df[[column_label]].rename(
        columns={column_label: 'value'})
    return _dataframe_to_json(payload_df)


def _json_to_dataframe(json_payload):
    # in the future, might worry about reading the response in chunks
    # to stream the data and avoid having it all in memory at once,
    # but 30 days of 1 minute data is probably ~4 MB of text. A better
    # approach would probably be to switch to a binary format.
    vals = json_payload['values']
    if len(vals) == 0:
        df = pd.DataFrame([], columns=['value', 'quality_flag'],
                          index=pd.DatetimeIndex(
                              start='now', freq='1min', periods=0,
                              name='timestamp'))
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
