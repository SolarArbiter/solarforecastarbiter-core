# coding: utf-8

"""Collection of Functions to convert API responses into python objects
and vice versa.
"""
import json
import pandas as pd


def observation_df_to_json_payload(
        observation_df,
        variable_label,
        questionable_label=None):
    """Extracts a variable from an observation DataFrame and formats it
    into a JSON payload for posting to the Solar Forecast Arbiter API.

    Parameters
    ----------
    observation_df: DataFrame
        Dataframe of observation data. Must contain a tz-aware DateTimeIndex
        and a column of variable values. May contain a column of data quality
        flags (0 or 1).
    variable_label: string
        Label of column containing the observation data.
    questionable_label: string
        Label of column containing questionable flag(0 or 1). Defaults to None,
        in which case the questionable field of each record in the payload will
        be set to 1.

    Returns
    -------
    string
        Solar Forecast Arbiter JSON payload for posting to the observation
        endpoint. An object in the following format:
        {
          'values': [
            {
              “timestamp”: “2018-11-22T12:01:48Z”, # ISO 8601 datetime in UTC
              “value”: 10.23, # floating point value of observation
              “questionable”: 0, # 0 or 1. 1 indicates data is questionable.
            },...
          ]
        }
    """
    payload_df = pd.DataFrame()
    timestamps = observation_df.index
    timestamps = timestamps.tz_convert('UTC')
    timestamps = timestamps.strftime('%Y-%m-%dT%H:%M:%SZ')
    payload_df['value'] = observation_df[variable_label]
    payload_df['timestamp'] = timestamps

    if questionable_label is None:
        payload_df['questionable'] = 0
    else:
        payload_df['questionable'] = observation_df[questionable_label]
    return json.dumps({'values': payload_df.to_dict(orient='records')})


def _json_to_dataframe(json_payload):
    # in the future, might worry about reading the response in chunks
    # to stream the data and avoid having it all in memory at once,
    # but 30 days of 1 minute data is probably ~4 MB of text. A better
    # approach would probably be to switch to a binary format.
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
