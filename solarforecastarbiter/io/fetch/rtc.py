"""Collection of code for requesting and parsing Data from the DOE
RTC pv-dashboard hosted by Sandia National Laboratories
"""
import logging


import pandas as pd
import requests

DOE_RTC_API_URL = "https://pv-dashboard.sandia.gov/api/v1.0/location/{location}/data/{data_type}/start/{start}/end/{end}/key/{api_key}" # NOQA
logger = logging.getLogger(__name__)


def request_doe_rtc_data(location, data_type, start, end, api_key):
    """Makes a request to DOE RTC pv dashboard with the provided parameters.

    Parameters
    ----------
    location: string
        Name of the DOE RTC location.
    data_type: string
        'system' or 'weather'
    api_key: string
        The Api key for accessing the RTC pv dashboard API.
    start: datetime
        Beginning of the period for which to request data.
    end: datetime
        End of the period for which to request data.

    Returns
    -------
    DataFrame
        DataFrame parsed from the json response.
    """
    request_url = DOE_RTC_API_URL.format(
        location=location,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        data_type=data_type,
        api_key=api_key)
    r = requests.get(request_url)
    resp = r.json()
    if 'access' in resp and resp['access'] == 'denied':
        raise ValueError('Invalid DOE RTC API key')
    else:
        return pd.DataFrame(resp)


def fetch_doe_rtc(location, api_key, start, end):
    """
    Requests and concatenates data from the DOE RTC pv dashboard API
    into a single dataframe.

    Parameters
    ----------
    location: string
        Name of the DOE RTC location.
    api_key: string
        The Api key for accessing the DOE RTC API.
    start: datetime
        Beginning of the period for which to request data.
    end: datetime
        End of the period for which to request data.

    Returns
    -------
    pandas.DataFrame
        With data from start to end. Index is a datetime-index NOT localized
        but in the timezone for the location
    """
    data_types = ['system', 'weather']
    dfs = []
    for data_type in data_types:
        df = request_doe_rtc_data(location, data_type, start, end, api_key)
        if df.empty:
            continue
        df.index = pd.to_datetime(df['TmStamp'], unit='ms', utc=False)
        df = df.drop('TmStamp', axis=1)
        # Append the datatype to the end of AmbientTemp so we can differentiate
        # system from weather temperatures
        df = df.rename(columns={'AmbientTemp': f'AmbientTemp_{data_type}'})
        dfs.append(df)
    try:
        data = pd.concat(dfs, axis=1)
    except ValueError:
        # empty data
        return pd.DataFrame()
    return data
