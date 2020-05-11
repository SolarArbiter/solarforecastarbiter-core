"""Functions to read NREL PVDAQ data.
"""

# Code originally written by Bennet Meyers (@bmeyers), Stanford, SLAC in
# https://github.com/pvlib/pvlib-python/pull/664
# Adapated by Will Holmgren (@wholmgren), University of Arizona

import json
from io import StringIO

import requests
import pandas as pd


# consider adding an auth=(username, password) kwarg (default None) to
# support private data queries

def get_pvdaq_metadata(system_id, api_key):
    """Query PV system metadata from NREL's PVDAQ data service.

    Parameters
    ----------
    system_id: int
        The system ID corresponding to the site that data should be
        queried from.

    api_key: string
        Your NREL API key (https://developer.nrel.gov/docs/api-key/)

    Returns
    -------
    list of dict
    """

    params = {'system_id': system_id, 'api_key': api_key}
    sites_url = 'https://developer.nrel.gov/api/pvdaq/v3/sites.json'
    r = requests.get(sites_url, params=params)
    r.raise_for_status()
    outputs = json.loads(r.content)['outputs']
    return outputs


def get_pvdaq_data(system_id, year, api_key='DEMO_KEY'):
    """Query PV system data from NREL's PVDAQ data service:

    https://maps.nrel.gov/pvdaq/

    This function uses the annual raw data file API, which is the most
    efficient way of accessing multi-year, sub-hourly time series data.

    Parameters
    ----------
    system_id: int
        The system ID corresponding to the site that data should be
        queried from.

    year: int or list of ints
        Either the year to request or the list of years to request.
        Multiple years will be concatenated into a single DataFrame.

    api_key: string
        Your NREL API key (https://developer.nrel.gov/docs/api-key/)

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the time series data from the
        PVDAQ service over the years requested. Times are typically
        in local time.

    Notes
    -----
    The PVDAQ metadata contains a key "available_years" that is a useful
    value for the *year* argument.
    """

    try:
        year = int(year)
    except TypeError:
        year = [int(yr) for yr in year]
    else:
        year = [year]

    # Each year must queries separately, so iterate over the years and
    # generate a list of dataframes.
    # Consider putting this loop in its own private function with
    # try / except / try again pattern for network issues and NREL API
    # throttling
    df_list = []
    for yr in year:
        params = {
            'api_key': api_key,
            'system_id': system_id,
            'year': yr
        }
        base_url = 'https://developer.nrel.gov/api/pvdaq/v3/data_file'
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df_list.append(df)

    # concatenate the list of yearly DataFrames
    df = pd.concat(df_list, axis=0, sort=True)
    df['Date-Time'] = pd.to_datetime(df['Date-Time'])
    df.set_index('Date-Time', inplace=True)
    return df
