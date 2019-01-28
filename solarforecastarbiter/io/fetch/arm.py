"""Collection of code for requesting and parsing ARM data.
"""
import json
import netCDF4
import numpy as np
import pandas as pd
import requests


ARM_FILES_LIST_URL = 'https://adc.arm.gov/armlive/data/query'


def format_date(date_object):
    return date_object.strftime('%Y-%m-%d')


def request_arm_file_list(user_id, api_key, datastream, start, end):
    """Make an http request to the ARM live API for filenames between start
    and end.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    datastream: string
        Name of the datastream to query for files.
    start: datetime
        Beginning of period for which to request data.
    end: datetime
        End of period for which to request data.

    Returns
    -------
    dict
        The json response parsed into a dictionary.
    """
    params = {'user': f'{user_id}:{api_key}',
              'ds': datastream,
              'start': format_date(start),
              'end': format_date(end),
              'wt': 'json'}
    response = requests.get(ARM_FILES_LIST_URL, params=params)
    return json.loads(response.text)


def list_arm_filenames(user_id, api_key, datastream, start, end):
    """Get a list of filenames from ARM for the given datastream between
    start and end.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    datastream: string
        Name of the datastream to query for files.
    start: datetime
        Beginning of period for which to request data.
    end: datetime
        End of period for which to request data.

    Returns
    -------
    list
        List of filenames as strings.
    """
    response = request_arm_file_list(user_id, api_key,
                                     datastream, start, end)
    if 'files' in response and len(response['files']) > 0:
        return response['files']
    return []


def request_arm_file(user_id, api_key, filename):
    """Get a file from ARM live in the form of a stream so that the python netCDF4
    module can read it.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    filename: string
        Filename to request

    Returns
    -------
    stream
        The API response in the form of a stream to be consumed by
        netCDF4.Dataset().

    Notes
    -----
    The stream handle must be closed by the user.
    """
    ARM_FILES_LIST_URL = 'https://adc.arm.gov/armlive/data/saveData'
    params = {'user': f'{user_id}:{api_key}',
              'file': filename}
    return requests.get(ARM_FILES_LIST_URL, params=params, stream=True)


def retrieve_arm_dataset(user_id, api_key, filename):
    """Request a file from the ARM Live API and return a netCDF4 Dataset.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    filename: string
        Filename to request

    Returns
    -------
    netCDF4.Dataset
        Dataset of the API response.
    """
    nc_data = request_arm_file(user_id, api_key, filename)
    nc_file = netCDF4.Dataset(f'/tmp/{filename}', mode='r',
                              memory=nc_data.content)
    nc_data.close()
    return nc_file


def extract_arm_variable(nc_file, var_name):
    """Returns a time series of the variable.

    Parameters
    ----------
    nc_file: netCDF4 Dataset
        The ARM file read into a Dataset.
    var_name: string
        The var label to extract

    Returns
    -------
    DataFrame
        A DataFrame with a DatetimeIndex and the requested variable data.
    """
    base_time = np.asscalar(nc_file['base_time'][0].data)
    delta_time = nc_file['time'][:]
    times = pd.to_datetime(base_time + delta_time, unit='s')
    var_data = nc_file[var_name][:]
    var_df = pd.DataFrame(index=times, data={var_name: var_data})
    return var_df


def fetch_arm(user_id, api_key, datastreams, start, end):
    """Gets data from ARM API and concatenates requested datastreams into
    a single Pandas Dataframe.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    datastreans: dictionary
        Dictionary of 'datastream': 'datastream type'
    start: datetime
        The start of the interval to request data for.
    end: datetime
        The end of the interval to request date for.

    Returns
    -------
    DataFrame
        A DataFrame containing all of the available variables over the
        requested period.
    """
    site_dfs = []
    for ds, ds_type in datastreams.items():
        fns = list_arm_filenames(user_id, api_key, ds, start, end)
        var_dfs = []
        for fn in fns:
            file = retrieve_arm_dataset(user_id, api_key, fn)
            # TODO: allow users to define ds_type and applicable variables
            if ds_type == 'irrad':
                var = extract_arm_variable(file, 'down_short_diffuse_hemisp')
            elif ds_type == 'weather':
                var = extract_arm_variable(file, 'temp_mean')
            var_dfs.append(var)
            file.close()
        site_dfs.append(pd.concat(var_dfs, sort=False))
    new_data = pd.concat(site_dfs, axis=1)
    return new_data
