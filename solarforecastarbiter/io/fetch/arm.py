"""Collection of code for requesting and parsing ARM data.
Documentation for the ARM Live Data Web Service can be found
here: https://adc.arm.gov/armlive/
"""
import json
import logging
import netCDF4
import pandas as pd
import requests
import time


logger = logging.getLogger(__name__)

ARM_FILES_LIST_URL = 'https://adc.arm.gov/armlive/data/query'
ARM_FILES_DOWNLOAD_URL = 'https://adc.arm.gov/armlive/data/saveData'


# These lists are the commonly available irradiance and meteorological
# variables found in ARM data. Users can import and pass these to fetch_arm
# to parse out these variables.
IRRAD_VARIABLES = ['down_short_hemisp', 'down_short_diffuse_hemisp',
                   'short_direct_normal']
MET_VARIABLES = ['temp_mean', 'rh_mean', 'wspd_arith_mean']


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
    return response['files']


def request_arm_file(user_id, api_key, filename, retries=5):
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
    retries: int
        Number of attempts remaining to successfully request data after
        ChunkedEncodingError.

    Returns
    -------
    stream
        The API response in the form of a stream to be consumed by
        netCDF4.Dataset().

    Raises
    ------
    request.exceptions.ChunkedEncodingError
        Reraises this error when all retries are exhausted.
    """
    params = {'user': f'{user_id}:{api_key}',
              'file': filename}

    try:
        request = requests.get(ARM_FILES_DOWNLOAD_URL, params=params)
    except requests.exceptions.ChunkedEncodingError:
        if retries > 0:
            logger.debug(f'Retrying DOE ARM file {filename}: {retries}'
                         'remaining.')
            time.sleep((5 - retries) * 0.1)
            return request_arm_file(user_id, api_key, filename, retries-1)
        else:
            logger.warning(f'Requesting ARM file {filename} failed')
            raise
    nc_data = request.content
    return nc_data


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
                              memory=nc_data)
    return nc_file


def extract_arm_variables(nc_file, variables):
    """Extracts variables and datetime index from an ARM netcdf.

    Parameters
    ----------
    nc_file: netCDF4 Dataset
        The ARM file read into a Dataset.
    variables: list
        List of string variable names to parse from the files.

    Returns
    -------
    DataFrame
        A pandas DataFrame with a column for each requested variable
        found in the ARM netcdf file, indexed by timestamp in UTC. If
        none of the requested variables are found, an empty DataFrame
        is returned.
    """
    var_data = {}
    for var in variables:
        try:
            var_data[var] = nc_file[var][:]
        except IndexError:
            continue
    if var_data:
        base_time = nc_file['base_time'][0].data.item()
        delta_time = nc_file['time'][:]
        times = pd.to_datetime(base_time + delta_time, unit='s', utc=True)
        return pd.DataFrame(index=times, data=var_data)
    else:
        return pd.DataFrame()


def fetch_arm(user_id, api_key, datastream, variables, start, end):
    """Gets data from ARM API and concatenates requested datastreams into
    a single Pandas Dataframe.

    Parameters
    ----------
    user_id: string
        ARM user id.
    api_key: string
        ARM live API access token.
    datastream: string
        The datastream to request.
    variables
        List of variables to parse from the datastream.
    start: datetime
        The start of the interval to request data for.
    end: datetime
        The end of the interval to request date for.

    Returns
    -------
    DataFrame
        A DataFrame containing all of the available variables over the
        requested period.

    Notes
    -----
    Elements of the variable list that are not found in the datastream
    are ignored, this is to allow iteration over many datastreams without
    knowing their exact contents. If none of the requested variables are
    found, an empty DataFrame will be returned. Users should verify the
    contents of the return value before use.

    Example
    -------
    A user requesting data for the variables 'down_short_hemisp' and
    'short_direct_normal' from the datastream 'sgpqcrad1longC1.c1' for
    the days between 2019-02-27 and 2019-03-01 could expect the
    following DataFrame.

    .. code::

                                  down_short_hemisp  short_direct_normal
       2019-02-27 00:00:00+00:00           7.182889            -1.399250
       2019-02-27 00:01:00+00:00           6.943601            -1.317890
       2019-02-27 00:02:00+00:00           6.686488            -1.235140
       ...
       2019-03-01 23:57:00+00:00           6.943601            -1.317890
       2019-03-01 23:58:00+00:00           6.686488            -1.235140
       2019-03-01 23:59:00+00:00           6.395981            -1.226730
    """
    datastream_dfs = []
    filenames = list_arm_filenames(user_id, api_key, datastream, start, end)
    for filename in filenames:
        try:
            nc_file = retrieve_arm_dataset(user_id, api_key, filename)
        except requests.exceptions.ChunkedEncodingError:
            logger.error(f'Request failed for DOE ARM file {filename}')
        else:
            datastream_df = extract_arm_variables(nc_file, variables)
            datastream_dfs.append(datastream_df)
    if len(datastream_dfs) > 0:
        new_data = pd.concat(datastream_dfs)
        return new_data
    else:
        return pd.DataFrame()
