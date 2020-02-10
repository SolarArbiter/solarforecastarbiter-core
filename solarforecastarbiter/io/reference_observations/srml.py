import logging
from urllib import error


import pandas as pd
from pvlib import iotools
from requests.exceptions import HTTPError


from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


# maps the desired variable names to those returned by pvlib.iotools
srml_variable_map = {
    'ghi_': 'ghi',
    'dni_': 'dni',
    'dhi_': 'dhi',
    'wind_speed_': 'wind_speed',
    'temp_air_': 'air_temperature',
}

# maps SolarForecastArbiter interval_label to the SRML infix which
# designates the time resolution of each file. The list of file types
# is tried in order, so file types starting with 'P' designating
# processed data are listed first, such that if processed data exists
# we retrieve that first.
FILE_TYPE_MAP = {
    1: ['PO', 'RO'],
    5: ['PF', 'RF'],
    15: ['PQ', 'RQ'],
    60: ['PH', 'RH'],
}


logger = logging.getLogger('reference_data')


def request_data(site, year, month):
    """Makes a request for each file type until successful or we
    run out of filetypes.

    Parameters
    ----------
    interval_length: int
        The number of minutes between each timestep in the data. Used
        to lookup filetypes in FILE_TYPE_MAP.
    station: string
        The two character station abbreviation found in filenames.
    year: int
        The year of the data to request.
    month: int
        The month of the data to request.

    Returns
    -------
    DataFrame
        A month of SRML data.
    """
    extra_params = common.decode_extra_parameters(site)
    station_code = extra_params['network_api_abbreviation']
    interval_length = extra_params['observation_interval_length']
    file_types = FILE_TYPE_MAP[interval_length]
    for file_type in file_types:
        # The list file_types are listed with processed data
        # file types first. On a successful retrieval we return
        # the month of data, otherwise we log info and continue
        # until we've exhausted the list.
        try:
            srml_month = iotools.read_srml_month_from_solardat(
                station_code, year, month, file_type)
        except error.URLError:
            logger.warning(f'Could not retrieve {file_type} for SRML data '
                           f'for site {site.name} on {year}/{month} .')
            logger.debug(f'Site abbreviation: {station_code}')
            continue
        except pd.errors.EmptyDataError:
            logger.warning(f'SRML returned an empty file for station '
                           f'{site.name} on {year}/{month}.')
            continue
        else:
            return srml_month


def fetch(api, site, start, end):
    """Retrieve observation data for a srml site between start and end.

    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data
        user.
    site : datamodel.Site
        Site object with the appropriate metadata.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    month_dfs = []
    start_year = start.year
    start_month = start.month
    # Retrieve each month file necessary
    if start.tzinfo != end.tzinfo:
        raise TypeError('start and end cannot have different timezones')
    while start_year * 100 + start_month <= end.year * 100 + end.month:
        logger.info(f'Requesting data for SRML site {site.name}'
                    f' for {start_year}-{start_month}')
        srml_month = request_data(site, start_year, start_month)
        if srml_month is not None:
            month_dfs.append(srml_month)
        start_month += 1
        if start_month > 12:
            start_month = 1
            start_year += 1
    try:
        all_period_data = pd.concat(month_dfs)
    except ValueError:
        logger.warning(f'No data available for site {site.name} '
                       f'from {start} to {end}.')
        return pd.DataFrame()
    var_columns = [col for col in all_period_data.columns
                   if '_flag' not in col]
    all_period_data = all_period_data[var_columns]
    return all_period_data


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in
    an SRML site's file.

    Parameters
    ----------
    api: io.api.APISession

    site : datamodel.Site
        The site object for which to create Observations.

    Notes
    -----
    Since variables are labelled with an integer instrument
    number, Observations are named with their variable and
    instrument number found in the source files.

    e.g. A SRML file contains two columns labelled, 1001, and
    1002. These columns represent GHI at instrument 1 and
    instrument 2 respectively. The `pvlib.iotools` package
    converts these to 'ghi_1' and 'ghi_2' for us. We use these
    labels to differentiate between measurements recorded by
    different instruments.
    """
    start = pd.Timestamp.now()
    end = pd.Timestamp.now()
    try:
        extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.warning('Cannot create reference observations at MIDC site '
                       f'{site.name}, missing required parameters.')
        return
    # use site name without network here to build
    # a name with the original column label rather than
    # the SFA variable
    site_name = common.site_name_no_network(site)
    try:
        site_df = fetch(api, site, start, end)
    except error.HTTPError:
        logger.error('Could not find data to create observations '
                     f'for SRML site {site_name}.')
        return
    else:
        if site_df is None:
            logger.error('Could not find data to create observations '
                         f'for SRML site {site_name}.')
            return
        for variable in srml_variable_map.keys():
            matches = [col for col in site_df.columns if variable in col]
            for match in matches:
                observation_extra_parameters = extra_params.copy()
                observation_extra_parameters.update({
                    'network_data_label': match})
                try:
                    # Here, we pass a name with match instead of variable
                    # to differentiate between multiple observations of
                    # the same variable
                    common.create_observation(
                        api, site, srml_variable_map[variable],
                        name=f'{site_name} {match}',
                        interval_label='beginning',
                        extra_params=observation_extra_parameters)
                except HTTPError as e:
                    logger.error(
                        f'Failed to create {variable} observation at Site '
                        f'{site.name}. Error: {e.response.text}')


def initialize_site_forecasts(api, site):
    """
    Create a forecasts for each variable measured at the site

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    common.create_forecasts(api, site, srml_variable_map.values(),
                            default_forecasts.TEMPLATE_DETERMINISTIC_FORECASTS)


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to a list of SRML Observations
    from start to end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list
        List of all reference sites as Objects
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    srml_sites = common.filter_by_networks(sites, 'UO SRML')
    for site in srml_sites:
        common.update_site_observations(api, fetch, site, observations,
                                        start, end)
