import logging
import json
from urllib import error
from pkg_resources import resource_filename, Requirement


import pandas as pd
from pvlib import iotools
from requests.exceptions import HTTPError


from solarforecastarbiter.datamodel import Observation, SolarPowerPlant
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


DEFAULT_SITEFILE = resource_filename(
    Requirement.parse('solarforecastarbiter'),
    'solarforecastarbiter/io/reference_observations/'
    'srml_reference_sites.json')


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


def adjust_site_parameters(site):
    """Inserts modeling parameters for sites with pv measurments

    Parameters
    ----------
    site: dict

    Returns
    -------
    dict
        Copy of inputs plus a new key 'modeling_parameters'.
    """
    return common.apply_json_site_parameters(DEFAULT_SITEFILE, site)


def request_data(site, year, month):
    """Makes a request for each file type until successful or we
    run out of filetypes.

    Parameters
    ----------
    site: :py:class:`solarforecastarbiter.datamodel.Site`
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
    logger.warning(f'Could not retrieve data for site {site.name} on '
                   f'{year}/{month}.')


def fetch(api, site, start, end):
    """Retrieve observation data for a srml site between start and end.

    Parameters
    ----------
    api : :py:class:`solarforecastarbiter.io.api.APISession`
        An APISession with a valid JWT for accessing the Reference Data
        user.
    site : :py:class:`solarforecastarbiter.datamodel.Site`
        Site object with the appropriate metadata.
    start : datetime
        The beginning of the period to request data for. Must include timezone.
    end : datetime
        The end of the period to request data for. Must include timezone.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.

    Raises
    ------
    TypeError
        If start and end have different timezones, or if they do not include a
        timezone.
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
    power_columns = [col for col in var_columns
                     if col.startswith('5')]
    # adjust power from watts to megawatts
    for column in power_columns:
        all_period_data[column] = all_period_data[column] / 1000000
    all_period_data = all_period_data.loc[start:end, var_columns]

    # remove possible trailing NaNs, it is necessary to do this after slicing
    # because SRML data has nighttime data prefilled with 0s through the end of
    # the month. This may not be effective if a given site has more than a 24
    # hour lag, which will cause last_valid_index to return the latest
    # timestamp just before sunrise, but will suffice for the typical lag on
    # the order of hours.
    all_period_data = all_period_data[:all_period_data.last_valid_index()]

    return all_period_data


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in
    an SRML site's file.

    Parameters
    ----------
    api: :py:class:`solarforecastarbiter.io.api.APISession`

    site : :py:class:`solarforecastarbiter.datamodel.Site
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
    # Request ~month old data at initialization to ensure we get a response.
    start = pd.Timestamp.utcnow() - pd.Timedelta('30 days')
    end = start
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
            matches = [col for col in site_df.columns
                       if col.startswith(variable)]
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
        with open(DEFAULT_SITEFILE) as fp:
            obs_metadata = json.load(fp)['observations']
        for obs in obs_metadata:
            obs_site_extra_params = json.loads(obs['site']['extra_parameters'])
            if obs_site_extra_params['network_api_id'] == extra_params[
                    'network_api_id']:
                obs['site'] = site
                observation = Observation.from_dict(obs)
                common.check_and_post_observation(api, observation)


def initialize_site_forecasts(api, site):
    """
    Create a forecasts for each variable measured at the site

    Parameters
    ----------
    api : :py:class:`solarforecastarbiter.io.api.APISession`
        An active Reference user session.
    site : :py:class:`solarforecastarbiter.datamodel.Site`
        The site object for which to create Forecasts.
    """
    variables = list(srml_variable_map.values())
    if isinstance(site, SolarPowerPlant):
        variables += ['ac_power', 'dc_power']
    common.create_forecasts(
        api, site, variables,
        default_forecasts.TEMPLATE_FORECASTS)


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to a list of SRML Observations
    from start to end.

    api : :py:class:`solarforecastarbiter.io.api.APISession`
        An active Reference user session.
    sites: list of :py:class:`solarforecastarbiter.datamodel.Site`
        List of all reference sites as Objects
    observations: list of :py:class:`solarforecastarbiter.datamodel.Observation`
        List of all reference observations as Objects
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    gaps_only : bool, default False
        If True, only update periods between start and end where there
        are data gaps.
    """  # noqa
    srml_sites = common.filter_by_networks(sites, 'UO SRML')
    for site in srml_sites:
        common.update_site_observations(api, fetch, site, observations,
                                        start, end, gaps_only=gaps_only)
