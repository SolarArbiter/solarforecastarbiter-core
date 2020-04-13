from functools import partial
import logging
import os


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.io.fetch import arm
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


logger = logging.getLogger('reference_data')


# ARM data streams include 'met' for meteorological sites and 'qcrad' for
# irradiance data.
DOE_ARM_SITE_VARIABLES = {
    'qcrad': arm.IRRAD_VARIABLES,
    'met': arm.MET_VARIABLES,
}

DOE_ARM_VARIABLE_MAP = {
    'down_short_hemisp': 'ghi',
    'short_direct_normal': 'dni',
    'down_short_diffuse_hemisp': 'dhi',
    'temp_mean': 'air_temperature',
    'rh_mean': 'relative_humidity',
    'wspd_arith_mean': 'wind_speed',
}


def _determine_site_vars(datastream):
    """Returns a list of variables available based on datastream name.

    Parameters
    ----------
    datastream: str
        datastream field of the site. This should be found in the
        `extra_parameters` field as `network_api_id`

    Returns
    -------
    list of str
        The variable names that can be found in the file.
    """
    available = []
    for stream_type, arm_vars in DOE_ARM_SITE_VARIABLES.items():
        if stream_type in datastream:
            available = available + arm_vars
    return available


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in
    the matched DOE_ARM_VARIABLE_MAP.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Observations.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.error(f'Failed to initialize observations  for {site.name} '
                     'extra parameters could not be loaded.')
        return
    site_arm_vars = _determine_site_vars(site_extra_params['network_api_id'])
    site_sfa_vars = [DOE_ARM_VARIABLE_MAP[v] for v in site_arm_vars]
    for sfa_var in site_sfa_vars:
        logger.info(f'Creating {sfa_var} at {site.name}')
        try:
            common.create_observation(
                api, site, sfa_var)
        except HTTPError as e:
            logger.error(f'Could not create Observation for "{sfa_var}" '
                         f'at DOE ARM site {site.name}')
            logger.debug(f'Error: {e.response.text}')


def initialize_site_forecasts(api, site):
    """
    Create a forecast for each variable at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.error('Failed to initialize reference forecasts for '
                     f'{site.name} extra parameters could not be loaded.')
        return
    site_vars = _determine_site_vars(site_extra_params['network_api_id'])
    common.create_forecasts(api, site, site_vars,
                            default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end, *, doe_arm_user_id, doe_arm_api_key):
    """Retrieve observation data for a DOE ARM site between start and end.

    Parameters
    ----------
    api : io.APISession
        Unused but conforms to common.update_site_observations call
    site : datamodel.Site
        Site object with the appropriate metadata.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    doe_arm_user_id : str
        User ID to access the DOE ARM api.
    doe_arm_api_key : str
        API key to access the DOE ARM api.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        return pd.DataFrame()
    doe_arm_datastream = site_extra_params['network_api_id']
    site_vars = _determine_site_vars(doe_arm_datastream)
    obs_df = arm.fetch_arm(
        doe_arm_user_id, doe_arm_api_key, doe_arm_datastream, site_vars,
        start.tz_convert(site.timezone), end.tz_convert(site.timezone))
    if obs_df.empty:
        logger.warning(f'Data for site {site.name} contained no '
                       f'entries from {start} to {end}.')
        return pd.DataFrame()
    obs_df = obs_df.rename(columns=DOE_ARM_VARIABLE_MAP).tz_localize(
        site.timezone)
    return obs_df


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to a list of DOE ARM Observations
    from start to end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list of solarforecastarbiter.datamodel.Site
        List of all reference sites as Objects
    observations: list of solarforecastarbiter.datamodel.Observation
        List of all reference observations.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    doe_arm_api_key = os.getenv('DOE_ARM_API_KEY')
    if doe_arm_api_key is None:
        raise KeyError('"DOE_ARM_API_KEY" environment variable must be '
                       'set to update DOE ARM observation data.')
    doe_arm_user_id = os.getenv('DOE_ARM_USER_ID')
    if doe_arm_user_id is None:
        raise KeyError('"DOE_ARM_USER_ID" environment variable must be '
                       'set to update DOE ARM observation data.')

    doe_arm_sites = common.filter_by_networks(sites, 'DOE ARM')
    for site in doe_arm_sites:
        common.update_site_observations(
            api, partial(fetch, doe_arm_user_id=doe_arm_user_id,
                         doe_arm_api_key=doe_arm_api_key),
            site, observations, start, end)
