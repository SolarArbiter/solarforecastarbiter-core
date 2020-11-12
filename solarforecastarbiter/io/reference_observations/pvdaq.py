from functools import partial
import json
import logging
import os
from pkg_resources import resource_filename, Requirement


import pandas as pd
from pytz.exceptions import NonExistentTimeError


from solarforecastarbiter.datamodel import Observation
from solarforecastarbiter.io.fetch import pvdaq
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


logger = logging.getLogger('reference_data')


DEFAULT_SITEFILE = resource_filename(
    Requirement.parse('solarforecastarbiter'),
    'solarforecastarbiter/io/reference_observations/'
    'pvdaq_reference_sites.json')


def adjust_site_parameters(site):
    """Kludge the extra metadata in a json file into the metadata dict
    derived from a csv file.

    Parameters
    ----------
    site: dict

    Returns
    -------
    dict
        Copy of input plus a new key 'modeling_parameters' and more
        metadata in extra_parameters.

    See also
    --------
    solarforecastarbiter.io.reference_observations.site_df_to_dicts
    """
    return common.apply_json_site_parameters(DEFAULT_SITEFILE, site)


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in the PVDAQ
    site's file.

    Parameters
    ----------
    api : io.api.APISession
        API Session object, authenticated for the Reference user.
    site : datamodel.Site
        The site object for which to create the Observations.
    """
    try:
        extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.warning('Cannot create reference observations at PVDAQ site '
                       f'{site.name}, missing required parameters.')
        return
    site_api_id = int(extra_params['network_api_id'])
    with open(DEFAULT_SITEFILE) as fp:
        obs_metadata = json.load(fp)['observations']

    for obs in obs_metadata:
        obs_extra_params = json.loads(obs['extra_parameters'])
        if obs_extra_params['network_api_id'] == site_api_id:
            obs['site'] = site
            observation = Observation.from_dict(obs)
            common.check_and_post_observation(api, observation)


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
    try:
        extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.warning('Cannot create reference observations at PVDAQ site '
                       f'{site.name}, missing required parameters.')
        return
    site_api_id = int(extra_params['network_api_id'])
    with open(DEFAULT_SITEFILE) as fp:
        obs_metadata = json.load(fp)['observations']

    obs_vars = []
    for obs in obs_metadata:
        obs_extra_params = json.loads(obs['extra_parameters'])
        if obs_extra_params['network_api_id'] == site_api_id:
            obs_vars.append(obs['variable'])

    common.create_forecasts(
        api, site, obs_vars, default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end, *, nrel_pvdaq_api_key):
    """Retrieve observation data for a PVDAQ site between start and end.

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
    nrel_pvdaq_api_key : str
        API key for developer.nrel.gov

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        return pd.DataFrame()
    try:
        years = list(range(start.year, end.year + 1))
        obs_df = pvdaq.get_pvdaq_data(
            site_extra_params['network_api_id'], years,
            api_key=nrel_pvdaq_api_key)
    except Exception:
        # Not yet sure what kind of errors we might hit in production
        logger.warning(f'Could not retrieve data for site {site.name}'
                       f' between {start} and {end}.')
        return pd.DataFrame()
    obs_df = _watts_to_mw(obs_df)
    try:
        obs_df = obs_df.tz_localize(site.timezone)
    except NonExistentTimeError as e:
        logger.warning(f'Could not localize data for site {site.name} '
                       f'due to DST issue: {e}')
        return pd.DataFrame()
    return obs_df


def _watts_to_mw(obs_df):
    obs_df_power_mw = obs_df.filter(regex='.*[pP]ower.*') / 1e6
    obs_df[obs_df_power_mw.columns] = obs_df_power_mw
    return obs_df


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to all PVDAQ observations from
    start to end.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list
        List of all reference sites as Objects
    observations: list of solarforecastarbiter.datamodel.Observation
        List of all reference observations.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    gaps_only : bool, default False
        If True, only update periods between start and end where there
        are data gaps.

    Raises
    ------
    KeyError
        If NREL_PVDAQ_API_KEY environmental variable is not set.
        Abuse of KeyError - should probably be ValueError - but kept for
        consistency with other reference_observations modules.
    """
    nrel_pvdaq_api_key = os.getenv('NREL_PVDAQ_API_KEY')
    if nrel_pvdaq_api_key is None:
        raise KeyError('"NREL_PVDAQ_API_KEY" environment variable must be '
                       'set to update PVDAQ observation data.')
    pvdaq_sites = common.filter_by_networks(sites, ['NREL PVDAQ'])
    for site in pvdaq_sites:
        common.update_site_observations(
            api, partial(fetch, nrel_pvdaq_api_key=nrel_pvdaq_api_key),
            site, observations, start, end, gaps_only=gaps_only)
