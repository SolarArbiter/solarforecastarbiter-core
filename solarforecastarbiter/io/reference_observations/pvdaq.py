import json
import logging
from pkg_resources import resource_filename, Requirement


import pandas as pd


from solarforecastarbiter.datamodel import Observation
from solarforecastarbiter.io.fetch import pvdaq
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


logger = logging.getLogger('reference_data')


DEFAULT_SITEFILE = resource_filename(
    Requirement.parse('solarforecastarbiter'),
    'solarforecastarbiter/io/reference_observations/'
    'pvdaq_reference_sites.json')


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
    site_api_id = extra_params['network_api_id']
    obs_metadata = json.load(DEFAULT_SITEFILE)['observations']
    site_obs_metadata = [
        obs for obs in obs_metadata if
        obs['extra_parameters']['network_api_id'] == site_api_id]
    for obs in site_obs_metadata:
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
    site_api_id = extra_params['network_api_id']
    obs_metadata = json.load(DEFAULT_SITEFILE)['observations']
    obs_vars = [
        obs.variable for obs in obs_metadata if
        obs['extra_parameters']['network_api_id'] == site_api_id]
    common.create_forecasts(
        api, site, obs_vars, default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end):
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
            site_extra_params['network_api_id'], years)
    except Exception:
        # Not yet sure what kind of errors we might hit in production
        logger.warning(f'Could not retrieve data for site {site.name}'
                       f' between {start} and {end}.')
        return pd.DataFrame()
    obs_df = obs_df.tz_localize(site.timezone)
    return obs_df


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to all PVDAQ observations from
    start to end.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list
        List of all reference sites as Objects
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    pvdaq_sites = common.filter_by_networks(sites, ['PVDAQ'])
    for site in pvdaq_sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end)
