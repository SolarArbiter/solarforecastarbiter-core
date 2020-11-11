import os
import logging
from functools import partial
import pandas as pd

from solarforecastarbiter.io.fetch import eia
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)

from requests.exceptions import HTTPError

logger = logging.getLogger('reference_data')


def initialize_site_observations(api, site):
    """Creates an observation at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        API Session object, authenticated for the Reference user.
    site : solarforecastarbiter.datamodel.Site
        The site object for which to create the Observations.

    Notes
    -----
    Currently only creates observations for net load [MW]
    (`f"EBA.{eia_site_id}.D.H"`), but EIA contains other variables that may be
    incorporated later (e.g. solar generation:
    `f"EBA.{eia_site_id}.NG.SUN.H"`).

    """

    sfa_var = "net_load"
    logger.info(f'Creating {sfa_var} at {site.name}')
    try:
        common.create_observation(api, site, sfa_var)
    except HTTPError as e:
        logger.error(f'Could not create Observation for "{sfa_var}" '
                     f'at EIA site {site.name}')
        logger.debug(f'Error: {e.response.text}')


def initialize_site_forecasts(api, site):
    """Creates a forecast at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        API Session object, authenticated for the Reference user.
    site : solarforecastarbiter.datamodel.Site
        The site object for which to create the Observations.

    """
    common.create_forecasts(
        api, site, ["net_load"],
        default_forecasts.TEMPLATE_NETLOAD_PERSISTENCE_FORECASTS)


def fetch(api, site, start, end, *, eia_api_key):
    """Retrieve observation data for a EIA site between start and end.

    Parameters
    ----------
    api : solarforecastarbiter.io.APISession
        Unused but conforms to common.update_site_observations call
    site : solarforecastarbiter.datamodel.Site
        Site object with the appropriate metadata.
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    eia_api_key : str
        API key for api.eia.gov

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data as a single DataFrame.

    Notes
    -----
    Currently only fetches observations for net load [MW]
    (`f"EBA.{eia_site_id}.D.H"`), but EIA contains other variables that may be
    incorporated later (e.g. solar generation:
    `f"EBA.{eia_site_id}.NG.SUN.H"`).

    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        return pd.DataFrame()
    eia_site_id = site_extra_params['network_api_id']
    series_id = f"EBA.{eia_site_id}.D.H"   # hourly net load (demand)
    obs_df = eia.get_eia_data(
        series_id, eia_api_key,
        start,
        end
    )
    if obs_df.empty:
        logger.warning(f'Data for site {site.name} contained no '
                       f'entries from {start} to {end}.')
        return pd.DataFrame()
    obs_df = obs_df.rename(columns={"value": "net_load"})
    return obs_df


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Retrieve data from the network, and then format and post it to each
    observation at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list of solarforecastarbiter.datamodel.Site
        List of all reference sites.
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
        If EIA_API_KEY environmental variable is not set.

    """

    eia_api_key = os.getenv("EIA_API_KEY")
    if eia_api_key is None:
        raise KeyError('"EIA_API_KEY" environment variable must be '
                       'set to update EIA observation data.')

    eia_sites = common.filter_by_networks(sites, ['EIA'])
    for site in eia_sites:
        common.update_site_observations(
            api, partial(fetch, eia_api_key=eia_api_key),
            site, observations, start, end, gaps_only=gaps_only)
