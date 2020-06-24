import os
from functools import partial
import logging

from solarforecastarbiter.io.fetch import eia
from solarforecastarbiter.io.reference_observations import common


logger = logging.getLogger('reference_data')


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in the EIA
    site's file.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        API Session object, authenticated for the Reference user.
    site : solarforecastarbiter.datamodel.Site
        The site object for which to create the Observations.
    """
    pass


def initialize_site_forecasts(api, site):
    """Creates a forecast for each observation at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        API Session object, authenticated for the Reference user.
    site : solarforecastarbiter.datamodel.Site
        The site object for which to create the Observations.

    """
    pass


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
        All of the requested data concatenated into a single DataFrame.
    """
    pass


def update_observation_data(api, sites, observations, start, end):
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

    Raises
    ------
    KeyError

    """

    eia_api_key = os.getenv("EIA_API_KEY")
    if eia_api_key is None:
        raise KeyError('"EIA_API_KEY" environment variable must be '
                       'set to update EIA observation data.')

    eia_sites = common.filter_by_networks(sites, ['EIA'])
    for site in eia_sites:
        common.update_site_observations(
            api, partial(fetch, eia_api_key=eia_api_key),
            site, observations, start, end)
