"""Initialize site obs/forecasts and fetch/update obs for BSRN sites."""

import logging

import pandas as pd

from solarforecastarbiter.io.fetch import bsrn
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


logger = logging.getLogger('reference_data')
bsrn_variables = ('ghi', 'dni', 'dhi', 'air_temperature', 'relative_humidity')


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in bsrn_variables.

    Parameters
    ----------
    site : datamodel.Site
        The site object for which to create Observations.
    """
    for variable in bsrn_variables:
        common.create_observation(api, site, variable)


def initialize_site_forecasts(api, site):
    """
    Create forecasts for each variable in bsrn_variables at the site

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    common.create_forecasts(api, site, bsrn_variables,
                            default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end):
    """Retrieve observation data for a BSRN site between start and end.

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
    if site.name != 'WRMC BSRN NASA Langley Research Center':
        raise NotImplementedError('Fetching BSRN data is currently only '
                                  'supported for NASA Langley site')
    try:
        data = bsrn.read_bsrn_from_nasa_larc(start, end)
    except Exception:
        # Not yet sure what kind of errors we might hit in production
        logger.warning(f'Could not retrieve data for site {site.name}'
                       f' between {start} and {end}.')
        return pd.DataFrame()
    data = data.rename(columns={'temp_air': 'air_temperature'})
    return data


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to all BSRN observations from
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
    """
    bsrn_sites = common.filter_by_networks(sites, ['WRMC BSRN'])
    for site in bsrn_sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end, gaps_only=gaps_only)
