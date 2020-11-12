import logging
from urllib.error import URLError


import pandas as pd
from pvlib import iotools

from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


CRN_URL = 'https://www1.ncdc.noaa.gov/pub/data/uscrn/products/subhourly01/'

crn_variables = ['ghi', 'air_temperature', 'relative_humidity', 'wind_speed']

logger = logging.getLogger('reference_data')


def get_filename(site, year):
    """Get the applicable file name for CRN a site on a given date.
    """
    extra_params = common.decode_extra_parameters(site)
    network_api_id = extra_params['network_api_id']
    filename = f'{year}/CRNS0101-05-{year}-{network_api_id}.txt'
    return CRN_URL + filename


def fetch(api, site, start, end):
    """Requests data for a CRN site containing the requested start,
    end interval.

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
    realtime : bool
        Whether or not to look for realtime data. Note that this data is
        raw, unverified data from the instruments.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    # CRN uses yearly files. To ensure we request the correct years,
    # we make a  request for each year between start and end
    year_dfs = []
    for year in range(start.year, end.year + 1):
        filename = get_filename(site, year)
        logger.info(f'requesting data for {site.name} on {year}')
        logger.debug(f'CRN filename: {filename}')
        try:
            crn_year = iotools.read_crn(filename)
        except URLError:
            logger.warning(f'Could not retrieve CRN data for site '
                           f'{site.name} for year {year}.')
            logger.debug(f'Failed CRN URL: {filename}.')
        else:
            year_dfs.append(crn_year)
    try:
        all_period_data = pd.concat(year_dfs)
    except ValueError:
        logger.warning(f'No data available for site {site.name} '
                       f'from {start} to {end}.')
        return pd.DataFrame()
    all_period_data = all_period_data.rename(
        columns={'temp_air': 'air_temperature'})
    return all_period_data


def initialize_site_observations(api, site):
    """Create an observation for each available variable at the SOLRAD site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active reference user session
    site : solarforecastarbiter.datamodel.Site

    """
    for variable in crn_variables:
        common.create_observation(api, site, variable)


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
    common.create_forecasts(
        api, site, crn_variables,
        default_forecasts.TEMPLATE_DETERMINISTIC_NWP_FORECASTS)


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """ Post new observation data to all reference observations at each
    USCRN site between start and end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list of solarforecastarbiter.datamodel.Site
        List of all reference sites
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
    crn_sites = common.filter_by_networks(sites, 'NOAA USCRN')
    for site in crn_sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end, gaps_only=gaps_only)
