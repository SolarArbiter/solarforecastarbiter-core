"""Functions for Creating and Updating NOAA SURFRAD related objects
within the SolarForecastArbiter
"""
import logging
from urllib.error import URLError


import pandas as pd


from pvlib import iotools
from solarforecastarbiter.io.utils import observation_df_to_json_payload as obs_to_payload # NOQA
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


# Format strings for the location of a surfrad data file. Expects the following
# variables:
#   abrv: The SURFRAD site's abbreviated name, i.e. Bondville, IL is 'bon'
#   year: Year as a 4 digit number.
#   year_2d: Year as a 0 padded 2 digit  number
#   jday: day of the year as a 3 digit number.
SURFRAD_FTP_DIR = "ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad"
REALTIME_URL = SURFRAD_FTP_DIR + "/realtime/{abrv}/{abrv}{year_2d}{jday}.dat"
ARCHIVE_URL = SURFRAD_FTP_DIR + "/{abrv}/{year}/{abrv}{year_2d}{jday}.dat"

# A list of variables that are available in all SURFRAD files to
# parse and create observations for.
surfrad_variables = ['ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed']

logger = logging.getLogger('reference_data')


def fetch(api, site, start, end, realtime=False):
    """Retrieve observation data for a surfrad site between start and end.

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
    if realtime:
        url_format = REALTIME_URL
    else:
        url_format = ARCHIVE_URL
    # load extra parameters for api arguments.
    extra_params = common.decode_extra_parameters(site)
    abbreviation = extra_params['network_api_abbreviation']
    single_day_dfs = []
    for day in pd.date_range(start, end):
        filename = url_format.format(abrv=abbreviation,
                                     year=day.year,
                                     year_2d=day.strftime('%y'),
                                     jday=day.strftime('%j'))
        logger.info(f'Requesting data for SURFRAD site {site.name}'
                    f' on {day.strftime("%Y%m%d")}.')
        try:
            # Only get dataframe from the returned tuple
            surfrad_day = iotools.read_surfrad(filename)[0]
        except URLError:
            logger.warning(f'Could not retrieve SURFRAD data for site '
                           f'{site.name} on {day.strftime("%Y%m%d")}.')
            logger.debug(f'Failed SURFRAD URL: {filename}.')
            continue
        else:
            single_day_dfs.append(surfrad_day)
    try:
        all_period_data = pd.concat(single_day_dfs)
    except ValueError:
        logger.warning(f'No data available for site {site.name} '
                       f'from {start} to {end}.')
        return pd.DataFrame()
    all_period_data = all_period_data.rename(
        columns={'temp_air': 'air_temperature'})
    return all_period_data


def initialize_site_observations(api, site):
    """Creates an observaiton at the site for each variable in surfrad_variables.

    Parameters
    ----------
    site : datamodel.Site
        The site object for which to create Observations.
    """
    for variable in surfrad_variables:
        common.create_observation(api, site, variable)


def initialize_site_forecasts(api, site):
    """
    Create forecasts for each variable in surfrad_variables at the site

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    common.create_forecasts(api, site, surfrad_variables,
                            default_forecasts.TEMPLATE_FORECASTS)


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to a list of Surfrad Observations
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
    gaps_only : bool, default False
        If True, only update periods between start and end where there
        are data gaps.
    """
    surfrad_sites = common.filter_by_networks(sites, 'NOAA SURFRAD')
    for site in surfrad_sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end, gaps_only=gaps_only)
