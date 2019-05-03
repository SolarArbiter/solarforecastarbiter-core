"""Functions for Creating and Updating NOAA SURFRAD related objects
within the SolarForecastArbiter
"""
from functools import partial
import logging
from urllib.error import URLError


import pandas as pd
from requests.exceptions import HTTPError


from pvlib import iotools
from solarforecastarbiter.io.utils import observation_df_to_json_payload as obs_to_payload # NOQA
from solarforecastarbiter.io.reference_observations import common
from solarforecastarbiter.datamodel import Observation


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
rename_mapping = {'temp_air': 'air_temperature'}

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
    all_period_data = pd.concat(single_day_dfs)
    all_period_data = all_period_data.rename(rename_mapping)
    all_period_data = all_period_data.rename(
        columns={'temp_air': 'air_temperature'})
    return all_period_data


def create_observation(api, site, variable):
    """ Creates a new Observation for the variable and site.

    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : solarforecastarbiter.datamodel.site
        A site object.

    Returns
    -------
    uuid : string
        The uuid of the newly created Observation.

    """
    # Copy network api data from the site, and get the observation's
    # interval length
    extra_parameters = common.decode_extra_parameters(site)
    observation = Observation.from_dict({
        'name': f"{site.name} {variable}",
        'interval_label': 'ending',
        'interval_length': extra_parameters['observation_interval_length'],
        'interval_value_type': 'interval_mean',
        'site': site,
        'uncertainty': 0,
        'variable': variable,
        'extra_parameters': site.extra_parameters
    })
    created = api.create_observation(observation)
    logger.info(f"{created.name} created successfully.")


def initialize_site_observations(api, site):
    """Creates an observaiton at the site for each variable in surfrad_variables.

    Parameters
    ----------
    site : datamodel.Site
        The site object for which to create Observations.
    """
    for variable in surfrad_variables:
        try:
            create_observation(api, site, variable)
        except HTTPError as e:
            logger.error(f'Failed to create {variable} observation as Site '
                         f'{site.name}. Error: {e}')


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to a list of Surfrad Observations
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
    sites = api.list_sites()
    surfrad_sites = filter(partial(common.check_network, 'NOAA SURFRAD'),
                           sites)
    for site in surfrad_sites:
        obs_df = fetch(api, site, start, end)
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            logger.info(
                f'Updating {obs.name} from '
                f'{obs_df.index[0].strftime("%Y%m%dT%H%MZ")} '
                f'to {obs_df.index[-1].strftime("%Y%m%dT%H%MZ")}.')
            var_df = obs_df[[obs.variable]]
            var_df = var_df.rename(columns={obs.variable: 'value'})
            var_df['quality_flag'] = 0
            var_df = var_df.dropna()
            api.post_observation_values(obs.observation_id, var_df)
