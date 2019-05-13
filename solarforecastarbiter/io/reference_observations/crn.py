from functools import partial
import logging
from urllib.error import URLError


import pandas as pd
from pvlib import iotools
from requests.exceptions import HTTPError

from solarforecastarbiter.io.reference_observations import common


CRN_URL = 'https://www1.ncdc.noaa.gov/pub/data/uscrn/products/subhourly01/'

crn_variables = ['ghi', 'temp_air', 'relative_humidity', 'wind_speed']

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
        return
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
        if variable == 'temp_air':
            common.create_observation(api, site, 'air_temperature')
        else:
            common.create_observation(api, site, variable)


def update_observation_data(api, sites, observations, start, end):
    crn_sites = filter(partial(common.check_network, 'NOAA USCRN'),
                       sites)
    for site in crn_sites:
        obs_df = fetch(api, site, start, end)
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            logger.info(
                f'Updating {obs.name} from '
                f'{obs_df.index[0]} to {obs_df.index[-1]}.')
            var_df = obs_df[[obs.variable]]
            var_df = var_df.rename(columns={obs.variable: 'value'})
            var_df['quality_flag'] = 0
            # temporary dropna
            var_df = var_df.dropna()
            if var_df.empty:
                logger.warning(
                    f'{obs.name} data empty from '
                    f'{obs_df.index[0]} to {obs_df.index[-1]}.')
                continue
            try:
                api.post_observation_values(obs.observation_id,
                                            var_df[start:end])
            except HTTPError as e:
                logger.error(f'Posting data to {obs.name} failed.')
                logger.debug(f'HTTP Error: {e.response.text}')
