import logging
import os


from requests.exceptions import HTTPError


from solarforecastarbiter.io.fetch import sandia
from solarforecastarbiter.io.reference_observations import common


logger = logging.getLogger('reference_data')


SANDIA_VARIABLE_MAP = {
    'AmbientTemp_weather': 'air_temperature',
    'DiffuseIrrad': 'dhi',
    'GlobalIrrad': 'ghi',
    'DirectIrrad': 'dni',
    'RelativeHumidity': 'relative_humidity',
}
def initialize_site_observations(api, site):
    """Creates an observaiton at the site for each variable in surfrad_variables.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Observations.
    """
    extra_params = common.decode_extra_parameters(site)
    for sfa_var in SANDIA_VARIABLE_MAP.values():
        logger.info(f'Creating {sfa_var} at {site.name}')
        try:
            common.create_observation(
                api, site, sfa_var)
        except HTTPError as e:
            logger.error(f'Could not create Observation for "{sfa_var}" '
                         f'at SANDIA site {site.name}')
            logger.debug(f'Error: {e.response.text}')
        

def update_observation_data(api, sites, observations, start, end):
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
    """
    sandia_api_key = os.getenv('SANDIA_API_KEY')
    if sandia_api_key is None:
        logger.error('"SANDIA_API_KEY" environment variable must be '
                     'set to update SANDIA observation data.')
    sandia_sites = common.filter_by_networks(sites, 'SANDIA')
    for site in sandia_sites:
        site_extra_params = common.decode_extra_parameters(site)
        sandia_site_id = site_extra_params['network_api_id']
        obs_df = sandia.fetch_sandia(
            sandia_site_id,
            sandia_api_key,
            start, end)
        obs_df = obs_df.rename(columns=SANDIA_VARIABLE_MAP)
        data_in_range = obs_df[start:end]
        if data_in_range.empty:
            logger.warning(f'Data for site {site.name} contained no '
                           f'entries from {start} to {end}.')
            continue
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            common.post_observation_data(api, obs, data_in_range)
