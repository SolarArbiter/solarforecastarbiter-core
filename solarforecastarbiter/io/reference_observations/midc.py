import logging


from pvlib import iotools
from requests.exceptions import HTTPError


from solarforecastarbiter.io.reference_observations import common, midc_config


logger = logging.getLogger('reference_data')


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in the MIDC
    site's file.

    Parameters
    ----------
    api : io.api.APISession
        API Session object, authenticated for the Reference user.
    site : datamodel.Site
        The site object for which to create the Observations.
    """
    extra_params = common.decode_extra_parameters(site)
    site_api_id = extra_params['network_api_id']
    for sfa_var, midc_var in midc_config.midc_var_map[site_api_id].items():
        obs_extra_params = extra_params.copy()
        obs_extra_params['network_data_label'] = midc_var
        logger.info(f'Creating {sfa_var} at {site.name}')
        try:
            common.create_observation(
                api, site, sfa_var,
                extra_params=obs_extra_params)
        except HTTPError as e:
            logger.error(f'Could not create Observation for "{sfa_var}" '
                         f'at MIDC site {site.name}')
            logger.debug(f'Error: {e.response.text}')


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to all MIDC observations from
    start to end.

    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    sites: list
        List of all reference sites as Objects
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.
    """
    midc_sites = common.filter_by_networks(sites, ['NREL MIDC'])
    for site in midc_sites:
        site_extra_params = common.decode_extra_parameters(site)
        try:
            obs_df = iotools.read_midc_raw_data_from_nrel(
                site_extra_params['network_api_id'], start, end)
        except IndexError:
            logger.warning(f'Could not retrieve data for site {site.name}'
                           f' between {start} and {end}.')
            continue
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            obs_extra_params = common.decode_extra_parameters(obs)
            var_label = obs_extra_params['network_data_label']
            try:
                var_df = obs_df[[var_label]]
            except KeyError:
                logger.error(f'Column {var_label} missing in MIDC data '
                             f'for observation {obs.name}')
            var_df = var_df.rename(columns={var_label: 'value'})
            var_df['quality_flag'] = 0
            # remove when nans work
            var_df = var_df.dropna()
            if var_df.empty:
                logger.warning(
                    f'{obs.name} data empty from '
                    f'{obs_df.index[0]} to {obs_df.index[-1]}.')
                continue
            logger.info(
                f'Updating {obs.name} from '
                f'{var_df.index[0]} to {var_df.index[-1]}.')
            try:
                api.post_observation_values(obs.observation_id,
                                            var_df[start:end])
            except HTTPError as e:
                logger.error(f'Posting data to {obs.name} failed.')
                logger.debug(f'HTTP Error: {e.response.text}')
