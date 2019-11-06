import logging


from pvlib import iotools


from solarforecastarbiter.io.reference_observations import (
    common, midc_config, default_forecasts)


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
    try:
        extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.warning('Cannot create reference observations at MIDC site '
                       f'{site.name}, missing required parameters.')
        return
    site_api_id = extra_params['network_api_id']
    for sfa_var, midc_var in midc_config.midc_var_map[site_api_id].items():
        obs_extra_params = extra_params.copy()
        obs_extra_params['network_data_label'] = midc_var
        common.create_observation(api, site, sfa_var,
                                  extra_params=obs_extra_params)


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
        logger.warning('Cannot create reference observations at MIDC site '
                       f'{site.name}, missing required parameters.')
        return
    site_api_id = extra_params['network_api_id']
    common.create_forecasts(
        api, site, midc_config.midc_var_map[site_api_id].keys(),
        default_forecasts.TEMPLATE_FORECASTS)


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
        try:
            site_extra_params = common.decode_extra_parameters(site)
        except ValueError:
            continue
        try:
            obs_df = iotools.read_midc_raw_data_from_nrel(
                site_extra_params['network_api_id'], start, end)
        except IndexError:
            # The MIDC api returns a text file on 404 that is parsed as a
            # 1-column csv and causes an index error.
            logger.warning(f'Could not retrieve data for site {site.name}'
                           f' between {start} and {end}.')
            continue
        site_observations = [obs for obs in observations if obs.site == site]
        for obs in site_observations:
            common.post_observation_data(api, obs, obs_df, start, end)
