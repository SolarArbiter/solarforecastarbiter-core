import logging


import pandas as pd
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


def fetch(api, site, start, end):
    """Retrieve observation data for a MIDC site between start and end.

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
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        return pd.DataFrame()
    try:
        obs_df = iotools.read_midc_raw_data_from_nrel(
            site_extra_params['network_api_id'], start, end)
    except IndexError:
        # The MIDC api returns a text file on 404 that is parsed as a
        # 1-column csv and causes an index error.
        logger.warning(f'Could not retrieve data for site {site.name}'
                       f' between {start} and {end}.')
        return pd.DataFrame()
    except ValueError as e:
        logger.error(f'Error retrieving data for site {site.name}'
                     f' between {start} and {end}: %s', e)
        return pd.DataFrame()
    return obs_df


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to all MIDC observations from
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
    midc_sites = common.filter_by_networks(sites, ['NREL MIDC'])
    for site in midc_sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end, gaps_only=gaps_only)
