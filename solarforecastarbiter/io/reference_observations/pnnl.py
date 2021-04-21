"""Initialize site obs/forecasts and fetch/update obs for PNNL site."""

import logging
import os
from pathlib import Path

import pandas as pd

from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)

logger = logging.getLogger('reference_data')

DATA_PATH = Path(os.getenv('PNNL_DATA_DIR', 'pnnl_data'))

# These columns are just the minute data. They do not include the daily
# summary columns.
COLUMNS = [
    'ID',
    'YEAR',
    'DAY',
    'HRMN',
    'PIR_avg(W/m2)',
    '848_avg(W/m2)',
    'NIP_avg(W/m2)',
    'PSP_avg(W/m2)',
    'IRnet_avg(W/m2)',
    'PIR_std(W/m2)',
    '848_std(W/m2)',
    'NIP_std(W/m2)',
    'PSP_std(W/m2)',
    'Tdome_avg(C)',
    'Tcase_avg(C)',
    'Tair_avg(C)',
    'RHair_avg(%)',
    'T10X_avg(C)',
    'p10X_avg(V)',
]

# variables to insert into database
VARIABLE_MAP = {
    'PSP_avg(W/m2)': 'ghi',
    'NIP_avg(W/m2)': 'dni',
    '848_avg(W/m2)': 'dhi',
    'Tair_avg(C)': 'air_temperature',
    'RHair_avg(%)': 'relative_humidity'
}


def initialize_site_observations(api, site):
    """Creates an observation at the site for each VARIABLE_MAP.values().

    Parameters
    ----------
    site : datamodel.Site
        The site object for which to create Observations.
    """
    try:
        extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.warning('Cannot create reference observations at PNNL site '
                       f'{site.name}, missing required parameters.')
        return
    for pnnl_var, sfa_var in VARIABLE_MAP.items():
        obs_extra_params = extra_params.copy()
        obs_extra_params['network_data_label'] = pnnl_var
        logger.info(f'Creating {sfa_var} at PNNL')
        common.create_observation(
            api, site, sfa_var, extra_params=obs_extra_params
        )


def initialize_site_forecasts(api, site):
    """
    Create forecasts for each variable in VARIABLE_MAP.values().

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    common.create_forecasts(api, site, VARIABLE_MAP.values(),
                            default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end):
    """Retrieve observation data for PNNL site for 1 year.

    Assumes that data is available in a directory structure
    ``DATA_PATH/rldradC1.00.{%Y}/rldradC1.00.%Y%m%d.%H%M%S.raw.%Y%m%D%Hpnl.sky``

    Parameters
    ----------
    api : io.APISession
        Unused but conforms to common.update_site_observations call
    site : datamodel.Site
        Unused but conforms to common.update_site_observations call
    start : datetime
        The beginning of the period to request data for.
    end : datetime
        The end of the period to request data for.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """  # noqa: E501
    start = start.tz_convert('UTC')
    end = end.tz_convert('UTC')
    dirs = (
        DATA_PATH / f'rldradC1.00.{year}' for year in
        range(start.year, end.year + 1)
    )
    files = []
    # format of 1st half of file name (before .raw). 2nd half is repeated info
    file_name_fmt = 'rldradC1.00.%Y%m%d.%H%M%S'
    for path in dirs:
        for f in path.iterdir():
            # pull the date from the file name, compare to start, end
            fdate = pd.to_datetime(
                f.name.split('.raw')[0], format=file_name_fmt, utc=True
            )
            if fdate >= start and fdate <= end:
                files.append(f)
    files = sorted(files)
    data = pd.concat([_read_data_file(f) for f in files])
    # handful of duplicates exist due to apparently bad data reads. The
    # first read appears to be the correct one. Also starting on 2018-08-14
    # there is a duplicate entry for the last time of the UTC day. That line
    # contains daily summary data in additional columns
    data = data[~data.index.duplicated()]
    data = data.tz_localize('UTC').tz_convert('Etc/GMT+8')
    return data


def _read_data_file(f):
    data = pd.read_csv(
        f, names=COLUMNS, dtype={'YEAR': str, 'DAY': str, 'HRMN': str}
    )
    date_strings = \
        data['YEAR'] + data['DAY'].str.zfill(3) + data['HRMN'].str.zfill(4)
    index = pd.to_datetime(date_strings, format='%Y%j%H%M')
    data.index = index
    return data


def update_observation_data(api, sites, observations, start, end, *,
                            gaps_only=False):
    """Post new observation data to all PNNL observations from
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
    sites = common.filter_by_networks(sites, ['PNNL'])
    for site in sites:
        common.update_site_observations(
            api, fetch, site, observations, start, end, gaps_only=gaps_only)
