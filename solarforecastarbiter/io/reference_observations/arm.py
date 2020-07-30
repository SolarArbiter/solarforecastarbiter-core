from functools import partial
import json
import logging
import os
from pkg_resources import resource_filename, Requirement


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.io.fetch import arm
from solarforecastarbiter.io.reference_observations import (
    common, default_forecasts)


DEFAULT_SITEFILE = resource_filename(
    Requirement.parse('solarforecastarbiter'),
    'solarforecastarbiter/io/reference_observations/'
    'arm_reference_sites.json')

DOE_ARM_SITE_VARIABLES = {
    'qcrad': arm.IRRAD_VARIABLES,
    'met': arm.MET_VARIABLES,
}

DOE_ARM_VARIABLE_MAP = {
    'down_short_hemisp': 'ghi',
    'short_direct_normal': 'dni',
    'down_short_diffuse_hemisp': 'dhi',
    'temp_mean': 'air_temperature',
    'rh_mean': 'relative_humidity',
    'wspd_arith_mean': 'wind_speed',
}

logger = logging.getLogger('reference_data')


def _determine_stream_vars(datastream):
    """Returns a list of variables available based on datastream name.

    Parameters
    ----------
    datastream: str
        Datastream name, or the product name. This string is searched for
        `met` or `qcrad` and returns a list of expected variables.

    Returns
    -------
    list of str
        The variable names that can be found in the file.
    """
    available = []
    for stream_type, arm_vars in DOE_ARM_SITE_VARIABLES.items():
        if stream_type in datastream:
            available = available + arm_vars
    return available


def initialize_site_observations(api, site):
    """Creates an observation at the site for each variable in
    the matched DOE_ARM_VARIABLE_MAP.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Observations.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.error(f'Failed to initialize observations  for {site.name} '
                     'extra parameters could not be loaded.')
        return

    site_vars = site_variables_from_extra_params(site_extra_params)
    for sfa_var in site_vars:
        logger.info(f'Creating {sfa_var} at {site.name}')
        try:
            common.create_observation(
                api, site, sfa_var)
        except HTTPError as e:
            logger.error(f'Could not create Observation for "{sfa_var}" '
                         f'at DOE ARM site {site.name}')
            logger.debug(f'Error: {e.response.text}')


def initialize_site_forecasts(api, site):
    """
    Create a forecast for each variable at the site.

    Parameters
    ----------
    api : solarforecastarbiter.io.api.APISession
        An active Reference user session.
    site : datamodel.Site
        The site object for which to create Forecasts.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        logger.error('Failed to initialize reference forecasts for '
                     f'{site.name} extra parameters could not be loaded.')
        return

    site_vars = site_variables_from_extra_params(site_extra_params)

    common.create_forecasts(api, site, site_vars,
                            default_forecasts.TEMPLATE_FORECASTS)


def fetch(api, site, start, end, *, doe_arm_user_id, doe_arm_api_key):
    """Retrieve observation data for a DOE ARM site between start and end.

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
    doe_arm_user_id : str
        User ID to access the DOE ARM api.
    doe_arm_api_key : str
        API key to access the DOE ARM api.

    Returns
    -------
    data : pandas.DataFrame
        All of the requested data concatenated into a single DataFrame.
    """
    try:
        site_extra_params = common.decode_extra_parameters(site)
    except ValueError:
        return pd.DataFrame()

    available_datastreams = site_extra_params['datastreams']

    datastreams = {}
    # Build a dict with top-level keys to 'met' and 'qcrad' if meteorological
    # or irradiance  data exists at the site. This is to later group dataframes
    # created from each datastream by the type of data found in the stream.
    for ds_type in ['met', 'qcrad']:
        if ds_type in available_datastreams:
            ds_type_dict = {}
            streams = available_datastreams[ds_type]

            # When a dict is present each key is a datastream and value is
            # a date range for which the datastream contains data. We need to
            # determine which streams to use to get all of the requested data.
            if isinstance(streams, dict):
                ds_type_dict.update(
                    find_stream_data_availability(streams, start, end))
            else:
                # If a single string datastream name exists, we assume that all
                # available data is contained in the stream. Deferring to the
                # data fetch process, which will fail to retrieve data and
                # continue gracefully.
                ds_type_dict[streams] = (start, end)
            datastreams[ds_type] = ds_type_dict

    site_dfs = []

    for stream_type in datastreams:
        # Stitch together all the datastreams with similar data.
        stream_type_dfs = []
        for datastream, date_range in datastreams[stream_type].items():
            stream_df = arm.fetch_arm(
                doe_arm_user_id,
                doe_arm_api_key,
                datastream,
                _determine_stream_vars(datastream),
                date_range[0].tz_convert(site.timezone),
                date_range[1].tz_convert(site.timezone)
            )
            if stream_df.empty:
                logger.warning(f'Datastream {datastream} for site {site.name} '
                               f'contained no entries from {start} to {end}.')
            else:
                stream_type_dfs.append(stream_df)
        if stream_type_dfs:
            # Concatenate all dataframes of similar data
            stream_type_df = pd.concat(stream_type_dfs)
            site_dfs.append(stream_type_df)

    if site_dfs:
        # Join dataframes with different variables along the index, this has
        # the side effect of introducing missing data if any requests have
        # failed.
        obs_df = pd.concat(site_dfs, axis=1)
        obs_df = obs_df.rename(columns=DOE_ARM_VARIABLE_MAP)
        return obs_df
    else:
        logger.warning(f'Data for site {site.name} contained no entries from '
                       f'{start} to {end}.')
        return pd.DataFrame()


def update_observation_data(api, sites, observations, start, end):
    """Post new observation data to a list of DOE ARM Observations
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
    doe_arm_api_key = os.getenv('DOE_ARM_API_KEY')
    if doe_arm_api_key is None:
        raise KeyError('"DOE_ARM_API_KEY" environment variable must be '
                       'set to update DOE ARM observation data.')
    doe_arm_user_id = os.getenv('DOE_ARM_USER_ID')
    if doe_arm_user_id is None:
        raise KeyError('"DOE_ARM_USER_ID" environment variable must be '
                       'set to update DOE ARM observation data.')

    doe_arm_sites = common.filter_by_networks(sites, 'DOE ARM')
    for site in doe_arm_sites:
        common.update_site_observations(
            api, partial(fetch, doe_arm_user_id=doe_arm_user_id,
                         doe_arm_api_key=doe_arm_api_key),
            site, observations, start, end)


def adjust_site_parameters(site):
    """Updates extra parameters with applicable datastreams from
    `arm_reference_sites.json`

    Parameters
    ----------
    site: dict

    Returns
    -------
    dict
        Copy of input with updated extra parameters.
    """
    with open(DEFAULT_SITEFILE) as fp:
        sites_metadata = json.load(fp)['sites']

    # ARM has multiple 'locations' at each 'site'. In the Solar Forecast
    # Arbiter we store each 'location' as a site. We use the `network_api_id`
    # to indicate location, and `network_api_abbreviation` together to indicate
    # the arm site. This is necessary to use both because location ids are only
    # unique for a given site.
    arm_location_id = site['extra_parameters']['network_api_id']
    arm_site_id = site['extra_parameters']['network_api_abbreviation']

    for site_metadata in sites_metadata:
        site_extra_params = json.loads(site_metadata['extra_parameters'])
        if (
            site_extra_params['network_api_id'] == arm_location_id
            and site_extra_params['network_api_abbreviation'] == arm_site_id

        ):
            site_out = site.copy()
            site_out['extra_parameters'] = site_extra_params
            return site_out
    return site


def find_stream_data_availability(streams, start, end):
    """Determines what date ranges to use for each datastream.

    Parameters
    ----------
    streams: dict
        Dict where values are string datastream names and values are iso8601
        date ranges `start/end` indicating the period of data available at
        that datastream.
    start: datetime
        The start of the period to request data for.
    end: datetime
        The end of the period to request data for.

    Returns
    -------
    dict:
        Dict where keys are datastreams and values are two element lists of
        `[start datetime, end datetime]` that when considered together should
        span all of the available data between the requested start and end.
    """
    stream_range_dict = {}

    # Find the overlap between each streams available data, and the requested
    # period
    for datastream, date_range in streams.items():
        stream_range = parse_iso_date_range(date_range)
        overlap = get_period_overlap(
            start, end, stream_range[0], stream_range[1])
        if overlap is None:
            # The datastream did not contain any data within the requested
            # range, we don't need to use it for this request.
            continue
        else:
            stream_range_dict[datastream] = overlap

    # Remove any overlap between streams
    prev_start = None
    prev_end = None
    for datastream, date_range in stream_range_dict.items():
        if prev_start is not None:
            # if end of this datastream overlaps previous start and the start
            # of this datastream is before the previous end, move the end of
            # this stream to the previous start to remove overlap
            if date_range[1] > prev_start and date_range[0] < prev_start:
                date_range[1] = prev_start
        else:
            prev_start = date_range[0]
        if prev_end is not None:
            # if start of this datastream is before the previous end, and the
            # end of this datastream is after the previous end, move the start
            # of this stream to the previous end to remove overlap
            if date_range[0] < prev_end and date_range[1] > prev_end:
                date_range[0] = prev_end
        else:
            prev_end = date_range[1]

    return stream_range_dict


def get_period_overlap(request_start, request_end, avail_start, avail_end):
    """Finds period of overlap between the requested time range and the
    available period.

    Parameters
    ----------
    request_start: datetime-like
        Start of the period of requested data.
    request_end: datetime-like
        End of the period of requested data.
    avail_start: datetime-like
        Start of the available data.
    avail_end: datatime-like
        End of available data.

    Returns
    -------
    start, end: list of datetime or None
        Start and end of overlapping period, or None if no overlap occurred.
    """
    if request_start < avail_end and request_end > avail_start:
        if request_start < avail_start:
            start = avail_start
        else:
            start = request_start
        if request_end > avail_end:
            end = avail_end
        else:
            end = request_end
        return [start, end]
    else:
        return None


def parse_iso_date_range(date_range_string):
    """Parses a date range string in iso8601 format ("start date/end date")
    into a tuple of pandas timestamps `(start, end)`.
    """
    start, end = date_range_string.split('/')
    return (pd.Timestamp(start, tz='utc'), pd.Timestamp(end, tz='utc'))


def site_variables_from_extra_params(site_extra_params):
    """Return variables expected at the site, based on the content of the
    `datastreams` attribute of the site's extra parameters.
    """
    return [DOE_ARM_VARIABLE_MAP[arm_var]
            for stream_type in site_extra_params['datastreams'].keys()
            for arm_var in _determine_stream_vars(stream_type)]
