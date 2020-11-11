"""A CLI tool for creating reference sites and observations and updating them
with data from their respective API.

As new networks are added, their name should be added as it appears in
`sfa_reference_data.csv` to `NETWORK_OPTIONS`. A module should be created
to handle observation initialization and data update. Each of these network
specific modules should implement two functions:

 * ``initialize_site_observations(api, site)``
    * Where api is an `io.api.APISession` and site is a `datamodel.Site`.
 * ``update_observation_data(api, sites, observations, start, end)``
    * Where api is an `io.api.APISession`, sites is the result of
      `APISession.list_sites`, observations is the result of
      `APISession.list_observations``start` and `end` are datetime
      objects.

The module should then be imported and added to `NETWORKHANDLER_MAP` below,
so that it may be selected based on command line arguments. See the existing
mappings for an example.
"""

import json
import logging
from pkg_resources import resource_filename, Requirement


from requests.exceptions import HTTPError


from solarforecastarbiter.datamodel import Site
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.io.reference_observations import (
    surfrad,
    solrad,
    crn,
    midc,
    srml,
    rtc,
    common,
    arm,
    pvdaq,
    eia,
    bsrn
)


# maps network names to the modules that interact with their api
NETWORKHANDLER_MAP = {
    'NOAA SURFRAD': surfrad,
    'NOAA SOLRAD': solrad,
    'NOAA USCRN': crn,
    'UO SRML': srml,
    'NREL MIDC': midc,
    'DOE RTC': rtc,
    'DOE ARM': arm,
    'NREL PVDAQ': pvdaq,
    'EIA': eia,
    'WRMC BSRN': bsrn,
}

# list of options for the 'network' argument
NETWORK_OPTIONS = ['NOAA SURFRAD', 'NOAA SOLRAD', 'NOAA USCRN', 'NREL MIDC',
                   'UO SRML', 'DOE RTC', 'DOE ARM', 'NREL PVDAQ', 'EIA',
                   'WRMC BSRN']

DEFAULT_SITEFILE = resource_filename(
    Requirement.parse('solarforecastarbiter'),
    'solarforecastarbiter/io/reference_observations/sfa_reference_sites.csv')


logger = logging.getLogger('reference_data')

CLI_DESCRIPTION = """
Tool for initializing and updating SolarForecastArbiter reference observation data.
Supports importing sites from the following networks:

NOAA (The National Oceanic and Atmospheric Administration)

    SURFRAD: Surface Radiation Budget Network
    https://www.esrl.noaa.gov/gmd/grad/surfrad/

    SOLRAD:
    https://www.esrl.noaa.gov/gmd/grad/solrad/index.html

    CRN: U.S. Climate Reference Network
    https://www.ncdc.noaa.gov/crn/

NREL MIDC: National Renewable Energy Laboratory Measurement and Instrumentation Data Center
https://midcdmz.nrel.gov/

UO SRML: University of Oregon Solar Radiation Monitoring Laboratory
http://solardat.uoregon.edu/

DOE RTC: DOE Regional Test Centers for Solar Technologies
https://pv-dashboard.sandia.gov/

DOE ARM: DOE Atmospheric Radiation Measurement
https://www.arm.gov/

NREL PVDAQ: National Renewable Energy Laboratory PV Data Acquisition
https://developer.nrel.gov/docs/solar/pvdaq-v3/

EIA: U.S. Energy Information Administration Open Data
https://www.eia.gov/opendata/

WRMC BSRN: World Radiation Monitoring Center - Baseline Surface Radiation Network
https://bsrn.awi.de
"""  # noqa: E501


def get_apisession(token, base_url=None):
    return APISession(token, base_url=base_url)


def create_site(api, site):
    """Post a new site to the API and create it's applicable reference
    observations.

    Parameters
    ----------
    api : io.APISession
        An APISession with a valid JWT for accessing the Reference Data user.
    site : dict
        Dictionary describing the site to post. This will be instantiated as
        a datamodel.Site object and the value of 'extra_parameters' will be
        serialized to json.

    Returns
    -------
    datamodel.Site
        The created site object.
    """
    # get a reference to network before we serialize extra_parameters
    network = site['extra_parameters']['network']
    network_handler = NETWORKHANDLER_MAP.get(network)
    if network_handler is None:
        logger.warning(f'Unrecognized network, {network} on Site '
                       f'{site["name"]} Observations cannot be '
                       'automatically generated.')
        return
    site.update({'extra_parameters': json.dumps(site['extra_parameters'])})
    site_name = f"{network} {common.clean_name(site['name'])}"
    existing = common.existing_sites(api)
    if site_name in existing:
        logger.info('Site, %s, already exists', site_name)
        created = existing[site_name]
    else:
        site['name'] = site_name
        site_to_create = Site.from_dict(site)
        try:
            created = api.create_site(site_to_create)
        except HTTPError as e:
            logger.error(f"Failed to create Site {site['name']}.")
            logger.debug(f'HTTP Error: {e.response.text}')
            return False
        else:
            logger.info(f'Created Site {created.name} successfully.')
    network_handler.initialize_site_observations(api, created)
    try:
        network_handler.initialize_site_forecasts(api, created)
    except ValueError as e:
        logger.error('Cannot create forecasts for %s: %s', site_name, e)
    return created


def initialize_reference_metadata_objects(token, sites, base_url=None):
    """Instantiate an API session and create reference sites and
    observations.

    Parameters
    ----------
    token : str
        Access token for the SFA API
    sites : list
        List of site dictionary objects. The 'extra_parameters'
        key should contain a nested dict with the following keys:
            network
            network_api_id
            network_api_abbreviation
            observation_interval_length
    base_url : str
        The alternate base url of the SFA API
    """
    logger.info('Initializing reference metadata...')
    api = get_apisession(token, base_url)
    sites_created = 0
    failures = 0
    for site in sites:
        if create_site(api, site):
            sites_created = sites_created + 1
        else:
            failures = failures + 1
    logger.info(f'Created {sites_created} sites successfully, with '
                f'{failures} failures.')


def update_reference_observations(token, start, end, networks, base_url=None,
                                  *, gaps_only=False):
    """Coordinate updating all existing reference observations.

    Parameters
    ----------
    token : str
        Access token for the SFA API
    start : datetime-like
    end : datetime-like
    networks : list
        List of network names to update.
    base_url : str
        The alternate base url of the SFA API
    gaps_only : bool, default False
        If True, only update periods between start and end where there
        are data gaps.
    """
    api = get_apisession(token, base_url)
    observations = api.list_observations()
    sites = {obs.site for obs in observations}
    for network in networks:
        network_handler = NETWORKHANDLER_MAP.get(network)
        if network_handler is None:
            logger.info(f'{network} observation updates not configured.')
        else:
            network_handler.update_observation_data(
                api, sites, observations, start, end, gaps_only=gaps_only)


def site_df_to_dicts(site_df):
    """ Creates a list of site dictionaries ready to be serialized
    and posted to the api.

    Parameters
    ----------
    site_df: DataFrame
        Pandas Dataframe with the columns:
        interval_length, name, latitude, longitude, elevation,
        timezone, network, network_api_id, network_api_abbreviation,
        attribution

    Returns
    -------
    list
        A list of Site dictionaries.
    """
    site_list = []
    for i, row in site_df.iterrows():
        row = row.fillna('')
        site = {
            'name': row['name'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'elevation': row['elevation'],
            'timezone': row['timezone'],
            'extra_parameters': {
                'network': row['network'],
                'network_api_id': row['network_api_id'],
                'network_api_abbreviation': row['network_api_abbreviation'],
                'observation_interval_length': row['interval_length'],
                'attribution': row['attribution']
            }
        }
        network_handler = NETWORKHANDLER_MAP.get(row['network'])
        if hasattr(network_handler, 'adjust_site_parameters'):
            site = network_handler.adjust_site_parameters(site)
        site_list.append(site)
    return site_list
