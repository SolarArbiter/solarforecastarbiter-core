"""A CLI tool for creating reference sites and observations and updating them
with data from their respective API.

Requires that you set the environment variables:

    SFA_REFERENCE_TOKEN
        A valid access token for the SolarForecastArbiter reference user.
    SFA_API_BASE_URL
        The base url of the SolarForecastArbiter API to use.
        e.g. https://api.solarforecastarbiter.org

As new networks are added, their name should be added as it appears in
`sfa_reference_data.csv` to `NETWORK_OPTIONS`. A module should be created
to handle observation initialization and data update. Each of these network
specific modules should implement two functions:

    initialize_site_observations(api, site)

        Where api is an `io.api.APISession` and site is a `datamodel.Site`.

    update_observation_data(api, sites, observations, start, end)

        Where api is an `io.api.APISession`, sites is the result of
        `APISession.list_sites`, observations is the result of
        `APISession.list_observations``start` and `end` are datetime
        objects.
The module should then be imported and added to `NETWORKHANDLER_MAP` below,
so that it may be selected based on command line arguments. See the existing
mappings for an example.
"""
import argparse
from argparse import RawTextHelpFormatter
import inspect
import json
import logging
import os
import sys


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.datamodel import Site
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.io.reference_observations import (surfrad, solrad,
                                                            srml, common)


# maps network names to the modules that interact with their api
NETWORKHANDLER_MAP = {
    'NOAA SURFRAD': surfrad,
    'NOAA SOLRAD': solrad,
    'UO SRML': srml,
}

# list of options for the 'network' argument
NETWORK_OPTIONS = ['NOAA SURFRAD', 'NOAA SOLRAD', 'NOAA USCRN', 'NREL MIDC',
                   'UO SRML', 'ARM', 'SANDIA']

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
DEFAULT_SITEFILE = os.path.join(SCRIPT_DIR, "sfa_reference_sites.csv")

SFA_REFERENCE_TOKEN = os.getenv('SFA_REFERENCE_TOKEN')
SFA_API_BASE_URL = os.getenv('SFA_API_BASE_URL')

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

SANDIA: Sandia National Laboratory Regional Test Centers for Solar Technologies
    https://pv-dashboard.sandia.gov/
"""  # noqa: E501


def add_network_arg(parser):
    """Adds the `--networks` argument to a subparser so that it does not consume
    the the init, update and remove commands
    """
    parser.add_argument(
        '--networks', nargs='+', default=NETWORK_OPTIONS,
        choices=NETWORK_OPTIONS,
        help="The Networks to act on. Defaults to all.")


def get_apisession():
    return APISession(SFA_REFERENCE_TOKEN, base_url=SFA_API_BASE_URL)


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
    uuid : string
        UUID of the created Site.
    """
    # get a reference to network before we serialize extra_parameters
    network = site['extra_parameters']['network']
    site.update({'extra_parameters': json.dumps(site['extra_parameters'])})
    site['name'] = f'{network} {site["name"]}'
    site_to_create = Site.from_dict(site)
    try:
        created = api.create_site(site_to_create)
    except HTTPError as e:
        logger.error(f"Failed to create Site {site['name']}.")
        logger.debug(f'HTTP Error: {e.response.text}')
    else:
        logger.info(f'Created Site {created.name} successfully.')
        network_handler = NETWORKHANDLER_MAP.get(network)
        if network_handler is None:
            logger.warning(f'Unrecognized network, {network} on Site '
                           f'{site["name"]} Observations cannot be '
                           'automatically generated.')
        else:
            network_handler.initialize_site_observations(api, created)
        return created


def initialize_reference_metadata_objects(sites):
    """Instantiate an API session and create reference sites and
    observations.

    Parameters
    ----------
    sites: list
        List of site dictionary objects. The 'extra_parameters'
        key should contain a nested dict with the following keys:
            network
            network_api_id
            network_api_abbreviation
            observation_interval_length
    """
    logger.info('Initializing reference metadata...')
    api = get_apisession()
    for site in sites:
        create_site(api, site)


def update_reference_observations(start, end, networks):
    """Coordinate updating all existing reference observations.

    Parameters
    ----------
    start : datetime-like
    end : datetime-like
    networks : list
        List of network names to update.
    """
    api = get_apisession()
    sites = api.list_sites()
    observations = api.list_observations()
    for network in networks:
        network_handler = NETWORKHANDLER_MAP.get(network)
        if network_handler is None:
            logger.info(f'{network} observation updates not configured.')
        else:
            network_handler.update_observation_data(api, sites, observations,
                                                    start, end)


def site_df_to_dicts(site_df):
    """ Creates a list of site dictionaries ready to be serialized
    and posted to the api.

    Parameters
    ----------
    site_df: DataFrame
        Pandas Dataframe with the columns:
        interval_length, name, latitude, longitude, elevation,
        timezone, network, network_api_id, network_api_abbreviation

    Returns
    -------
    list
        A list of Site dictionaries.
    """
    site_list = []
    for i, row in site_df.iterrows():
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
                'observation_interval_length': row['interval_length']
            }
        }
        site_list.append(site)
    return site_list


if __name__ == '__main__':
    logging.basicConfig()

    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=CLI_DESCRIPTION)
    parser.add_argument('-v', '--verbose', action='count')
    subparsers = parser.add_subparsers(help='Commands', dest='command',
                                       required=True)
    # Init command parser/options
    init_parser = subparsers.add_parser(
        'init',
        help='Creates sites and observations from a site file.')
    add_network_arg(init_parser)
    init_parser.add_argument(
        '--site-file', default=DEFAULT_SITEFILE,
        help='The file from which to load all of the reference site metadata. '
             'Defaults to `sfa_reference_sites.csv.')

    # Update command parser/options
    update_parser = subparsers.add_parser(
        'update',
        help='Updates reference data for the given period.')
    add_network_arg(update_parser)
    update_parser.add_argument(
        'start',
        help="Beginning of the period to update as a ISO8601 datetime string.")
    update_parser.add_argument(
        'end',
        help="End of the period to update as an ISO8601 datetime string.")

    # Remove command parser/options
    remove_parser = subparsers.add_parser(
        'remove', help='Deletes all reference sites and observations.')
    add_network_arg(remove_parser)

    cli_args = parser.parse_args()
    if cli_args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif cli_args.verbose and cli_args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    networks = cli_args.networks

    cmd = cli_args.command

    if cmd == 'init':
        # Create sites and optionally create observations from rows of a csv
        filename = cli_args.site_file
        try:
            all_sites = pd.read_csv(filename, comment='#')
        except FileNotFoundError:
            logger.error(f'Site file does not exist: {filename}')
            sys.exit()
        network_filtered_sites = all_sites[all_sites['network'].isin(networks)]
        site_dictionaries = site_df_to_dicts(network_filtered_sites)
        initialize_reference_metadata_objects(site_dictionaries)

    elif cmd == 'update':
        # Update observations with network data from the period between
        # start and end
        try:
            start = pd.Timestamp(cli_args.start)
        except ValueError:
            logger.error('Invalid start datetime.')
            sys.exit()

        end = cli_args.end
        if end is not None:
            try:
                end = pd.Timestamp(end)
            except ValueError:
                logger.error('Invalid end datetime.')
                sys.exit()

        networks = cli_args.networks
        update_reference_observations(start, end, networks)

    elif cmd == 'remove':
        # Remove all sites and observations in networks
        confirm = input('This action will delete all sites and '
                        'observations in the networks: '
                        f'{", ".join(networks)} from the Reference '
                        'organization are you sure you want to '
                        'continue? [y/n]:')
        if confirm != 'y':
            print('Exiting.')
            sys.exit()
        api = get_apisession()
        observations = common.filter_by_networks(api.list_observations(),
                                                 networks)
        for obs in observations:
            try:
                api.delete(f'/observations/{obs.observation_id}')
            except HTTPError as e:
                logger.error(f'Could not delete observation {obs.name}'
                             f'with observation_id {obs.observation_id}.')
                logger.debug(f'HTTP Error: {e.response.text}')
        sites = common.filter_by_networks(api.list_sites(),
                                          networks)
        for site in sites:
            try:
                api.delete(f'/sites/{site.site_id}')
            except HTTPError as e:
                logger.error(f'Could not delete site {site.name} with '
                             f'site_id {site.site_id}.')
                logger.debug(f'HTTP Error: {e.response.text}')
