"""A CLI tool for creating reference sites and observations and updating them
with data from their respective API.
"""
import argparse
from argparse import RawTextHelpFormatter
import inspect
import logging
import os
import sys


import pandas as pd
from requests.exceptions import HTTPError


from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.io.reference_observations import surfrad


SFA_REFERENCE_TOKEN = os.getenv('SFA_REFERENCE_TOKEN')
SFA_API_BASE_URL = os.getenv('SFA_API_BASE_URL')

# list of options for the 'network' argument
NETWORK_OPTIONS = ['surfrad', 'midc', 'srml', 'solrad', 'arm']

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
DEFAULT_SITEFILE = os.path.join(SCRIPT_DIR, "sfa_reference_sites.csv")

logging.basicConfig()
logger = logging.getLogger('reference_data')

CLI_DESCRIPTION = """
Tool for initializing and updating SolarForecastArbiter reference observation data.
Supports importing sites from the following networks:

NOAA SURFRAD: The National Oceanic and Atmospheric Administration Surface Radiation BudgetNetwork
    https://www.esrl.noaa.gov/gmd/grad/surfrad/

NREL MIDC: National Renewable Energy Laboratory Measurement and Instrumentation Data Center
    https://midcdmz.nrel.gov/

UO SRML: University of Oregon Solar Radiation Monitoring Laboratory
    http://solardat.uoregon.edu/

SANDIA: Sandia National Laboratory Regional Test Centers for Solar Technologies
    https://pv-dashboard.sandia.gov/
"""  # noqa: E501
parser = argparse.ArgumentParser(
    formatter_class=RawTextHelpFormatter,
    description=CLI_DESCRIPTION)
parser.add_argument('-v', '--verbose', action='count')
subparsers = parser.add_subparsers(help='Commands', dest='command',
                                   required=True)

init_parser = subparsers.add_parser(
    'init',
    help='Creates sites and observations from a site file.')
init_parser.add_argument(
    '--site-file', default=DEFAULT_SITEFILE,
    help='The file from which to load all of the reference site metadata. '
         'Defaults to `sfa_reference_sites.csv.')

update_parser = subparsers.add_parser(
    'update',
    help='Updates reference data for the given period.')
update_parser.add_argument(
    '--networks', nargs='+', default=NETWORK_OPTIONS,
    choices=NETWORK_OPTIONS)
update_parser.add_argument(
    'start',
    help="Beginning of the period to update as a ISO8601 datetime string.")
update_parser.add_argument(
    'end',
    help="End of the period to update as an ISO8601 datetime string.")

init_parser = subparsers.add_parser(
    'remove', help='Deletes all reference sites and observations.')


def get_apisession():
    return APISession(SFA_REFERENCE_TOKEN, base_url=SFA_API_BASE_URL)


def initialize_reference_objects(sites):
    """Instantiate an API session and create reference sites.

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
    surfrad_sites = [site for site in sites
                     if site['extra_parameters']['network'] == "NOAA SURFRAD"]
    surfrad.initialize_metadata_objects(api, surfrad_sites)


def update_reference_observations(start, end, networks):
    """Coordinate updating all existing reference observations
    """
    api = get_apisession()
    surfrad.update_observation_data(api, start, end)


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
    cli_args = parser.parse_args()
    if cli_args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif cli_args.verbose and cli_args.verbose > 1:
        logger.setLevel(logging.DEBUG)

    cmd = cli_args.command
    if cmd == 'init':
        filename = cli_args.site_file
        try:
            all_sites = pd.read_csv(filename, comment='#')
        except FileNotFoundError:
            logger.error(f'Site file does not exist: {filename}')
            sys.exit()
        site_dictionaries = site_df_to_dicts(all_sites)
        initialize_reference_objects(site_dictionaries)

    elif cmd == 'update':
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
        confirm = input('This action will delete all sites and '
                        'observations in the Reference organization '
                        'are you sure you want to continue? [y/n]:')
        if confirm != 'y':
            print('Exiting.')
            sys.exit()
        api = get_apisession()
        observations = api.list_observations()
        for obs in observations:
            try:
                api.delete(f'/observations/{obs.observation_id}')
            except HTTPError:
                logger.error(f'Could not delete observation {obs.name}'
                             'with observation_id {obs.observation_id}.')
        sites = api.list_sites()
        for site in sites:
            try:
                api.delete(f'/sites/{site.site_id}')
            except HTTPError:
                logger.error(f'Could not delete site {site.name} with '
                             'site_id {site.site_id}.')
