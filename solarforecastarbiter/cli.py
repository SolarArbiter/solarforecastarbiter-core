import logging
import sys


import click
import pandas as pd
import requests
import sentry_sdk


from solarforecastarbiter import __version__
from solarforecastarbiter.io.api import request_cli_access_token
from solarforecastarbiter.io.reference_observations import reference_data
from solarforecastarbiter.validation import tasks as validation_tasks


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.WARNING)
sentry_sdk.init(send_default_pii=False)
midnight = pd.Timestamp.utcnow().floor('1d')


def cli_access_token(user, password):
    try:
        token = request_cli_access_token(user, password)
    except requests.HTTPError as e:
        click.echo(click.style(
            e.response.json()['error_description'], fg='red'))
        sys.exit(1)
    else:
        return token


def set_log_level(verbose):
    if verbose == 1:
        loglevel = 'INFO'
    elif verbose > 1:
        loglevel = 'DEBUG'
    else:
        loglevel = 'WARNING'
    logging.getLogger().setLevel(loglevel)


class UTCTimestamp(click.ParamType):
    """Convert a timestamp string to a Pandas Timestamp localized to UTC"""
    name = 'UTCTimestamp'

    def convert(self, value, param, ctx):
        try:
            out = pd.Timestamp(value)
        except ValueError:
            self.fail('%s cannot be converted into a Pandas.Timestamp'
                      % value, param, ctx)
        else:
            if out.tzinfo:
                return out.tz_convert('UTC')
            else:
                return out.tz_localize('UTC')


UTCTIMESTAMP = UTCTimestamp()


def common_options(cmd):
    """Combine common options into one decorator"""
    def wrapper(f):
        decs = [
            click.option('-v', '--verbose', count=True,
                         help='Increase logging verbosity'),
            click.option('-u', '--user', show_envvar=True,
                         help='Username to access API.',
                         envvar='SFA_API_USER',
                         required=True),
            click.option('-p', '--password', show_envvar=True,
                         envvar='SFA_API_PASSWORD',
                         required=True,
                         prompt=True, hide_input=True,
                         help='Password to access API'),
            click.option('--base-url', show_default=True,
                         envvar='SFA_API_BASE_URL',
                         show_envvar=True,
                         default='https://api.solarforecastarbiter.org',
                         help='URL of the SolarForecastArbiter API')
            ]
        for dec in reversed(decs):
            f = dec(f)
        return f
    return wrapper(cmd)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def cli():
    """
    The SolarForecastArbiter core command line tool
    """
    pass  # pragma: no cover


@cli.command()
@common_options
@click.option('--start', show_default='00:00:00 Yesterday (UTC)',
              type=UTCTIMESTAMP,
              default=lambda: midnight - pd.Timedelta(days=1),
              help='datetime to start validation at')
@click.option('--end', default=lambda: midnight - pd.Timedelta(seconds=1),
              type=UTCTIMESTAMP,
              show_default='23:59:59 Yesterday (UTC)',
              help='datetime to end validation at')
@click.argument('observation_id', nargs=-1)
def dailyvalidation(verbose, user, password, start, end, base_url,
                    observation_id):
    """
    Run the daily validation tasks for a given set of observations
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    if not observation_id:
        logger.info(
            ('Validating daily observation data from %s to %s for all '
             'observations'), start, end)
        validation_tasks.daily_observation_validation(token, start, end,
                                                      base_url)
    else:
        logger.info(
            ('Validating daily observation data from %s to %s for '
             'observations:\n\t%s'), start, end, ','.join(observation_id))
        for obsid in observation_id:
            validation_tasks.daily_single_observation_validation(
                token, obsid, start, end, base_url)


@cli.group(help=reference_data.CLI_DESCRIPTION)
def referencedata():
    pass  # pragma: no cover


network_opt = click.option(
    '--network', multiple=True,
    help="The Networks to act on. Defaults to all.",
    default=reference_data.NETWORK_OPTIONS,
    type=click.Choice(reference_data.NETWORK_OPTIONS))


@referencedata.command(name='init')
@common_options
@network_opt
@click.option(
    '--site-file', type=click.Path(exists=True, resolve_path=True),
    default=reference_data.DEFAULT_SITEFILE,
    help='The file from which to load all of the reference site metadata.')
def referencedata_init(verbose, user, password, base_url, network, site_file):
    """
    Creates sites and observations from a site file.
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    # click checks if path exists
    all_sites = pd.read_csv(site_file, comment='#')
    network_filtered_sites = all_sites[all_sites['network'].isin(network)]
    site_dictionaries = reference_data.site_df_to_dicts(network_filtered_sites)
    reference_data.initialize_reference_metadata_objects(
        token, site_dictionaries, base_url)


@referencedata.command(name='update')
@common_options
@network_opt
@click.argument('start', type=UTCTIMESTAMP)
@click.argument('end', type=UTCTIMESTAMP)
def referencedata_update(verbose, user, password, base_url, network, start,
                         end):
    """
    Updates reference data for the given period. START and END should be given
    as ISO8601 datetime strings. If no timezone is defined, UTC will be
    assumed.
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    reference_data.update_reference_observations(token, start, end, network,
                                                 base_url)
