import datetime as dt
import logging
import sys


import click
import requests
import sentry_sdk


from solarforecastarbiter import __version__
from solarforecastarbiter.io.api import request_cli_access_token
from solarforecastarbiter.validation import tasks as validation_tasks


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.WARNING)
sentry_sdk.init(send_default_pii=False)
midnight = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0,
                                                    microsecond=0)


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
    logger.setLevel(loglevel)


def common_options(cmd):
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
                         help='Password to access API')
            ]
        for dec in reversed(decs):
            f = dec(f)
        return f
    return wrapper(cmd)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__)
def cli():
    pass  # pragma: no cover


@cli.command()
@common_options
@click.option('--start', show_default='00:00:00 Yesterday (UTC)',
              default=lambda: midnight - dt.timedelta(days=1),
              help='datetime to start validation at')
@click.option('--end', default=lambda: midnight - dt.timedelta(seconds=1),
              show_default='23:59:59 Yesterday (UTC)',
              help='datetime to end validation at')
@click.option('--base-url', show_default=True,
              default='https://api.solarforecastarbiter.org',
              help='URL of the SolarForecastArbiter API')
@click.argument('observation_id', nargs=-1)
def dailyvalidation(verbose, user, password, start, end, base_url,
                    observation_id):
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
