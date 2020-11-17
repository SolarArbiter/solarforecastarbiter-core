import asyncio
from functools import partial
import json
import logging
from pathlib import Path
import signal
import sys


import click
import pandas as pd
import requests
import sentry_sdk


from solarforecastarbiter import __version__
from solarforecastarbiter.io import nwp, reference_aggregates
from solarforecastarbiter.io.api import request_cli_access_token, APISession
from solarforecastarbiter.io.fetch import update_num_workers
from solarforecastarbiter.io.reference_observations import reference_data
from solarforecastarbiter.io.utils import mock_raw_report_endpoints
import solarforecastarbiter.reference_forecasts.main as reference_forecasts
from solarforecastarbiter.validation import tasks as validation_tasks
import solarforecastarbiter.reports.main as reports
from solarforecastarbiter.reports import template


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.WARNING)
sentry_sdk.init(send_default_pii=False,
                release=f'solarforecastarbiter-core@{__version__}')
midnight = pd.Timestamp.utcnow().floor('1d')


def handle_exception(exc_type, exc_value, exc_traceback):  # pragma: no cover
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception",
                  exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


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
@click.option('--end', default=lambda: midnight,
              type=UTCTIMESTAMP,
              show_default='00:00:00 Today (UTC)',
              help='datetime to end validation at')
@click.option(
    '--only-missing/--not-only-missing',
    is_flag=True, default=True,
    help='Only apply validation to periods where daily validation is missing')
@click.argument('observation_id', nargs=-1)
def validate(verbose, user, password, start, end, base_url,
             only_missing, observation_id):
    """
    Run the validation tasks for a given set of observations
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    if not observation_id:
        logger.info(
            ('Validating observation data from %s to %s for all '
             'observations'), start, end)
        validation_tasks.fetch_and_validate_all_observations(
            token, start, end, only_missing=only_missing,
            base_url=base_url)
    else:
        logger.info(
            ('Validating observation data from %s to %s for '
             'observations:\n\t%s'), start, end, ','.join(observation_id))
        for obsid in observation_id:
            validation_tasks.fetch_and_validate_observation(
                token, obsid, start, end, only_missing=only_missing,
                base_url=base_url)


@cli.group(help=reference_data.CLI_DESCRIPTION)
def referencedata():
    pass  # pragma: no cover


network_opt = click.option(
    '--network', multiple=True,
    help="The networks to act on. Defaults to all.",
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
@click.option('--end', type=UTCTIMESTAMP,
              help='End time (ISO8601) to fetch data for. Default is now')
@click.option('--start', type=UTCTIMESTAMP,
              help=('Start time (ISO8601) to fetch data for. Default is'
                    ' max of last timestamp in API and end - 7 days'))
@click.option('--gaps-only', is_flag=True,
              help='Only fetch and upload gaps in the observation values')
def referencedata_update(verbose, user, password, base_url, network, start,
                         end, gaps_only):
    """
    Updates reference data for the given period.
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    reference_data.update_reference_observations(token, start, end, network,
                                                 base_url, gaps_only=gaps_only)


@cli.command()
@common_options
@click.option('--provider', default='Reference',
              help='Provider that all observations should belong to')
def referenceaggregates(verbose, user, password, base_url, provider):
    """
    Updates reference data for the given period.
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    reference_aggregates.make_reference_aggregates(token, provider, base_url)


@cli.command()
@click.option('-v', '--verbose', count=True,
              help='Increase logging verbosity')
@click.option('--chunksize', default=128,
              help='Size of a chunk (in KB) to save at one time')
@click.option('--once', is_flag=True,
              help='Only get one forecast initialization time')
@click.option('--use-tmp', is_flag=True,
              help='Save grib files to /tmp')
@click.option('--netcdf-only', is_flag=True,
              help='Only convert files at save_directory to netcdf')
@click.option('--workers', type=int, default=1,
              help='Number of worker processes')
@click.argument('save_directory', type=click.Path(
    exists=True, writable=True, resolve_path=True, file_okay=False))
@click.argument('model', type=click.Choice([
    'gfs_0p25', 'nam_12km', 'rap', 'hrrr_hourly', 'hrrr_subhourly', 'gefs']))
def fetchnwp(verbose, chunksize, once, use_tmp, netcdf_only, workers,
             save_directory, model):
    """
    Retrieve weather forecasts with variables relevant to solar power
    from the NCEP NOMADS server. The utility function wgrib2 is
    required to convert these forecasts into netCDF format.
    """
    set_log_level(verbose)
    from solarforecastarbiter.io.fetch import nwp
    nwp.check_wgrib2()
    update_num_workers(workers)
    basepath = Path(save_directory)
    if netcdf_only:
        path_to_files = basepath
        if (
                not path_to_files.is_dir() or
                len(list(path_to_files.glob('*.grib2'))) == 0
        ):
            logger.error('%s is not a valid directory with grib files',
                         path_to_files)
            sys.exit(1)
        fut = asyncio.ensure_future(nwp.optimize_only(path_to_files, model))
    else:
        logger.info('Fetching NWP forecasts for %s', model)
        fut = asyncio.ensure_future(nwp.run(basepath, model, chunksize,
                                            once, use_tmp))

    loop = asyncio.get_event_loop()

    def bail(ecode):
        fut.cancel()
        sys.exit(ecode)

    loop.add_signal_handler(signal.SIGUSR1, partial(bail, 1))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, partial(bail, 0))

    loop.run_until_complete(fut)


@cli.group(help="Make reference forecasts")
def referencefx():
    pass  # pragma: no cover


@referencefx.group(name='nwp', help='Make NWP based forecasts.')
def ref_nwp():
    pass  # pragma: no cover


run_time = click.option('--run-time', type=UTCTIMESTAMP,
                        help='Run time for the forecasts',
                        show_default='now',
                        default=pd.Timestamp.utcnow())
itbuffer = click.option('--issue-time-buffer', type=str,
                        help=('Max time-delta between the run time and next '
                              'initialization time'),
                        show_default=True,
                        default='10min')
nwpdir = click.argument('nwp_directory', type=click.Path(
    exists=True, resolve_path=True, file_okay=False),
                        required=False)


@ref_nwp.command(name='latest')
@common_options
@run_time
@itbuffer
@nwpdir
def refnwp_latest(verbose, user, password, base_url, run_time,
                  issue_time_buffer, nwp_directory):

    """
    Make the reference NWP forecasts that should be issued around run_time
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    issue_buffer = pd.Timedelta(issue_time_buffer)
    nwp.set_base_path(nwp_directory)
    reference_forecasts.make_latest_nwp_forecasts(
        token, run_time, issue_buffer, base_url)


fxstart = click.option('--start', show_default='00:00:00 Yesterday (UTC)',
                       type=UTCTIMESTAMP,
                       default=lambda: midnight - pd.Timedelta(days=1),
                       help='Datetime to start filling forecasts at')
fxend = click.option('--end', default=lambda: midnight,
                     type=UTCTIMESTAMP,
                     show_default='00:00:00 Today (UTC)',
                     help='Datetime to end filling forecasts at')


@ref_nwp.command(name='fill')
@common_options
@fxstart
@fxend
@nwpdir
def refnwp_fill(verbose, user, password, base_url, start, end,
                nwp_directory):
    """Fill in any missing NWP forecasts from start to end"""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    nwp.set_base_path(nwp_directory)
    reference_forecasts.fill_nwp_forecast_gaps(token, start, end, base_url)


@referencefx.group(name='persistence', help='Make persistence forecasts')
def ref_persistence():
    pass  # pragma: no cover


prob_option = click.option('--probabilistic/--not-probabilistic', is_flag=True,
                           help='Make probabilistic persistence forecasts')


@ref_persistence.command(name='latest')
@common_options
@click.option('--max-run-time', type=UTCTIMESTAMP,
              help='Make forecasts up to this time',
              show_default='now',
              default=pd.Timestamp.utcnow())
@prob_option
def refpers_latest(verbose, user, password, base_url, max_run_time,
                   probabilistic):
    """Make all reference persistence forecasts that need to be made
    up to max_run_time"""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    if not probabilistic:
        reference_forecasts.make_latest_persistence_forecasts(
            token, max_run_time, base_url)
    else:
        reference_forecasts.make_latest_probabilistic_persistence_forecasts(
            token, max_run_time, base_url)


@ref_persistence.command(name='fill')
@common_options
@fxstart
@fxend
@prob_option
def refpers_fill(verbose, user, password, base_url, start, end,
                 probabilistic):
    """Fill in any gaps in the reference persistence forecasts from
    start to end"""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    if not probabilistic:
        reference_forecasts.fill_persistence_forecasts_gaps(
            token, start, end, base_url)
    else:
        reference_forecasts.fill_probabilistic_persistence_forecasts_gaps(
            token, start, end, base_url)


@cli.command()
@common_options
@run_time
@itbuffer
@nwpdir
def referencenwp(verbose, user, password, base_url, run_time,
                 issue_time_buffer, nwp_directory):
    """
    Make the reference NWP forecasts that should be issued around run_time
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    issue_buffer = pd.Timedelta(issue_time_buffer)
    nwp.set_base_path(nwp_directory)
    reference_forecasts.make_latest_nwp_forecasts(
        token, run_time, issue_buffer, base_url)


@cli.command()
@common_options
@click.option(
    '--format', default='detect',
    help=('Format of output file. "detect" tries to infer from '
          'the file extension of OUTPUT-FILE'),
    type=click.Choice(['detect', 'pdf', 'html'], case_sensitive=False)
)
@click.option(
    '--serialization-roundtrip', is_flag=True,
    help='Run the raw report through a mock API with serialization'
)
@click.option(
    '--orca-server-url', help=(
        'URL to the plotly orca server to generate PDF images if '
        'orca is not installed locally')
)
@click.argument(
    'report-file', type=click.Path(exists=True, resolve_path=True))
@click.argument(
    'output-file', type=click.Path(resolve_path=True))
def report(verbose, user, password, base_url, report_file, output_file,
           format, serialization_roundtrip, orca_server_url):
    """
    Make a report. See API documentation's POST reports section for
    REPORT_FILE requirements.
    """
    set_log_level(verbose)
    token = cli_access_token(user, password)
    with open(report_file) as f:
        metadata = json.load(f)
    session = APISession(token, base_url=base_url)
    report = session.process_report_dict(metadata)
    if orca_server_url is not None:
        import plotly.io as pio
        pio.orca.config.server_url = orca_server_url
    if serialization_roundtrip:
        with mock_raw_report_endpoints(base_url):
            session.create_report(report)
            reports.compute_report(token, 'no_id', base_url)
            full_report = session.get_report('no_id')
    else:
        data = reports.get_data_for_report(session, report)
        raw_report = reports.create_raw_report_from_data(report, data)
        full_report = report.replace(raw_report=raw_report, status='complete')
    # assumed dashboard url based on api url
    dash_url = base_url.replace('api', 'dashboard')
    if (
            (format == 'detect' and output_file.endswith('.html'))
            or format == 'html'
    ):
        html_report = template.render_html(
            full_report, dash_url,
            with_timeseries=True, body_only=False)
        with open(output_file, 'w') as f:
            f.write(html_report)
    elif (
            (format == 'detect' and output_file.endswith('.pdf'))
            or format == 'pdf'
    ):
        pdf_report = template.render_pdf(full_report, dash_url)
        with open(output_file, 'wb') as f:
            f.write(pdf_report)
    else:
        raise ValueError("Unable to detect format")


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('pytest_args', nargs=-1, type=click.UNPROCESSED)
def test(pytest_args):  # pragma: no cover
    """Test this installation of solarforecastarbiter"""
    import pytest
    ret_code = pytest.main(
        ['--pyargs', 'solarforecastarbiter'] + list(pytest_args))
    sys.exit(ret_code)


if __name__ == "__main__":  # pragma: no cover
    cli()
