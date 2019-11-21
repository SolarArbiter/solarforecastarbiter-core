import asyncio
from functools import partial
import logging
from pathlib import Path
import tempfile


import click
from click.testing import CliRunner
import pandas as pd
import pytest
import requests


from solarforecastarbiter import cli, __version__


@pytest.fixture()
def cli_token(mocker):
    return mocker.patch(
        'solarforecastarbiter.cli.request_cli_access_token',
        return_value='TOKEN')


def test_cli_access_token(cli_token):
    token = cli.cli_access_token('user', 'pass')
    assert token == 'TOKEN'


def test_cli_access_token_err(mocker):
    mocker.patch(
        'solarforecastarbiter.io.api.request_cli_access_token',
        side_effect=requests.HTTPError)
    with pytest.raises(SystemExit):
        cli.cli_access_token('user', 'pass')


def test_set_log_level():
    root_logger = logging.getLogger()
    cli.set_log_level(0)
    assert root_logger.level == logging.WARNING
    cli.set_log_level(1)
    assert root_logger.level == logging.INFO
    cli.set_log_level(2)
    assert root_logger.level == logging.DEBUG


@pytest.mark.parametrize('val', [
    '20190101T0000Z',
    '2019-03-04T03:04:58-0700',
    "'2019-03-12 12:00'",
    'now'
])
def test_utctimestamp(val):
    @click.command()
    @click.option('--timestamp', type=cli.UTCTIMESTAMP)
    def testtime(timestamp):
        if isinstance(timestamp, pd.Timestamp):
            print('OK')

    runner = CliRunner()
    res = runner.invoke(testtime, f'--timestamp {val}')
    assert res.output == 'OK\n'


def test_utctimestamp_none():
    @click.command()
    @click.option('--timestamp', type=cli.UTCTIMESTAMP)
    def testtime(timestamp):
        if timestamp is None:
            print('OK')

    runner = CliRunner()
    res = runner.invoke(testtime)
    assert res.output == 'OK\n'


@pytest.mark.parametrize('val', [
    '20190101T000Z',
    '2019-03-0403:04:58-0700',
    "'2019-03-32 12:00'"
])
def test_utctimestamp_invalid(val):
    @click.command()
    @click.option('--timestamp', type=cli.UTCTIMESTAMP)
    def testtime(timestamp):
        return  # pragma: no cover

    runner = CliRunner()
    res = runner.invoke(testtime, f'--timestamp {val}')
    assert res.output.endswith('cannot be converted into a Pandas.Timestamp\n')


def test_version():
    runner = CliRunner()
    res = runner.invoke(cli.cli, ['--version'])
    assert res.output.rstrip('\n').endswith(__version__)


@pytest.fixture()
def common_runner():
    @click.command()
    @cli.common_options
    def testit(**kwargs):
        # make sure common options added
        if all([kw in kwargs for kw in ('verbose', 'user', 'password')]):
            print('OK')

    runner = CliRunner()
    return partial(runner.invoke, testit)


@pytest.mark.parametrize('opts,envs', [
    (['-u test', '-p tt'], {}),
    (['-u test', '-p tt', '-vv'], {}),
    ([], {'SFA_API_PASSWORD': 'testpass', 'SFA_API_USER': 'user'}),
    (['-p pass'], {'SFA_API_USER': 'user'}),
    (['-u user'], {'SFA_API_PASSWORD': 'testpass'})
])
def test_common_options(common_runner, opts, envs, monkeypatch):
    for k, v in envs.items():
        monkeypatch.setenv(k, v)
    res = common_runner(opts)
    assert res.output == 'OK\n'


def test_dailyvalidation_cmd_all(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.daily_observation_validation')
    mocker.patch.object(cli, 'midnight',
                        new=pd.Timestamp('2019-01-02T00:00:00Z'))
    runner = CliRunner()
    runner.invoke(cli.dailyvalidation, ['-u user', '-p pass'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   pd.Timestamp('2019-01-01T00:00Z'),
                                   pd.Timestamp('2019-01-01T23:59:59Z'),
                                   'https://api.solarforecastarbiter.org')


def test_dailyvalidation_cmd_single(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.daily_single_observation_validation')  # NOQA
    mocker.patch.object(cli, 'midnight',
                        new=pd.Timestamp('2019-01-02T00:00:00Z'))
    runner = CliRunner()
    runner.invoke(cli.dailyvalidation, ['-u user', '-p pass', 'OBS_ID'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   'OBS_ID',
                                   pd.Timestamp('2019-01-01T00:00Z'),
                                   pd.Timestamp('2019-01-01T23:59:59Z'),
                                   'https://api.solarforecastarbiter.org')


def test_referencedata_init(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_data.initialize_reference_metadata_objects')  # NOQA
    runner = CliRunner()
    runner.invoke(cli.referencedata_init, ['-u user', '-p pass'])
    assert mocked.called
    assert mocked.call_args[0][0] == 'TOKEN'
    assert mocked.call_args[0][-1] == 'https://api.solarforecastarbiter.org'


def test_referencedata_update(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_data.update_reference_observations')  # NOQA
    runner = CliRunner()
    runner.invoke(cli.referencedata_update,
                  ['-u user', '-p pass', '20190101T0000Z', '20190101T235959Z'])
    assert mocked.called
    assert mocked.call_args[0][:3] == ('TOKEN', pd.Timestamp('20190101T0000Z'),
                                       pd.Timestamp('20190101T235959Z'))
    assert mocked.call_args[0][-1] == 'https://api.solarforecastarbiter.org'


def test_fetchnwp(mocker):
    mocker.patch('solarforecastarbiter.cli.start_cluster')
    mocker.patch('solarforecastarbiter.io.fetch.nwp.check_wgrib2')
    mocked = mocker.patch('solarforecastarbiter.io.fetch.nwp.run',
                          return_value=asyncio.sleep(0))
    runner = CliRunner()
    res = runner.invoke(cli.fetchnwp, ['/tmp', 'rap'])
    assert res.exit_code == 0
    assert mocked.called


def test_fetchnwp_netcdfonly(mocker):
    mocker.patch('solarforecastarbiter.cli.start_cluster')
    mocker.patch('solarforecastarbiter.io.fetch.nwp.check_wgrib2')
    mocked = mocker.patch('solarforecastarbiter.io.fetch.nwp.optimize_only',
                          return_value=asyncio.sleep(0))
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.grib2', dir='/tmp'):
        res = runner.invoke(cli.fetchnwp, ['--netcdf-only', '/tmp', 'rap'])
    assert res.exit_code == 0
    assert mocked.called


def test_fetchnwp_netcdfonly_nogrib(mocker):
    mocker.patch('solarforecastarbiter.cli.start_cluster')
    mocker.patch('solarforecastarbiter.io.fetch.nwp.check_wgrib2')
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        res = runner.invoke(cli.fetchnwp, ['--netcdf-only', tmpdir, 'rap'])
    assert res.exit_code == 1


def test_reference_nwp(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.make_latest_nwp_forecasts')  # NOQA
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        res = runner.invoke(cli.referencenwp,
                            ['-u user', '-p pass', '--run-time=20190501T1200Z',
                             '--issue-time-buffer=2h',
                             tmpdir])
        assert cli.nwp.BASE_PATH.replace('/private', '') == tmpdir
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('20190501T1200Z'),
                              pd.Timedelta('2h'), mocker.ANY)


def test_report(cli_token, mocker, report_objects):
    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report_objects[0])
    index = pd.date_range(
        start="2019-04-01T00:00:00Z", end="2019-04-04T23:59:00Z",
        freq='1h')
    data = pd.Series([0] * len(index), index=index)
    obs = pd.DataFrame({'value': data, 'quality_flag': data + 3})
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value={report_objects[2]: data,
                               report_objects[3]: data,
                               report_objects[1]: obs,
                               report_objects[4]: obs,
                               report_objects[5]: data})
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = (Path(cli.__file__).resolve().parents[0] / 'tests/data' /
                  'report_metadata.json')
        outfile = tmpdir + '/test_out.html'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', str(infile), outfile])
    assert res.exit_code == 0
