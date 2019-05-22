import datetime as dt
from functools import partial
import logging


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
    cli.set_log_level(0)
    assert cli.logger.level == logging.WARNING
    cli.set_log_level(1)
    assert cli.logger.level == logging.INFO
    cli.set_log_level(2)
    assert cli.logger.level == logging.DEBUG


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


@pytest.mark.parametrize('val', [
    '20190101T000Z',
    '2019-03-0403:04:58-0700',
    "'2019-03-32 12:00'"
])
def test_utctimestamp_invalid(val):
    @click.command()
    @click.option('--timestamp', type=cli.UTCTIMESTAMP)
    def testtime(timestamp):
        if isinstance(timestamp, pd.Timestamp):
            print('OK')

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
                        new=dt.datetime(2019, 1, 2, 0, 0,
                                        tzinfo=dt.timezone.utc))
    runner = CliRunner()
    runner.invoke(cli.dailyvalidation, ['-u user', '-p pass'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   dt.datetime(2019, 1, 1, 0, 0,
                                               tzinfo=dt.timezone.utc),
                                   dt.datetime(2019, 1, 1, 23, 59, 59,
                                               tzinfo=dt.timezone.utc),
                                   'https://api.solarforecastarbiter.org')


def test_dailyvalidation_cmd_single(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.daily_single_observation_validation')  # NOQA
    mocker.patch.object(cli, 'midnight',
                        new=dt.datetime(2019, 1, 2, 0, 0,
                                        tzinfo=dt.timezone.utc))
    runner = CliRunner()
    runner.invoke(cli.dailyvalidation, ['-u user', '-p pass', 'OBS_ID'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   'OBS_ID',
                                   dt.datetime(2019, 1, 1, 0, 0,
                                               tzinfo=dt.timezone.utc),
                                   dt.datetime(2019, 1, 1, 23, 59, 59,
                                               tzinfo=dt.timezone.utc),
                                   'https://api.solarforecastarbiter.org')
