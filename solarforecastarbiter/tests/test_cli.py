import asyncio
from functools import partial
import logging
from pathlib import Path
import re
import tempfile


import click
from click.testing import CliRunner
import pandas as pd
import pytest
import requests


from solarforecastarbiter import cli, __version__

from solarforecastarbiter.conftest import mark_skip_pdflatex


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


def test_validate_cmd_all(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.fetch_and_validate_all_observations')  # NOQA
    mocker.patch.object(cli, 'midnight',
                        new=pd.Timestamp('2019-01-02T00:00:00Z'))
    runner = CliRunner()
    runner.invoke(cli.validate, ['-u user', '-p pass'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   pd.Timestamp('2019-01-01T00:00Z'),
                                   pd.Timestamp('2019-01-02T00:00Z'))
    assert mocked.call_args[1] == {
        'only_missing': True,
        'base_url': 'https://api.solarforecastarbiter.org'}


def test_validate_cmd_single(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.validation.tasks.fetch_and_validate_observation')  # NOQA
    mocker.patch.object(cli, 'midnight',
                        new=pd.Timestamp('2019-01-02T00:00:00Z'))
    runner = CliRunner()
    runner.invoke(cli.validate, ['-u user', '-p pass', 'OBS_ID'])
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN',
                                   'OBS_ID',
                                   pd.Timestamp('2019-01-01T00:00Z'),
                                   pd.Timestamp('2019-01-02T00:00Z'))
    assert mocked.call_args[1] == {
        'only_missing': True,
        'base_url': 'https://api.solarforecastarbiter.org'}


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
    res = runner.invoke(cli.referencedata_update,
                        ['-u user', '-p pass', '--start=20190101T0000Z',
                         '--end=20190101T235959Z'])
    assert res.exit_code == 0
    assert mocked.called
    assert mocked.call_args[0][:3] == ('TOKEN', pd.Timestamp('20190101T0000Z'),
                                       pd.Timestamp('20190101T235959Z'))
    assert mocked.call_args[0][-1] == 'https://api.solarforecastarbiter.org'


def test_referenceaggregates(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_aggregates.make_reference_aggregates')  # NOQA
    runner = CliRunner()
    res = runner.invoke(cli.referenceaggregates,
                        ['-u user', '-p pass'])
    assert res.exit_code == 0
    assert mocked.called
    assert mocked.call_args[0] == ('TOKEN', 'Reference',
                                   'https://api.solarforecastarbiter.org')


def test_fetchnwp(mocker):
    mocker.patch('solarforecastarbiter.io.fetch.nwp.check_wgrib2')
    mocked = mocker.patch('solarforecastarbiter.io.fetch.nwp.run',
                          return_value=asyncio.sleep(0))
    runner = CliRunner()
    res = runner.invoke(cli.fetchnwp, ['/tmp', 'rap'])
    assert res.exit_code == 0
    assert mocked.called


def test_fetchnwp_netcdfonly(mocker):
    mocker.patch('solarforecastarbiter.io.fetch.nwp.check_wgrib2')
    mocked = mocker.patch('solarforecastarbiter.io.fetch.nwp.optimize_only',
                          return_value=asyncio.sleep(0))
    runner = CliRunner()
    with tempfile.NamedTemporaryFile(suffix='.grib2', dir='/tmp'):
        res = runner.invoke(cli.fetchnwp, ['--netcdf-only', '/tmp', 'rap'])
    assert res.exit_code == 0
    assert mocked.called


def test_fetchnwp_netcdfonly_nogrib(mocker):
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
        assert cli.nwp.BASE_PATH == str(Path(tmpdir).resolve())
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('20190501T1200Z'),
                              pd.Timedelta('2h'), mocker.ANY)


def test_report(cli_token, mocker, report_objects):
    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report_objects[0].replace(status=''))
    index = pd.date_range(
        start="2019-04-01T00:00:00Z", end="2019-04-04T23:59:00Z",
        freq='1h')
    data = pd.Series(1., index=index)
    obs = pd.DataFrame({'value': data, 'quality_flag': 2})
    ref_fx = \
        report_objects[0].report_parameters.object_pairs[1].reference_forecast
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value={report_objects[2]: data,
                               report_objects[3]: data,
                               ref_fx: data,
                               report_objects[1]: obs,
                               report_objects[4]: obs,
                               report_objects[5]: data})
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + '/test_out.html'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', infile, outfile])
    assert res.exit_code == 0


def test_report_roundtrip(cli_token, mocker, various_report_objects_data,
                          requests_mock):
    report_objects, report_data = various_report_objects_data
    # mock all endpoints
    base_url = 'http://baseurl'
    requests_mock.register_uri(['POST', 'GET'], re.compile(base_url + '/.*'))
    # faster to not make real pdf
    from solarforecastarbiter.reports.figures.plotly_figures import fail_pdf
    mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.output_pdf',
        return_value=fail_pdf)
    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report_objects[0].replace(status='complete'))
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value=report_data)
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + '/test_out.html'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', f'--base-url={base_url}',
                             '--serialization-roundtrip',
                             infile, outfile])
        assert res.exit_code == 0


@mark_skip_pdflatex
def test_report_pdf(cli_token, mocker, report_objects, remove_orca):
    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report_objects[0].replace(status=''))
    index = pd.date_range(
        start="2019-04-01T00:00:00Z", end="2019-04-04T23:59:00Z",
        freq='1h')
    data = pd.Series(1., index=index)
    obs = pd.DataFrame({'value': data, 'quality_flag': 2})
    ref_fx = \
        report_objects[0].report_parameters.object_pairs[1].reference_forecast
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value={report_objects[2]: data,
                               report_objects[3]: data,
                               ref_fx: data,
                               report_objects[1]: obs,
                               report_objects[4]: obs,
                               report_objects[5]: data})
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + '/test_out.pdf'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', infile, outfile])
        assert res.exit_code == 0
        with open(outfile, 'rb') as f:
            assert f.read(4) == b'%PDF'


@pytest.mark.parametrize('outfmt', [
    'html',
    pytest.param('pdf', marks=mark_skip_pdflatex)
])
def test_report_probabilistic(
        cli_token, mocker, cdf_and_cv_report_objects, cdf_and_cv_report_data,
        outfmt):
    report, *_ = cdf_and_cv_report_objects

    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report.replace(status=''))
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value=cdf_and_cv_report_data)
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + f'/test_out.{outfmt}'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', infile, outfile])
    assert res.exit_code == 0


@pytest.mark.parametrize('outfmt', [
    'html',
    pytest.param('pdf', marks=mark_skip_pdflatex)
])
def test_report_probabilistic_xy(
        cli_token, mocker, cdf_and_cv_report_objects_xy,
        cdf_and_cv_report_data_xy, outfmt):
    report, *_ = cdf_and_cv_report_objects_xy

    mocker.patch('solarforecastarbiter.cli.APISession.process_report_dict',
                 return_value=report.replace(status=''))
    mocker.patch('solarforecastarbiter.cli.reports.get_data_for_report',
                 return_value=cdf_and_cv_report_data_xy)
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + f'/test_out.{outfmt}'
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', infile, outfile])
    assert res.exit_code == 0


@pytest.mark.parametrize('format_,res_code,suffix,called', [
    ('pdf', 0, '.pdf', 'pdf'),
    ('pdf', 0, '.pnotf', 'pdf'),
    ('detect', 0, '.pdf', 'pdf'),
    ('detect', 1, '.json', ''),
    ('html', 0, '.html', 'html'),
    ('html', 0, '.pnotf', 'html'),
    ('detect', 0, '.html', 'html'),
])
def test_report_format(cli_token, mocker, format_, res_code, suffix, called):
    mocker.patch('solarforecastarbiter.cli.APISession')
    html = mocker.patch('solarforecastarbiter.cli.template.render_html',
                        return_value='html')
    pdf = mocker.patch('solarforecastarbiter.cli.template.render_pdf',
                       return_value=b'%PDF')
    mocks = {'html': html, 'pdf': pdf}
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tmpdir + '/report.json'
        with open(infile, 'w') as f:
            f.write('{}')
        outfile = tmpdir + '/test_out' + suffix
        res = runner.invoke(cli.report,
                            ['-u user', '-p pass', f'--format={format_}',
                             infile, outfile])
        assert res.exit_code == res_code
        if called:
            mocks[called].assert_called()


def test_refnwp_latest(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.make_latest_nwp_forecasts')  # NOQA
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        res = runner.invoke(cli.refnwp_latest,
                            ['-u user', '-p pass', '--run-time=20190501T1200Z',
                             '--issue-time-buffer=2h',
                             tmpdir])
        assert cli.nwp.BASE_PATH == str(Path(tmpdir).resolve())
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('20190501T1200Z'),
                              pd.Timedelta('2h'), mocker.ANY)


def test_refnwp_fill(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.fill_nwp_forecast_gaps')  # NOQA
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        res = runner.invoke(cli.refnwp_fill,
                            ['-u user', '-p pass', '--start=2020-01-02T12:00Z',
                             '--end=2020-04-01T23:23Z',
                             tmpdir])
        assert cli.nwp.BASE_PATH == str(Path(tmpdir).resolve())
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('2020-01-02T12:00Z'),
                              pd.Timestamp('2020-04-01T23:23Z'), mocker.ANY)


def test_refpers_latest(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.make_latest_persistence_forecasts')  # NOQA
    runner = CliRunner()
    res = runner.invoke(cli.refpers_latest,
                        ['-u user', '-p pass',
                         '--max-run-time=20190501T1200Z'])
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('20190501T1200Z'),
                              mocker.ANY)


def test_refpers_latest_prob(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.make_latest_probabilistic_persistence_forecasts')  # NOQA
    runner = CliRunner()
    res = runner.invoke(cli.refpers_latest,
                        ['-u user', '-p pass', '--probabilistic',
                         '--max-run-time=20190501T1200Z'])
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('20190501T1200Z'),
                              mocker.ANY)


def test_refpers_fill(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.fill_persistence_forecasts_gaps')  # NOQA
    runner = CliRunner()
    res = runner.invoke(cli.refpers_fill,
                        ['-u user', '-p pass', '--start=2020-01-02T12:00Z',
                         '--end=2020-04-01T23:23Z'])
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('2020-01-02T12:00Z'),
                              pd.Timestamp('2020-04-01T23:23Z'), mocker.ANY)


def test_refpers_fill_prob(cli_token, mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.cli.reference_forecasts.fill_probabilistic_persistence_forecasts_gaps')  # NOQA
    runner = CliRunner()
    res = runner.invoke(cli.refpers_fill,
                        ['-u user', '-p pass', '--start=2020-01-02T12:00Z',
                         '--end=2020-04-01T23:23Z', '--probabilistic'])
    assert res.exit_code == 0
    mocked.assert_called_with('TOKEN', pd.Timestamp('2020-01-02T12:00Z'),
                              pd.Timestamp('2020-04-01T23:23Z'), mocker.ANY)
