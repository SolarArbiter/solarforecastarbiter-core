import logging
from pathlib import Path


import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api
from solarforecastarbiter.reports import main


EMPTY_DF = pd.DataFrame(columns=['value', 'quality_flag'],
                        index=pd.DatetimeIndex([], tz='UTC'))


@pytest.fixture()
def _test_data(report_objects):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    base = Path(__file__).resolve().parent
    data = {
        observation.observation_id: pd.read_csv(
            base / 'observation_values.csv', index_col='timestamp',
            parse_dates=True),
        forecast_0.forecast_id: pd.read_csv(
            base / 'forecast_0_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
        forecast_1.forecast_id: pd.read_csv(
            base / 'forecast_1_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
        aggregate.aggregate_id: pd.read_csv(
            base / 'observation_values.csv', index_col='timestamp',
            parse_dates=True),
        forecast_agg.forecast_id: pd.read_csv(
            base / 'forecast_0_values.csv', header=None, parse_dates=True,
            names=['timestamp', 'value'], index_col='timestamp')['value'],
    }
    return data


@pytest.fixture()
def mock_data(mocker, _test_data):
    def get_data(id_, start, end, interval_label=None):
        return _test_data[id_].loc[start:end]

    get_forecast_values = mocker.patch(
        'solarforecastarbiter.reports.main.APISession.get_forecast_values',
        side_effect=get_data)
    get_observation_values = mocker.patch(
        'solarforecastarbiter.reports.main.APISession.get_observation_values',
        side_effect=get_data)
    get_aggregate_values = mocker.patch(
        'solarforecastarbiter.reports.main.APISession.get_aggregate_values',
        side_effect=get_data)

    return get_forecast_values, get_observation_values, get_aggregate_values


def test_get_data_for_report(mock_data, report_objects):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)
    assert isinstance(data[aggregate], pd.DataFrame)
    assert isinstance(data[forecast_agg], pd.Series)
    get_forecast_values, get_observation_values, get_aggregate_values = \
        mock_data
    assert get_forecast_values.call_count == 3
    assert get_observation_values.call_count == 1
    assert get_aggregate_values.call_count == 1


def test_get_version():
    vers = main.get_versions()
    assert set(vers.keys()) > {'solarforecastarbiter', 'python'}


def test_infer_timezone(report_objects):
    report = report_objects[0]
    assert main.infer_timezone(report.report_parameters) == "Etc/GMT+7"


def test_listhandler():
    logger = logging.getLogger('testlisthandler')
    handler = main.ListHandler()
    logger.addHandler(handler)
    logger.setLevel('DEBUG')
    logger.warning('Test it')
    logger.debug('What?')
    out = handler.export_records()
    assert len(out) == 1
    assert out[0].message == 'Test it'
    assert len(handler.export_records(logging.DEBUG)) == 2


def test_listhandler_recreate():
    logger = logging.getLogger('testlisthandler')
    handler = main.ListHandler()
    logger.addHandler(handler)
    logger.setLevel('DEBUG')
    logger.warning('Test it')
    logger.debug('What?')
    out = handler.export_records()
    assert len(out) == 1
    assert out[0].message == 'Test it'
    assert len(handler.export_records(logging.DEBUG)) == 2

    l2 = logging.getLogger('testlist2')
    h2 = main.ListHandler()
    l2.addHandler(h2)
    l2.error('Second fail')
    out = h2.export_records()
    assert len(out) == 1
    assert out[0].message == 'Second fail'


def test_hijack_loggers(mocker):
    old_handler = mocker.MagicMock()
    new_handler = mocker.MagicMock()
    mocker.patch('solarforecastarbiter.reports.main.ListHandler',
                 return_value=new_handler)
    logger = logging.getLogger('testhijack')
    logger.addHandler(old_handler)
    assert logger.handlers[0] == old_handler
    with main.hijack_loggers(['testhijack']):
        assert logger.handlers[0] == new_handler
    assert logger.handlers[0] == old_handler


def test_create_raw_report_from_data(mocker, report_objects, _test_data):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = _test_data[fxobs.forecast.forecast_id]
        if isinstance(fxobs, datamodel.ForecastAggregate):
            data[fxobs.aggregate] = _test_data[fxobs.aggregate.aggregate_id]
        else:
            data[fxobs.observation] = _test_data[
                fxobs.observation.observation_id]

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


def test_create_raw_report_from_data_no_fx(mocker, report_objects, _test_data):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = EMPTY_DF['value']
        if isinstance(fxobs, datamodel.ForecastAggregate):
            data[fxobs.aggregate] = _test_data[fxobs.aggregate.aggregate_id]
        else:
            data[fxobs.observation] = _test_data[
                fxobs.observation.observation_id]

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


def test_create_raw_report_from_data_no_obs(mocker, report_objects,
                                            _test_data):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = _test_data[fxobs.forecast.forecast_id]
        data[fxobs.data_object] = EMPTY_DF

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


def test_create_raw_report_from_data_no_data(mocker, report_objects):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = EMPTY_DF['value']
        data[fxobs.data_object] = EMPTY_DF
    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


def test_capture_report_failure(mocker):
    api_post = mocker.patch('solarforecastarbiter.io.api.APISession.post')

    def fail():
        raise TypeError()

    session = api.APISession('nope')
    failwrap = main.capture_report_failure('report_id', session)
    with pytest.raises(TypeError):
        failwrap(fail)()
    assert 'Critical ' in api_post.call_args_list[0][1]['json']['messages'][0]['message']  # NOQA
    assert api_post.call_args_list[1][0][0] == '/reports/report_id/status/failed'  # NOQA


def test_capture_report_failure_msg(mocker):
    api_post = mocker.patch('solarforecastarbiter.io.api.APISession.post')

    def fail():
        raise TypeError()

    session = api.APISession('nope')
    failwrap = main.capture_report_failure('report_id', session)
    err_msg = 'Super bad error message'
    with pytest.raises(TypeError):
        failwrap(fail, err_msg=err_msg)()
    assert api_post.call_args_list[0][1]['json']['messages'][0]['message'] == err_msg  # NOQA
    assert api_post.call_args_list[1][0][0] == '/reports/report_id/status/failed'  # NOQA


def test_compute_report(mocker, report_objects, mock_data):
    report = report_objects[0]
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_report',
        return_value=report)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_raw_report'
    )
    raw = main.compute_report('nope', report.report_id)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


@pytest.fixture()
def assert_post_called(mocker):
    post_report = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_raw_report')
    yield
    assert post_report.called
    assert post_report.call_args[0][0] == 'repid'
    assert post_report.call_args[0][-1] == 'failed'
    assert len(post_report.call_args[0][1].messages[0].message) > 0


def test_compute_report_get_report_fail(mocker, assert_post_called):
    mocker.patch('solarforecastarbiter.io.api.APISession.get_report',
                 side_effect=TypeError)
    with pytest.raises(TypeError):
        main.compute_report('nope', 'repid')


@pytest.fixture()
def get_report_mocked(mocker, report_objects):
    mocker.patch('solarforecastarbiter.io.api.APISession.get_report',
                 return_value=report_objects[0])


def test_compute_report_get_data_fail(
        mocker, get_report_mocked, assert_post_called):
    mocker.patch('solarforecastarbiter.reports.main.get_data_for_report',
                 side_effect=TypeError)
    with pytest.raises(TypeError):
        main.compute_report('nope', 'repid')


def test_compute_report_compute_fail(mocker, get_report_mocked, mock_data,
                                     assert_post_called):
    mocker.patch(
        'solarforecastarbiter.reports.main.create_raw_report_from_data',
        side_effect=TypeError)
    with pytest.raises(TypeError):
        main.compute_report('nope', 'repid')


def test_report_to_html_body(report_with_raw):
    out = main.report_to_html_body(report_with_raw)
    assert len(out) > 0
    assert report_with_raw.report_parameters.name in out


def test_report_to_pdf(report_with_raw):
    with pytest.raises(NotImplementedError):
        main.report_to_pdf(report_with_raw)


def test_report_to_jupyter(report_with_raw):
    with pytest.raises(NotImplementedError):
        main.report_to_jupyter(report_with_raw)
