from pathlib import Path
import re
import uuid


import numpy as np
import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.io import api
from solarforecastarbiter.reports import main


EMPTY_DF = pd.DataFrame(columns=['value', 'quality_flag'],
                        index=pd.DatetimeIndex([], tz='UTC'))


@pytest.fixture()
def _test_data(report_objects, ref_forecast_id, remove_orca):
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
    data[ref_forecast_id] = data[forecast_0.forecast_id] + 1
    return data


@pytest.fixture()
def mock_data(mocker, _test_data):
    def get_data(id_, start, end, interval_label=None, **kwargs):
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


def test_get_data_for_report(mock_data, report_objects, mocker):
    report, observation, forecast_0, forecast_1, aggregate, forecast_agg = \
        report_objects
    session = api.APISession('nope')
    apply_obs = mocker.spy(main, 'apply_validation')
    data = main.get_data_for_report(session, report)
    assert apply_obs.call_count == 2
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)
    assert isinstance(data[aggregate], pd.DataFrame)
    assert isinstance(data[forecast_agg], pd.Series)
    get_forecast_values, get_observation_values, get_aggregate_values = \
        mock_data
    assert get_forecast_values.call_count == 4
    assert get_observation_values.call_count == 1
    assert get_aggregate_values.call_count == 1


@pytest.fixture()
def _test_event_data(event_report_objects, remove_orca):
    report, observation, forecast_0, forecast_1 = event_report_objects

    tz = "US/Pacific"
    index = pd.date_range(start="20200401T0000", end="20200404T2359",
                          freq="1h", tz=tz)
    obs_series = pd.DataFrame(data={
        "value": np.random.randint(0, 2, len(index), dtype=bool),
        "quality_flag": 2
    }, index=index)

    fx0_series = pd.Series(
        index=index, data=np.random.randint(0, 2, len(index), dtype=bool)
    )

    fx1_series = pd.Series(
        index=index, data=np.random.randint(0, 2, len(index), dtype=bool)
    )

    data = {
        observation.observation_id: obs_series,
        forecast_0.forecast_id: fx0_series,
        forecast_1.forecast_id: fx1_series,
    }
    return data


@pytest.fixture()
def mock_event_data(mocker, _test_event_data):
    def get_data(id_, start, end, interval_label=None, **kwargs):
        return _test_event_data[id_].loc[start:end]

    get_forecast_values = mocker.patch(
        'solarforecastarbiter.reports.main.APISession.get_forecast_values',
        side_effect=get_data)
    get_observation_values = mocker.patch(
        'solarforecastarbiter.reports.main.APISession.get_observation_values',
        side_effect=get_data)

    return get_forecast_values, get_observation_values


def test_get_data_for_report_event(mock_event_data, event_report_objects):
    report, observation, forecast_0, forecast_1 = event_report_objects
    session = api.APISession('nope')
    data = main.get_data_for_report(session, report)
    assert isinstance(data[observation], pd.DataFrame)
    assert isinstance(data[forecast_0], pd.Series)
    assert isinstance(data[forecast_1], pd.Series)
    get_forecast_values, get_observation_values = mock_event_data
    assert get_forecast_values.call_count == 2
    assert get_observation_values.call_count == 1


def test_get_version():
    vers = main.get_versions()
    assert {v[0] for v in vers} > {'solarforecastarbiter', 'python'}


def test_infer_timezone(report_objects):
    report = report_objects[0]
    assert main.infer_timezone(report.report_parameters) == "Etc/GMT+7"


def test_infer_timezone_agg(report_objects, single_forecast_aggregate,
                            single_forecast_observation):
    report_params = report_objects[0].report_parameters
    rp = report_params.replace(object_pairs=(single_forecast_aggregate,
                                             single_forecast_observation))
    assert main.infer_timezone(rp) == 'America/Denver'


def test_create_raw_report_from_data(mocker, report_objects, _test_data):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = _test_data[fxobs.forecast.forecast_id]
        if fxobs.reference_forecast is not None:
            data[fxobs.reference_forecast] = _test_data[
                fxobs.reference_forecast.forecast_id]
        if isinstance(fxobs, datamodel.ForecastAggregate):
            data[fxobs.aggregate] = _test_data[fxobs.aggregate.aggregate_id]
        else:
            data[fxobs.observation] = _test_data[
                fxobs.observation.observation_id]

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0


def test_create_raw_report_from_data_no_fx(mocker, report_objects, _test_data,
                                           caplog):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = EMPTY_DF['value']
        if fxobs.reference_forecast is not None:
            data[fxobs.reference_forecast] = _test_data[
                fxobs.reference_forecast.forecast_id]
        if isinstance(fxobs, datamodel.ForecastAggregate):
            data[fxobs.aggregate] = _test_data[fxobs.aggregate.aggregate_id]
        else:
            data[fxobs.observation] = _test_data[
                fxobs.observation.observation_id]

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0
    assert len(caplog.records) == 0


def test_create_raw_report_from_data_no_obs(mocker, report_objects,
                                            _test_data, caplog):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = _test_data[fxobs.forecast.forecast_id]
        if fxobs.reference_forecast is not None:
            data[fxobs.reference_forecast] = _test_data[
                fxobs.reference_forecast.forecast_id]
        data[fxobs.data_object] = EMPTY_DF

    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0
    assert len(caplog.records) == 0


def test_create_raw_report_from_data_no_data(mocker, report_objects, caplog):
    report = report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = EMPTY_DF['value']
        if fxobs.reference_forecast is not None:
            data[fxobs.reference_forecast] = EMPTY_DF['value']
        data[fxobs.data_object] = EMPTY_DF
    raw = main.create_raw_report_from_data(report, data)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0
    assert len(caplog.records) == 0


def test_create_raw_report_from_data_event(mocker, event_report_objects,
                                           _test_event_data):
    report = event_report_objects[0]
    data = {}
    for fxobs in report.report_parameters.object_pairs:
        data[fxobs.forecast] = _test_event_data[fxobs.forecast.forecast_id]
        data[fxobs.observation] = _test_event_data[
            fxobs.observation.observation_id
        ]

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


def test_capture_report_failure_is_valid_datamodel(mocker):
    api_post = mocker.patch('solarforecastarbiter.io.api.APISession.post')

    def fail():
        raise TypeError()

    session = api.APISession('nope')
    failwrap = main.capture_report_failure('report_id', session)
    with pytest.raises(TypeError):
        failwrap(fail)()
    raw = datamodel.RawReport.from_dict(api_post.call_args_list[0][1]['json'])
    assert 'CRITICAL' == raw.messages[0].level


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


def test_compute_report_request_mock(
        mocker, report_objects, mock_data, requests_mock):
    report = report_objects[0]
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_report',
        return_value=report)
    requests_mock.register_uri(
        'POST', re.compile('.*/reports/.*/values'),
        text=lambda *x: str(uuid.uuid1()))
    rep_post = requests_mock.register_uri(
        'POST', re.compile('.*/reports/.*/raw')
    )
    status_up = requests_mock.register_uri(
        'POST', re.compile('.*/reports/.*/status')
    )
    raw = main.compute_report('nope', report.report_id)
    assert isinstance(raw, datamodel.RawReport)
    assert len(raw.plots.figures) > 0
    assert rep_post.called
    assert 'complete' in status_up.last_request.path


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
