from pathlib import Path
import re
import uuid


import datetime as dt
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


def test_create_raw_report_from_data_event(site_metadata):

    tz = "America/Phoenix"
    start = pd.Timestamp("20200401T00", tz=tz)
    end = pd.Timestamp("20200404T2359", tz=tz)

    obs = datamodel.Observation(
        site=site_metadata,
        name="dummy obs",
        uncertainty=1,
        interval_length=pd.Timedelta("15min"),
        interval_value_type="instantaneous",
        variable="event",
        interval_label="event",
    )

    fx0 = datamodel.EventForecast(
        site=site_metadata,
        name="dummy fx",
        interval_length=pd.Timedelta("15min"),
        interval_value_type="instantaneous",
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta("1h"),
        run_length=pd.Timedelta("1h"),
        variable="event",
        interval_label="event",
    )

    fx1 = datamodel.EventForecast(
        site=site_metadata,
        name="dummy fx 2",
        interval_length=pd.Timedelta("15min"),
        interval_value_type="instantaneous",
        issue_time_of_day=dt.time(hour=5),
        lead_time_to_start=pd.Timedelta("1h"),
        run_length=pd.Timedelta("1h"),
        variable="event",
        interval_label="event",
    )

    fxobs0 = datamodel.ForecastObservation(observation=obs, forecast=fx0)
    fxobs1 = datamodel.ForecastObservation(observation=obs, forecast=fx1)

    quality_flag_filter = datamodel.QualityFlagFilter(
        ("USER FLAGGED", "NIGHTTIME")
    )

    timeofdayfilter = datamodel.TimeOfDayFilter((dt.time(12, 0),
                                                 dt.time(14, 0)))

    report_params = datamodel.ReportParameters(
        name="",
        start=start,
        end=end,
        object_pairs=(fxobs0, fxobs1),
        metrics=("pod", "far", "pofd"),
        categories=("total", "hour"),
        filters=(quality_flag_filter, timeofdayfilter),
    )

    report = datamodel.Report(
        report_id="f00-ba7",
        report_parameters=report_params
    )

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
