import datetime as dt
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest


from solarforecastarbiter.datamodel import Observation
from solarforecastarbiter.validation import tasks, validator
from solarforecastarbiter.validation.quality_mapping import (
    LATEST_VERSION_FLAG, DESCRIPTION_MASK_MAPPING,
    DAILY_VALIDATION_FLAG)


@pytest.fixture(params=['beginning', 'ending', 'instant'])
def make_observation(single_site, request):
    def f(variable):
        return Observation(
            name='test', variable=variable,
            interval_value_type='interval_mean',
            interval_length=pd.Timedelta('1hr'), interval_label=request.param,
            site=single_site, uncertainty=0.1, observation_id='OBSID',
            provider='Organization 1', extra_parameters='')
    return f


@pytest.fixture()
def default_index(single_site):
    return [pd.Timestamp('2019-01-01T08:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T09:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T10:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T11:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T13:00:00', tz=single_site.timezone)]


@pytest.fixture()
def daily_index(single_site):
    out = pd.date_range(start='2019-01-01T08:00:00',
                        end='2019-01-01T19:00:00',
                        freq='1h',
                        tz=single_site.timezone)
    return out.append(
        pd.Index([pd.Timestamp('2019-01-02T09:00:00',
                               tz=single_site.timezone)]))


def nighttime_func_mask(obs, index):
    if obs.interval_label == 'beginning':
        flag = [0, 0, 0, 0, 0]
        func = 'check_day_night_interval'
    elif obs.interval_label == 'ending':
        flag = [1, 0, 0, 0, 0]
        func = 'check_day_night_interval'
    else:
        flag = [1, 0, 0, 0, 0]
        func = 'check_day_night'
    flag = pd.Series(flag, index=index, dtype='int')
    mask = flag * DESCRIPTION_MASK_MAPPING['NIGHTTIME']
    return func, mask


def nighttime_func_mask_daily(obs, index):
    if obs.interval_label == 'beginning':
        flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        func = 'check_day_night_interval'
    elif obs.interval_label == 'ending':
        flag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        func = 'check_day_night_interval'
    else:
        flag = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        func = 'check_day_night'
    flag = pd.Series(flag, index=index)
    mask = flag * DESCRIPTION_MASK_MAPPING['NIGHTTIME']
    return func, mask


def test_validate_ghi(mocker, make_observation, default_index):
    obs = make_observation('ghi')
    data = pd.Series([10, 1000, -100, 500, 300], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi']]
    flags = tasks.validate_ghi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                pd.Series([0, 1, 0, 1, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'],
                pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_validate_mostly_clear(mocker, make_observation):
    obs = make_observation('ghi').replace(interval_length=pd.Timedelta('5min'))
    index = pd.date_range(start='2019-04-01T11:00', freq='5min',
                          tz=obs.site.timezone, periods=11)
    data = pd.Series([742, 749, 756, 763, 769, 774, 779, 784, 789, 793, 700],
                     index=index)
    # only care about night_func here, so index just needs to be right length
    # because mask is not used.
    night_func, _ = nighttime_func_mask(obs, data.index[:5])
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi']]
    flags = tasks.validate_ghi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'],
                pd.Series([1] * 10 + [0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_apply_immediate_validation(
        mocker, make_observation, default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0), (175, 0), (150, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    val = tasks.apply_immediate_validation(obs, data)

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']
    ]
    assert_frame_equal(val, out)


def test_apply_immediate_validation_already_validated(
        mocker, make_observation, default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 18), (175, 18), (150, 18), (-1, 19), (1500, 18)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    val = tasks.apply_immediate_validation(obs, data)

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']
    ]
    assert_frame_equal(val, out)


@pytest.mark.parametrize('var', ['air_temperature', 'wind_speed', 'dni', 'dhi',
                                 'poa_global', 'relative_humidity'])
def test_apply_immediate_validation_other(
        mocker, make_observation, default_index, var):
    mock = mocker.MagicMock()
    mocker.patch.dict(
        'solarforecastarbiter.validation.tasks.IMMEDIATE_VALIDATION_FUNCS',
        {var: mock})
    obs = make_observation(var)
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    tasks.apply_immediate_validation(obs, data)
    assert mock.called


@pytest.mark.parametrize('var', ['availability', 'curtailment', 'event',
                                 'net_load'])
def test_apply_immediate_validation_defaults(
        mocker, make_observation, default_index, var):
    mock = mocker.spy(tasks, 'validate_defaults')
    obs = make_observation(var)
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])

    tasks.apply_immediate_validation(obs, data)
    assert mock.called


def test_fetch_and_validate_observation_ghi(mocker, make_observation,
                                            default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0), (175, 0), (150, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']
    ]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_fetch_and_validate_observation_ghi_nones(
        mocker, make_observation, default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(None, 1)] * 5, index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    base = (
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        LATEST_VERSION_FLAG
    )
    out['quality_flag'] = [
        base | night_mask[0],
        base,
        base,
        base,
        base | DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY']
    ]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_fetch_and_validate_observation_not_listed(mocker, make_observation,
                                                   default_index):
    obs = make_observation('curtailment')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        LATEST_VERSION_FLAG,
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_dni(mocker, make_observation, default_index):
    obs = make_observation('dni')
    data = pd.Series([10, 1000, -100, 500, 500], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_dni_limits_QCRad']]
    flags = tasks.validate_dni(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 0, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_dni(mocker, make_observation,
                                            default_index):
    obs = make_observation('dni')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED']]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_dhi(mocker, make_observation, default_index):
    obs = make_observation('dhi')
    data = pd.Series([10, 1000, -100, 200, 200], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_dhi_limits_QCRad']]
    flags = tasks.validate_dhi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_dhi(mocker, make_observation,
                                            default_index):
    obs = make_observation('dhi')
    data = pd.DataFrame(
        [(0, 0), (50, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED']]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_poa_global(mocker, make_observation, default_index):
    obs = make_observation('poa_global')
    data = pd.Series([10, 1000, -400, 300, 300], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_poa_clearsky']]
    flags = tasks.validate_poa_global(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_poa_global(mocker, make_observation,
                                                   default_index):
    obs = make_observation('poa_global')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_air_temp(mocker, make_observation, default_index):
    obs = make_observation('air_temperature')
    data = pd.Series([10, 1000, -400, 30, 20], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_temperature_limits']]
    flags = tasks.validate_air_temperature(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_air_temperature(
        mocker, make_observation, default_index):
    obs = make_observation('air_temperature')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (20, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        night_mask[0] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_wind_speed(mocker, make_observation, default_index):
    obs = make_observation('wind_speed')
    data = pd.Series([10, 1000, -400, 3, 20], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_wind_limits']]
    flags = tasks.validate_wind_speed(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_wind_speed(
        mocker, make_observation, default_index):
    obs = make_observation('wind_speed')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (15, 0), (1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        night_mask[0] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_relative_humidity(mocker, make_observation, default_index):
    obs = make_observation('relative_humidity')
    data = pd.Series([10, 101, -400, 60, 20], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_rh_limits']]
    flags = tasks.validate_relative_humidity(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_relative_humidity(
        mocker, make_observation, default_index):
    obs = make_observation('relative_humidity')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (15, 0), (40, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        night_mask[0] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_ac_power(mocker, make_observation, default_index):
    obs = make_observation('ac_power')
    data = pd.Series([0, 1, -1, 0.001, 0.001], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_ac_power_limits']]
    flags = tasks.validate_ac_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_ac_power(mocker, make_observation,
                                                 default_index):
    obs = make_observation('ac_power')
    data = pd.DataFrame(
        [(0, 0), (1, 0), (-1, 0), (0.001, 1), (0.001, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG
    ]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_dc_power(mocker, make_observation, default_index):
    obs = make_observation('dc_power')
    data = pd.Series([0, 1, -1, 0.001, 0.001], index=default_index)
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_dc_power_limits']]
    flags = tasks.validate_dc_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_dc_power(mocker, make_observation,
                                                 default_index):
    obs = make_observation('dc_power')
    data = pd.DataFrame(
        [(0, 0), (1, 0), (-1, 0), (0.001, 1), (0.001, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG
    ]
    assert post_mock.call_count == 2
    assert_frame_equal(post_mock.call_args_list[0][0][1], out[:-1])
    assert_frame_equal(post_mock.call_args_list[1][0][1], out[-1:])


def test_validate_daily_ghi(mocker, make_observation, daily_index):
    obs = make_observation('ghi')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [10, 1000, -100, 500, 300, 300, 300, 300, 175, 0, 100, 0, 0],
        index=daily_index)
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi',
                       'detect_stale_values',
                       'detect_interpolation']]
    flags = tasks.validate_daily_ghi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                pd.Series([0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'],
                pd.Series(0, index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['STALE VALUES'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'],
                )
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_ghi_daily(mocker, make_observation,
                                                  daily_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(10, 0), (1000, 0), (-100, 0), (500, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (175, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    BASE_FLAG = LATEST_VERSION_FLAG | DAILY_VALIDATION_FLAG

    out = data.copy()
    out['quality_flag'] = [
        night_mask[0] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | BASE_FLAG,
        night_mask[-4] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        BASE_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_fetch_and_validate_observation_ghi_zeros(mocker, make_observation,
                                                  daily_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0)] * 13,
        index=daily_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    base = (
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        LATEST_VERSION_FLAG | DAILY_VALIDATION_FLAG
    )
    out = data.copy()
    out['quality_flag'] = [
        LATEST_VERSION_FLAG | DAILY_VALIDATION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG |
        DAILY_VALIDATION_FLAG,
        base,
        base,
        base,
        base,
        base,
        base,
        base,
        base,
        base,
        base,
        base | DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY']
    ]
    out['quality_flag'] |= night_mask
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_validate_daily_dc_power(mocker, make_observation, daily_index):
    obs = make_observation('dc_power')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [0, 1000, -100, 500, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        index=daily_index)
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'detect_stale_values',
                       'detect_interpolation']]
    flags = tasks.validate_daily_dc_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['STALE VALUES'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'],
                )
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_dc_power_daily(
        mocker, make_observation, daily_index):
    obs = make_observation('dc_power')
    data = pd.DataFrame(
        [(10, 0), (1000, 0), (-100, 0), (500, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    BASE_FLAG = LATEST_VERSION_FLAG | DAILY_VALIDATION_FLAG
    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        night_mask[0] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        night_mask[-4] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        BASE_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_validate_daily_ac_power(mocker, make_observation, daily_index):
    obs = make_observation('ac_power')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [0, 100, -100, 100, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        index=daily_index)
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       night_func,
                       'detect_stale_values',
                       'detect_interpolation',
                       'detect_clipping']]
    flags = tasks.validate_daily_ac_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                night_mask,
                pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['STALE VALUES'],
                pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLIPPED VALUES']
                )
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_fetch_and_validate_observation_ac_power_daily(
        mocker, make_observation, daily_index):
    obs = make_observation('ac_power')
    data = pd.DataFrame(
        [(10, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])

    BASE_FLAG = LATEST_VERSION_FLAG | DAILY_VALIDATION_FLAG
    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        night_mask[0] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLIPPED VALUES'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLIPPED VALUES'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        night_mask[-4] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        BASE_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        BASE_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


@pytest.mark.parametrize('var', ['air_temperature', 'wind_speed', 'dni', 'dhi',
                                 'poa_global', 'relative_humidity', 'net_load',
                                 ])
def test_fetch_and_validate_observation_other(var, mocker, make_observation,
                                              daily_index):
    obs = make_observation(var)
    data = pd.DataFrame(
        [(0, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')
    validated = pd.Series(2, index=daily_index)
    validate_mock = mocker.MagicMock(return_value=validated)
    mocker.patch.dict(
        'solarforecastarbiter.validation.tasks.IMMEDIATE_VALIDATION_FUNCS',
        {var: validate_mock})
    tasks.fetch_and_validate_observation(
        '', obs.observation_id, data.index[0], data.index[-1])
    assert post_mock.called
    assert validate_mock.called


@pytest.mark.parametrize('var', ['air_temperature', 'wind_speed', 'dni', 'dhi',
                                 'poa_global', 'relative_humidity'])
def test_apply_daily_validation_other(
        mocker, make_observation, daily_index, var):
    mock = mocker.MagicMock()
    mocker.patch.dict(
        'solarforecastarbiter.validation.tasks.IMMEDIATE_VALIDATION_FUNCS',
        {var: mock})
    mocks = [mock,
             mocker.spy(tasks, '_validate_stale_interpolated')]
    obs = make_observation(var)
    data = pd.DataFrame({
        'value': [
            # 8     9     10   11   12  13    14   15  16  17  18  19  23
            10, 1900, -100, 500, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        'quality_flag': 0}, index=daily_index)
    out = tasks.apply_daily_validation(obs, data)
    assert (out['quality_flag'] | DAILY_VALIDATION_FLAG).all()
    for mock in mocks:
        assert mock.called


@pytest.mark.parametrize('var', ['net_load'])
def test_apply_daily_validation_defaults(
        mocker, make_observation, daily_index, var):
    mocks = [mocker.spy(tasks, 'validate_defaults'),
             mocker.spy(tasks, '_validate_stale_interpolated')]
    obs = make_observation(var)
    data = pd.DataFrame({
        'value': [
            # 8     9     10   11   12  13    14   15  16  17  18  19  23
            10, 1900, -100, 500, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        'quality_flag': 0}, index=daily_index)
    out = tasks.apply_daily_validation(obs, data)
    assert (out['quality_flag'] | DAILY_VALIDATION_FLAG).all()
    for mock in mocks:
        assert mock.called


def test_apply_daily_validation(mocker, make_observation, daily_index):
    obs = make_observation('ac_power')
    data = pd.DataFrame({
        'value': [
            # 8     9     10   11   12  13    14   15  16  17  18  19  23
            0, 100, -100, 100, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        'quality_flag': 94},
        index=daily_index)
    night_func, night_mask = nighttime_func_mask_daily(obs, data.index)

    out = tasks.apply_daily_validation(obs, data)
    qf = (pd.Series(LATEST_VERSION_FLAG, index=data.index),
          pd.Series(DAILY_VALIDATION_FLAG, index=data.index),
          pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    index=data.index) *
          DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
          night_mask,
          pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
                    index=data.index) *
          DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
          pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    index=data.index) *
          DESCRIPTION_MASK_MAPPING['STALE VALUES'],
          pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                    index=data.index) *
          DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'],
          pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                    index=data.index) *
          DESCRIPTION_MASK_MAPPING['CLIPPED VALUES']
          )
    exp = data.copy()
    exp['quality_flag'] = sum(qf)
    assert_frame_equal(exp, out)


def test_apply_daily_validation_not_enough(mocker, make_observation):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0)],
        index=pd.date_range(start='2019-01-01T0000Z',
                            end='2019-01-01T0100Z',
                            tz='UTC',
                            freq='1h'),
        columns=['value', 'quality_flag'])
    with pytest.raises(IndexError):
        tasks.apply_daily_validation(obs, data)


def test_fetch_and_validate_all_observations(mocker, make_observation,
                                             daily_index):
    obs = [make_observation('dhi'), make_observation('dni')]
    obs += [make_observation('ghi').replace(provider='Organization 2')]
    data = pd.DataFrame(
        [(0, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.list_observations',
                 return_value=obs)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_user_info',
                 return_value={'organization': obs[0].provider})
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')
    validated = pd.Series(2, index=daily_index)
    validate_mock = mocker.MagicMock(return_value=validated)
    mocker.patch.dict(
        'solarforecastarbiter.validation.tasks.IMMEDIATE_VALIDATION_FUNCS',
        {'dhi': validate_mock, 'dni': validate_mock})
    tasks.fetch_and_validate_all_observations(
        '', data.index[0], data.index[-1], only_missing=False)
    assert post_mock.called
    assert validate_mock.call_count == 2


def test_fetch_and_validate_all_observations_only_missing(
        mocker, make_observation, daily_index):
    obs = [make_observation('dhi'), make_observation('dni')]
    obs += [make_observation('ghi').replace(provider='Organization 2')]
    data = pd.DataFrame(
        [(0, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.list_observations',
                 return_value=obs)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_user_info',
                 return_value={'organization': obs[0].provider})
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values_not_flagged',  # NOQA
        return_value=np.array(['2019-01-01', '2019-01-02'],
                              dtype='datetime64[D]'))
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')
    tasks.fetch_and_validate_all_observations(
        '', data.index[0], data.index[-1], only_missing=True)
    assert post_mock.called
    assert (post_mock.call_args_list[0][0][1].index.date ==
            dt.date(2019, 1, 1)).all()
    assert (post_mock.call_args_list[1][0][1].index.date ==
            dt.date(2019, 1, 2)).all()
    assert (post_mock.call_args_list[2][0][1].index.date ==
            dt.date(2019, 1, 1)).all()
    assert (post_mock.call_args_list[3][0][1].index.date ==
            dt.date(2019, 1, 2)).all()


def test_fetch_and_validate_observation_only_missing(
        mocker, make_observation, daily_index):
    obs = make_observation('ac_power')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
         (300, 0), (300, 0), (300, 0), (100, 0), (0, 0),
         (100, 1), (0, 0), (0, 0)],
        index=daily_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_user_info',
                 return_value={'organization': obs.provider})
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values_not_flagged',  # NOQA
        return_value=np.array(['2019-01-01', '2019-01-02'],
                              dtype='datetime64[D]'))
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')
    tasks.fetch_and_validate_observation(
        'token', 'obsid', data.index[0], data.index[-1], only_missing=True)
    assert post_mock.called
    assert (post_mock.call_args_list[0][0][1].index.date ==
            dt.date(2019, 1, 1)).all()
    assert (post_mock.call_args_list[1][0][1].index.date ==
            dt.date(2019, 1, 2)).all()


def test__group_continuous_week_post(mocker, make_observation):
    split_dfs = [
        pd.DataFrame([(0, LATEST_VERSION_FLAG)],
                     columns=['value', 'quality_flag'],
                     index=pd.date_range(
                         start='2020-05-03T00:00',
                         end='2020-05-03T23:59',
                         tz='UTC',
                         freq='1h')),
        # new week split
        pd.DataFrame([(0, LATEST_VERSION_FLAG)],
                     columns=['value', 'quality_flag'],
                     index=pd.date_range(
                         start='2020-05-04T00:00',
                         end='2020-05-04T11:59',
                         tz='UTC',
                         freq='1h')),
        # missing 12
        pd.DataFrame(
            [(0, LATEST_VERSION_FLAG | DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'])] +  # NOQA
            [(1, LATEST_VERSION_FLAG)] * 7,
            columns=['value', 'quality_flag'],
            index=pd.date_range(
                start='2020-05-04T13:00',
                end='2020-05-04T20:00',
                tz='UTC',
                freq='1h')),
        # missing a week+
        pd.DataFrame(
            [(9, LATEST_VERSION_FLAG | DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'])] +  # NOQA
            [(3, LATEST_VERSION_FLAG)] * 7,
            columns=['value', 'quality_flag'],
            index=pd.date_range(
                start='2020-05-13T09:00',
                end='2020-05-13T16:59',
                tz='UTC',
                freq='1h')),
    ]
    ov = pd.concat(split_dfs, axis=0)
    obs = make_observation('ghi')
    session = mocker.MagicMock()
    tasks._group_continuous_week_post(session, obs, ov)
    call_list = session.post_observation_values.call_args_list
    assert len(call_list) == 4
    for i, cal in enumerate(call_list):
        assert_frame_equal(split_dfs[i], cal[0][1])


@pytest.mark.parametrize('vals,func', [
    (pd.DataFrame({'value': 0, 'quality_flag': 4}, index=pd.DatetimeIndex(
        [pd.Timestamp.utcnow()], name='timestamp')),
     'apply_immediate_validation'),
    (pd.DataFrame({'value': [0.0] * 5 + [None] * 10, 'quality_flag': 4},
                  index=pd.date_range('now', name='timestamp', freq='2h',
                                      periods=15)),
     'apply_immediate_validation'),
    (pd.DataFrame({'value': [0.0] * 15 + [None] * 11, 'quality_flag': 4},
                  index=pd.date_range('now', name='timestamp', freq='1h',
                                      periods=26)),
     'apply_daily_validation'),
])
def test_apply_validation(make_observation, mocker, vals, func):
    obs = make_observation('ac_power')
    fmock = mocker.patch.object(tasks, func, autospec=True)
    tasks.apply_validation(obs, vals)
    assert fmock.called


def test_apply_validation_empty(make_observation, mocker):
    obs = make_observation('dhi')
    daily = mocker.patch.object(tasks, 'apply_daily_validation')
    immediate = mocker.patch.object(tasks, 'apply_immediate_validation')
    data = pd.DataFrame({'value': [], 'quality_flag': []},
                        index=pd.DatetimeIndex([], name='timestamp'))
    out = tasks.apply_validation(obs, data)
    assert_frame_equal(out, data)
    assert not daily.called
    assert not immediate.called


def test_apply_validation_bad_df(make_observation, mocker):
    obs = make_observation('dhi')
    data = pd.DataFrame()
    with pytest.raises(TypeError):
        tasks.apply_validation(obs, data)

    with pytest.raises(TypeError):
        tasks.apply_validation(obs, pd.Series(
            index=pd.DatetimeIndex([]),
            dtype=float))


def test_apply_validation_inconsistent_interval(make_observation):
    obs = make_observation('ghi')
    index = pd.date_range(start='20200924', end='20200925', freq='1min')
    data = pd.DataFrame({'value': 1, 'quality_flag': 2}, index=index)
    if obs.interval_label == 'instant':
        pass
    else:
        with pytest.raises(KeyError, match='Missing times'):
            tasks.apply_validation(obs, data)


def test_apply_validation_agg(aggregate, mocker):
    data = pd.DataFrame({'value': [1], 'quality_flag': [0]},
                        index=pd.DatetimeIndex(
                            ['2020-01-01T00:00Z'], name='timestamp'))
    out = tasks.apply_validation(aggregate, data)
    assert_frame_equal(data, out)


def test_find_unvalidated_time_ranges(mocker):
    session = mocker.MagicMock()
    session.get_observation_values_not_flagged.return_value = np.array(
        ['2019-04-13', '2019-04-14', '2019-04-15', '2019-04-16', '2019-04-18',
         '2019-05-22', '2019-05-23'], dtype='datetime64[D]')
    obs = mocker.MagicMock()
    obs.observation_id = ''
    obs.site.timezone = 'UTC'
    out = list(tasks._find_unvalidated_time_ranges(
        session, obs, '2019-01-01T00:00Z', '2020-01-01T00:00Z'))
    assert out == [
        (pd.Timestamp('2019-04-13T00:00Z'), pd.Timestamp('2019-04-17T00:00Z')),
        (pd.Timestamp('2019-04-18T00:00Z'), pd.Timestamp('2019-04-19T00:00Z')),
        (pd.Timestamp('2019-05-22T00:00Z'), pd.Timestamp('2019-05-24T00:00Z')),
    ]


def test_find_unvalidated_time_ranges_all(mocker):
    session = mocker.MagicMock()
    session.get_observation_values_not_flagged.return_value = np.array(
        ['2019-04-13', '2019-04-14', '2019-04-15', '2019-04-16'],
        dtype='datetime64[D]')
    obs = mocker.MagicMock()
    obs.observation_id = ''
    obs.site.timezone = 'Etc/GMT+7'
    out = list(tasks._find_unvalidated_time_ranges(
        session, obs, '2019-01-01T00:00Z', '2020-01-01T00:00Z'))
    assert out == [
        (pd.Timestamp('2019-04-13T00:00-07:00'),
         pd.Timestamp('2019-04-17T00:00-07:00')),
    ]


def test_find_unvalidated_time_ranges_single(mocker):
    session = mocker.MagicMock()
    session.get_observation_values_not_flagged.return_value = np.array(
        ['2019-04-13'], dtype='datetime64[D]')
    obs = mocker.MagicMock()
    obs.observation_id = ''
    obs.site.timezone = 'Etc/GMT+5'
    out = list(tasks._find_unvalidated_time_ranges(
        session, obs, '2019-01-01T00:00Z', '2020-01-01T00:00Z'))
    assert out == [
        (pd.Timestamp('2019-04-13T00:00-05:00'),
         pd.Timestamp('2019-04-14T00:00-05:00')),
    ]


def test_find_unvalidated_time_ranges_disjoint(mocker):
    session = mocker.MagicMock()
    session.get_observation_values_not_flagged.return_value = np.array(
        ['2019-04-13', '2019-05-22'], dtype='datetime64[D]')
    obs = mocker.MagicMock()
    obs.observation_id = ''
    obs.site.timezone = 'Etc/GMT+5'
    out = list(tasks._find_unvalidated_time_ranges(
        session, obs, '2019-01-01T00:00Z', '2020-01-01T00:00Z'))
    assert out == [
        (pd.Timestamp('2019-04-13T00:00-05:00'),
         pd.Timestamp('2019-04-14T00:00-05:00')),
        (pd.Timestamp('2019-05-22T00:00-05:00'),
         pd.Timestamp('2019-05-23T00:00-05:00')),
    ]


def test_find_unvalidated_time_ranges_empty(mocker):
    session = mocker.MagicMock()
    session.get_observation_values_not_flagged.return_value = np.array(
         [], dtype='datetime64[D]')
    obs = mocker.MagicMock()
    obs.observation_id = ''
    obs.site.timezone = 'UTC'
    out = list(tasks._find_unvalidated_time_ranges(
        session, obs, '2019-01-01T00:00Z', '2020-01-01T00:00Z'))
    assert out == []
