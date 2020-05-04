import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest


from solarforecastarbiter.datamodel import Observation
from solarforecastarbiter.validation import tasks, validator
from solarforecastarbiter.validation.quality_mapping import (
    LATEST_VERSION_FLAG, DESCRIPTION_MASK_MAPPING)


@pytest.fixture()
def make_observation(single_site):
    def f(variable):
        return Observation(
            name='test', variable=variable, interval_value_type='mean',
            interval_length=pd.Timedelta('1hr'), interval_label='beginning',
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
        pd.Index([pd.Timestamp('2019-01-01T23:00:00',
                               tz=single_site.timezone)]))


def test_validate_ghi(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi']]
    obs = make_observation('ghi')
    data = pd.Series([10, 1000, -100, 500, 300], index=default_index)
    flags = tasks.validate_ghi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
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
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi']]

    obs = make_observation('ghi').replace(interval_length=pd.Timedelta('5min'))
    index = pd.date_range(start='2019-04-01T11:00', freq='5min',
                          tz=obs.site.timezone, periods=11)
    data = pd.Series([742, 749, 756, 763, 769, 774, 779, 784, 789, 793, 700],
                     index=index)
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


def test_immediate_observation_validation_ghi(mocker, make_observation,
                                              default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']
    ]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_immediate_observation_validation_ghi_nones(
        mocker, make_observation, default_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(None, 1)] * 5, index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    base = (
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        LATEST_VERSION_FLAG
    )
    out['quality_flag'] = [
        base | DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
        base,
        base,
        base,
        base | DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY']
    ]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_immediate_observation_validation_not_listed(mocker, make_observation,
                                                     default_index):
    obs = make_observation('curtailment')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        LATEST_VERSION_FLAG,
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_dni(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_dni_limits_QCRad']]
    obs = make_observation('dni')
    data = pd.Series([10, 1000, -100, 500, 500], index=default_index)
    flags = tasks.validate_dni(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 0, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_dni(mocker, make_observation,
                                              default_index):
    obs = make_observation('dni')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED']]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_dhi(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_dhi_limits_QCRad']]
    obs = make_observation('dhi')
    data = pd.Series([10, 1000, -100, 200, 200], index=default_index)
    flags = tasks.validate_dhi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_dhi(mocker, make_observation,
                                              default_index):
    obs = make_observation('dhi')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED']]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_poa_global(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_poa_clearsky']]
    obs = make_observation('poa_global')
    data = pd.Series([10, 1000, -400, 300, 300], index=default_index)
    flags = tasks.validate_poa_global(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 1, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_poa_global(mocker, make_observation,
                                                     default_index):
    obs = make_observation('poa_global')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] | LATEST_VERSION_FLAG |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_air_temp(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_temperature_limits']]
    obs = make_observation('air_temperature')
    data = pd.Series([10, 1000, -400, 30, 20], index=default_index)
    flags = tasks.validate_air_temperature(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_air_temperature(
        mocker, make_observation, default_index):
    obs = make_observation('air_temperature')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (20, 0), (-1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_wind_speed(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_wind_limits']]
    obs = make_observation('wind_speed')
    data = pd.Series([10, 1000, -400, 3, 20], index=default_index)
    flags = tasks.validate_wind_speed(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_wind_speed(
        mocker, make_observation, default_index):
    obs = make_observation('wind_speed')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (15, 0), (1, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_relative_humidity(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_rh_limits']]
    obs = make_observation('relative_humidity')
    data = pd.Series([10, 101, -400, 60, 20], index=default_index)
    flags = tasks.validate_relative_humidity(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 1, 1, 0, 0], index=data.index) *
                DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'])
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_immediate_observation_validation_relative_humidity(
        mocker, make_observation, default_index):
    obs = make_observation('relative_humidity')
    data = pd.DataFrame(
        [(0, 0), (200, 0), (15, 0), (40, 1), (1500, 0)],
        index=default_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.immediate_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)


def test_validate_daily_ghi(mocker, make_observation, daily_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky',
                       'detect_clearsky_ghi',
                       'detect_stale_values',
                       'detect_interpolation']]
    obs = make_observation('ghi')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [10, 1000, -100, 500, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        index=daily_index)
    flags = tasks.validate_daily_ghi(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
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


def test_daily_observation_validation_ghi(mocker, make_observation,
                                          daily_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(10, 0), (1000, 0), (-100, 0), (500, 0), (300, 0),
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

    tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        LATEST_VERSION_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_daily_observation_validation_ghi_zeros(mocker, make_observation,
                                                daily_index):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0)] * 13,
        index=daily_index,
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)

    post_mock = mocker.patch(
        'solarforecastarbiter.io.api.APISession.post_observation_values')

    tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    base = (
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        LATEST_VERSION_FLAG
    )
    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        base,
        base,
        base,
        base,
        base,
        base,
        base,
        base | DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
        base | DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
        base | DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
        base | DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY']
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_validate_daily_dc_power(mocker, make_observation, daily_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'detect_stale_values',
                       'detect_interpolation']]
    obs = make_observation('dc_power')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [0, 1000, -100, 500, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        index=daily_index)
    flags = tasks.validate_daily_dc_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['STALE VALUES'],
                pd.Series([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES']
                )
    for flag, exp in zip(flags, expected):
        assert_series_equal(flag, exp | LATEST_VERSION_FLAG,
                            check_names=False)


def test_daily_observation_validation_dc_power(mocker, make_observation,
                                               daily_index):
    obs = make_observation('dc_power')
    data = pd.DataFrame(
        [(10, 0), (1000, 0), (-100, 0), (500, 0), (300, 0),
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

    tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


def test_validate_daily_ac_power(mocker, make_observation, daily_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'detect_stale_values',
                       'detect_interpolation',
                       'detect_clipping']]
    obs = make_observation('ac_power')
    data = pd.Series(
        # 8     9     10   11   12  13    14   15  16  17  18  19  23
        [0, 100, -100, 100, 300, 300, 300, 300, 100, 0, 100, 0, 0],
        index=daily_index)
    flags = tasks.validate_daily_ac_power(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
                pd.Series([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                          index=data.index) *
                DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
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


def test_daily_observation_validation_ac_power(mocker, make_observation,
                                               daily_index):
    obs = make_observation('ac_power')
    data = pd.DataFrame(
        [(10, 0), (100, 0), (-100, 0), (100, 0), (300, 0),
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

    tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])

    out = data.copy()
    out['quality_flag'] = [
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['CLIPPED VALUES'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['STALE VALUES'] |
        DESCRIPTION_MASK_MAPPING['INTERPOLATED VALUES'] |
        DESCRIPTION_MASK_MAPPING['CLIPPED VALUES'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['NIGHTTIME'] |
        LATEST_VERSION_FLAG
    ]
    assert post_mock.called
    posted_df = pd.concat([cal[0][1] for cal in post_mock.call_args_list])
    assert_frame_equal(posted_df, out)


@pytest.mark.parametrize('var', ['air_temperature', 'wind_speed', 'dni', 'dhi',
                                 'poa_global', 'relative_humidity'])
def test_daily_observation_validation_other(var, mocker, make_observation,
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
    tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])
    assert post_mock.called
    assert validate_mock.called


def test_daily_observation_validation_many(mocker, make_observation,
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
    tasks.daily_observation_validation(
        '', data.index[0], data.index[-1])
    assert post_mock.called
    assert validate_mock.call_count == 2


def test_daily_single_observation_validation_not_enough(mocker,
                                                        make_observation):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0)],
        index=pd.date_range(start='2019-01-01T0000Z',
                            end='2019-01-01T0100Z',
                            tz='UTC',
                            freq='1h'),
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.get_observation',
                 return_value=obs)
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    log = mocker.patch('solarforecastarbiter.validation.tasks.logger.warning')
    out = tasks.daily_single_observation_validation(
        '', obs.observation_id, data.index[0], data.index[-1])
    assert out is None
    assert log.called


def test_daily_observation_validation_not_enough(mocker, make_observation):
    obs = [make_observation('ghi')]
    data = pd.DataFrame(
        [(0, 0)],
        index=pd.date_range(start='2019-01-01T0000Z',
                            end='2019-01-01T0100Z',
                            tz='UTC',
                            freq='1h'),
        columns=['value', 'quality_flag'])
    mocker.patch('solarforecastarbiter.io.api.APISession.list_observations',
                 return_value=obs)
    mocker.patch('solarforecastarbiter.io.api.APISession.get_user_info',
                 return_value={'organization': obs[0].provider})
    mocker.patch(
        'solarforecastarbiter.io.api.APISession.get_observation_values',
        return_value=data)
    log = mocker.patch('solarforecastarbiter.validation.tasks.logger.warning')
    out = tasks.daily_observation_validation(
        '', data.index[0], data.index[-1])
    assert out is None
    assert log.called


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
    for i, cal in enumerate(session.post_observation_values.call_arg_list):
        assert_frame_equal(split_dfs[i], cal[0][1])
