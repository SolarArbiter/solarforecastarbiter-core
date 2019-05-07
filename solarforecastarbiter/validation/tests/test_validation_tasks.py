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
            extra_parameters='')
    return f


@pytest.fixture()
def default_index(single_site):
    return [pd.Timestamp('2019-01-01T08:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T09:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T10:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T11:00:00', tz=single_site.timezone),
            pd.Timestamp('2019-01-01T13:00:00', tz=single_site.timezone)]


def test_validate_ghi(mocker, make_observation, default_index):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky']]
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
                DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'])
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
        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED']]
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
        LATEST_VERSION_FLAG,
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
                       'check_temperature_limits']]
    obs = make_observation('air_temperature')
    data = pd.Series([10, 1000, -400, 30, 20], index=default_index)
    flags = tasks.validate_air_temperature(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
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
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
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
                       'check_wind_limits']]
    obs = make_observation('wind_speed')
    data = pd.Series([10, 1000, -400, 3, 20], index=default_index)
    flags = tasks.validate_wind_speed(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
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
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
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
                       'check_rh_limits']]
    obs = make_observation('relative_humidity')
    data = pd.Series([10, 101, -400, 60, 20], index=default_index)
    flags = tasks.validate_relative_humidity(obs, data)
    for mock in mocks:
        assert mock.called

    expected = (pd.Series([0, 0, 0, 0, 1], index=data.index) *
                DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'],
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
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['OK'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['USER FLAGGED'] | LATEST_VERSION_FLAG,
        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY'] |
        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'] | LATEST_VERSION_FLAG]
    assert post_mock.called_once
    assert_frame_equal(post_mock.call_args[0][1], out)
