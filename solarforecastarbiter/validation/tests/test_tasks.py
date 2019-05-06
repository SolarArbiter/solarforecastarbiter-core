import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest


from solarforecastarbiter.datamodel import Observation
from solarforecastarbiter.validation import tasks, validator
from solarforecastarbiter.validation.quality_mapping import (
    LATEST_VERSION_FLAG, DESCRIPTION_MASK_MAPPING)


@pytest.fixture()
def make_observation(site_metadata):
    def f(variable):
        return Observation(
            name='test', variable=variable, interval_value_type='mean',
            interval_length=pd.Timedelta('1hr'), interval_label='beginning',
            site=site_metadata, uncertainty=0.1, observation_id='OBSID',
            extra_parameters='')
    return f


def test_validate_ghi(mocker, make_observation, observation_values):
    mocks = [mocker.patch.object(validator, f,
                                 new=mocker.MagicMock(
                                     wraps=getattr(validator, f)))
             for f in ['check_timestamp_spacing',
                       'check_irradiance_day_night',
                       'check_ghi_limits_QCRad',
                       'check_ghi_clearsky']]
    obs = make_observation('ghi')
    data = pd.Series([10, 1000, -100, 500, 500], index=[
        pd.Timestamp('2019-01-01T07:00:00', tz=obs.site.timezone),
        pd.Timestamp('2019-01-01T08:00:00', tz=obs.site.timezone),
        pd.Timestamp('2019-01-01T09:00:00', tz=obs.site.timezone),
        pd.Timestamp('2019-01-01T10:00:00', tz=obs.site.timezone),
        pd.Timestamp('2019-01-01T12:00:00', tz=obs.site.timezone)])
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


def test_immediate_observation_validation_ghi(mocker, make_observation):
    obs = make_observation('ghi')
    data = pd.DataFrame(
        [(0, 0), (100, 0), (200, 0), (-1, 1), (1500, 0)],
        index=[
            pd.Timestamp('2019-01-01T07:00:00', tz=obs.site.timezone),
            pd.Timestamp('2019-01-01T08:00:00', tz=obs.site.timezone),
            pd.Timestamp('2019-01-01T09:00:00', tz=obs.site.timezone),
            pd.Timestamp('2019-01-01T10:00:00', tz=obs.site.timezone),
            pd.Timestamp('2019-01-01T12:00:00', tz=obs.site.timezone)],
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
