import pandas as pd
from pandas.testing import assert_series_equal
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
    out = tasks.validate_ghi(obs, data)
    for mock in mocks:
        assert mock.called
    assert_series_equal(
        out, pd.Series([DESCRIPTION_MASK_MAPPING['NIGHTTIME'],
                        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'] |
                        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                        DESCRIPTION_MASK_MAPPING['LIMITS EXCEEDED'],
                        DESCRIPTION_MASK_MAPPING['CLEARSKY EXCEEDED'],
                        DESCRIPTION_MASK_MAPPING['UNEVEN FREQUENCY']],
                       index=data.index) | LATEST_VERSION_FLAG)
