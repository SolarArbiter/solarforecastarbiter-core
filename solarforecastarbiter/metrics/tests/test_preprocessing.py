import copy
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.io import api
from solarforecastarbiter.reports import template, main
from solarforecastarbiter.metrics import preprocessing

THREE_HOURS = pd.date_range(start='2019-03-31T12:00:00',
                            periods=3,
                            freq='60min',
                            tz='MST',
                            name='timestamp')

THREE_HOUR_SERIES = pd.Series(np.arange(1.,4.,1.), index=THREE_HOURS,
                              name='value')

THIRTEEN_10MIN = pd.date_range(start='2019-03-31T12:00:00',
                                periods=13,
                                freq='10min',
                                tz='MST',
                                name='timestamp')

THIRTEEN_10MIN_SERIES = pd.Series((np.arange(0.,13.,1.)/6)+1,
                                   index=THIRTEEN_10MIN)

# Bitwise-flag integers (only test validated and versioned data)
CL_UF = int(0b10010)  # Cloudy and User Flagged (18)
CSE_NT = int(0b1000010010) # Clearsky exceeded, nighttime, and version 0 (530)
CSE = int(0b1000000010) # Clearsky exceeded and version 0 (514)
OK = int(0b10) # OK version 0 (2)



# @pytest.fixture
# def observation_dataframe():
#     df_index = copy.deepcopy(ONE_DAY)
#     df = pd.DataFrame(
#         data={'value': np.arange(df_index.size).astype(float),
#               'quality_flag': np.zeros(df_index.size).astype(int)},
#         index=df_index)
#     return df
#
#
# @pytest.fixture
# def forecast_series():
#     series_index = copy.deepcopy(ONE_DAY)
#     series = pd.Series(data=np.arange(series_index.size).astype(float),
#                        index=series_index)
#     return series

@pytest.mark.parametrize('fx0', [
    pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
    THREE_HOUR_SERIES
])
@pytest.mark.parametrize('fx1', [
    pd.Series(index=pd.DatetimeIndex([], name='timestamp'), name='value'),
    THREE_HOUR_SERIES
])
@pytest.mark.parametrize('obs', [
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [OK, OK, OK]},
                  index=THREE_HOURS),
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [CL_UF, CL_UF , CL_UF]},
                  index=THREE_HOURS),
    pd.DataFrame(index=pd.DatetimeIndex([], name='timestamp'),
                 columns=['value', 'quality_flag']),
    pd.DataFrame({'value': [1., 2., 3.],
                  'quality_flag': [CSE_NT, CSE , OK]},
                  index=THREE_HOURS),
])
@pytest.mark.parametrize('handle_func',[preprocessing.exclude])
def test_apply_validation(report_objects, fx0, fx1, obs, handle_func):
    report, obs_model, fx0_model, fx1_model = report_objects
    data = {
        obs_model: obs,
        fx0_model: fx0,
        fx1_model: fx1
    }
    result = preprocessing.apply_validation(data, report.filters[0], handle_func)

    # Check length and timestamps of observation
    assert len(obs[obs.quality_flag.isin([OK, CSE])]) \
                == len(result[obs_model])
    if not result[obs_model].empty:
        assert obs[obs.quality_flag.isin([OK, CSE])].index.equals(
                    result[obs_model].index)


@pytest.mark.parametrize('values,qflags,expectation', [
    (THREE_HOUR_SERIES, None, THREE_HOUR_SERIES),
    (THREE_HOUR_SERIES,
        pd.Series([0, 0, 0], index=THREE_HOURS),
        THREE_HOUR_SERIES),
    (THREE_HOUR_SERIES,
        pd.Series([0, 1, 0], index=THREE_HOURS),
        THREE_HOUR_SERIES[[True, False, True]]),
    (THREE_HOUR_SERIES,
        pd.Series([1, 1, 0], index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, True]]),
    (THREE_HOUR_SERIES,
        pd.Series([1, 1, 1], index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, False]]),
    (pd.Series([np.NaN, np.NaN, np.NaN], index=THREE_HOURS),
        None,
        THREE_HOUR_SERIES[[False, False, False]]),
    (pd.Series([1., np.NaN, 3.], index=THREE_HOURS),
        pd.Series([0, 0, 0], index=THREE_HOURS),
        THREE_HOUR_SERIES[[True, False, True]]),
    (pd.Series([1., np.NaN, 3.], index=THREE_HOURS),
        pd.Series([1, 1, 0], index=THREE_HOURS),
        THREE_HOUR_SERIES[[False, False, True]])
])
def test_exclude(values, qflags, expectation):
    result = preprocessing.exclude(values, qflags)
    print(result)
    assert result.equals(expectation)
