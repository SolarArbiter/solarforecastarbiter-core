import copy
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.metrics import preprocessing


DATE_INDEXES = pd.date_range(start='2019-03-31T12:00:00.0000000000',
                             end='2019-04-02T12:00:00.0000000000',
                             freq='5min')

@pytest.fixture
def observation_dataframe():
    df_index = copy.deepcopy(DATE_INDEXES)
    df = pd.DataFrame(data = {'value' : np.arange(df_index.size).astype(float),
                              'quality_flag' : np.zeros(df_index.size).astype(int)},
                      index=df_index)
    return df


@pytest.fixture
def forecast_series():
    series_index = copy.deepcopy(DATE_INDEXES)
    series = pd.Series(data=[np.arange(series_index.size)].astype(float),
                       index=series_index)
    return series


def test_exclude_on_observation(observation_dataframe):
    df_obs = observation_dataframe
    n_values = df_obs.value.size
    
    # no bad data
    processed_obs_values = preprocessing.exclude(df_obs.value,
                                                 df_obs.quality_flag)
    
    assert n_values == processed_obs_values.size
    assert processed_obs_values.isna().any() == False
    pd.testing.assert_series_equal(df_obs.value,
                                   processed_obs_values)
    
    # add missing data to values
    df_obs_missing = df_obs.copy(deep=True)
    n_miss = int(0.1 * n_values)  # 10%
    df_sample = df_obs_missing.sample(n_miss)
    df_sample.value = 'nan'  # have to use string because udpate won't replace NaN
    df_obs_missing.update(df_sample)
    df_obs_missing.replace('nan', np.NaN, inplace=True)
    processed_obs_values = preprocessing.exclude(df_obs_missing.value,
                                                 df_obs_missing.quality_flag)
    
    assert (n_values - n_miss) == processed_obs_values.size
    assert processed_obs_values.isna().any() == False 
    pd.testing.assert_series_equal(df_obs_missing.value[~df_obs_missing.value.isna()],
                                   processed_obs_values)
    
    # add bad quality flags
    df_obs_bad_qual = df_obs.copy(deep=True)
    n_quality = int(0.2 * n_values)  # 20%
    
    
    # both missing data and bad quality flags
    
    

# def test_exclude_on_forecast(forecast_series):
#     ser_fx = forecast_series
    
    
    
    
