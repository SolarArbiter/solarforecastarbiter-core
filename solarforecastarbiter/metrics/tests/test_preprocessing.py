import copy
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.metrics import preprocessing


DATE_INDEXES = pd.date_range(start='2019-03-31T12:00:00.0000000000',
                             end='2019-03-31T16:00:00.0000000000',
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
    series = pd.Series(data=np.arange(series_index.size).astype(float),
                       index=series_index)
    return series


def test_exclude_on_forecast(forecast_series):
    ser_fx = forecast_series
    n_values = ser_fx.size
    
    # No bad data
    processed_fx_values = preprocessing.exclude(ser_fx)
    
    assert processed_fx_values.isna().any() == False
    pd.testing.assert_series_equal(ser_fx,
                                   processed_fx_values)
    
    # Missing data
    ser_fx_missing = ser_fx.copy(deep=True)
    n_miss = int(0.25 * n_values)  # 25%
    ser_sample = ser_fx_missing.sample(n_miss)
    ser_sample.values[:] = -99999 # have to use placeholder as update doesn't copy NaNs
    ser_fx_missing.update(ser_sample)
    ser_fx_missing.replace(-99999, np.NaN, inplace=True)
    processed_fx_values = preprocessing.exclude(ser_fx_missing)
    
    assert (n_values - n_miss) == processed_fx_values.size
    assert processed_fx_values.isna().any() == False
    pd.testing.assert_series_equal(ser_fx_missing[~ser_fx_missing.isna()],
                                   processed_fx_values)
    

def test_exclude_on_observation(observation_dataframe):
    df_obs = observation_dataframe
    n_values = df_obs.value.size
    
    # No bad data
    processed_obs_values = preprocessing.exclude(df_obs.value,
                                                 df_obs.quality_flag)
    
    assert n_values == processed_obs_values.size
    assert processed_obs_values.isna().any() == False
    pd.testing.assert_series_equal(df_obs.value,
                                   processed_obs_values)
    
    # Missing Data
    df_obs_missing = df_obs.copy(deep=True)
    n_miss = int(0.1 * n_values)  # 10%
    df_sample = df_obs_missing.sample(n_miss)
    df_sample.value = 'nan'  # have to use placeholder as update doesn't copy NaNs
    df_obs_missing.update(df_sample)
    df_obs_missing.replace('nan', np.NaN, inplace=True)
    processed_obs_values = preprocessing.exclude(df_obs_missing.value,
                                                 df_obs_missing.quality_flag)
    
    assert (n_values - n_miss) == processed_obs_values.size
    assert processed_obs_values.isna().any() == False 
    pd.testing.assert_series_equal(df_obs_missing.value[~df_obs_missing.value.isna()],
                                   processed_obs_values)
    
    # Bad Quality Flags
    df_obs_bad_qual = df_obs.copy(deep=True)
    n_bad_quality = int(0.2 * n_values)  # 20%
    df_sample = df_obs_missing.sample(n_bad_quality)
    df_sample.quality_flag = 1
    df_obs_bad_qual.update(df_sample)
    processed_obs_values = preprocessing.exclude(df_obs_bad_qual.value,
                                                 df_obs_bad_qual.quality_flag)
    
    assert (n_values - n_bad_quality) == processed_obs_values.size
    assert np.intersect1d(df_sample.index.values, 
                          processed_obs_values.index.values).size == 0
    pd.testing.assert_series_equal(df_obs_bad_qual.value[df_obs_bad_qual.quality_flag == 0],
                                   processed_obs_values)
    
    # Missing and Bad Quality
    df_obs_mixed = pd.DataFrame(data={'value' : df_obs_missing.value,
                                      'quality_flag' : df_obs_bad_qual.quality_flag},
                                index=df_obs.index)
    processed_obs_values = preprocessing.exclude(df_obs_missing.value,
                                                 df_obs_bad_qual.quality_flag)
    
    check_values = df_obs_mixed[~(df_obs_mixed.value.isna() |
                                  df_obs_mixed.quality_flag == 1)].value
    pd.testing.assert_series_equal(check_values,
                                   processed_obs_values)
    
