import sys
import copy
import itertools
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.metrics import evaluator, context


EPSILON = sys.float_info.epsilon


SHORT_DATE_INDEXES = pd.date_range(start='2019-02-20T12:00:00.0000000000',
                                   end='2019-02-20T16:00:00.0000000000',
                                   freq='1H')


LONG_DATE_INDEXES = pd.date_range(start='2019-01-10T00:00:00.0000000000',
                                  end='2019-12-31T23:00:00.0000000000',
                                  freq='1H')


@pytest.fixture
def observation_dataframe():
    df_index = copy.deepcopy(SHORT_DATE_INDEXES)
    df = pd.DataFrame(
        data={'value': np.arange(df_index.size).astype(float),
              'quality_flag': np.zeros(df_index.size).astype(int)},
        index=df_index)
    return df


@pytest.fixture
def forecast_series():
    series_index = copy.deepcopy(SHORT_DATE_INDEXES)
    series = pd.Series(np.arange(series_index.size).astype(float)+1.0,
                       index=series_index)
    return series


@pytest.fixture
def random_observation_dataframe():
    df_index = copy.deepcopy(SHORT_DATE_INDEXES)
    df = pd.DataFrame(
        data={'value': np.random.randn(df_index.size).astype(float),
              'quality_flag': np.zeros(df_index.size).astype(int)},
        index=df_index)
    return df


@pytest.fixture
def random_forecast_series():
    series_index = copy.deepcopy(SHORT_DATE_INDEXES)
    series = pd.Series(data=np.random.randn(series_index.size).astype(float),
                       index=series_index)
    return series


def test_evaluator_with_single_deterministic(observation_dataframe,
                                             forecast_series):
    # Default context
    default_context = context.get_default_deterministic_context()
    default_result = evaluator.evaluate(observation_dataframe,
                                        forecast_series,
                                        default_context)

    assert np.array_equal(default_result.processed_observations,
                          observation_dataframe.value)
    assert np.array_equal(default_result.processed_forecasts,
                          forecast_series)

    # Check Metrics
    for metric_name, metric_val in default_context['metrics'].items():
        
        # Total
        metrics_list = list(default_result.metrics.columns)
        if not metric_val:
            assert metric_name not in metrics_list
        else:
            # Metric is defined
            assert metric_name in metrics_list

            # Check groups
            res_groups = default_context['results']['groupings']
            for group_name, group_val in res_groups.items():
                group_list = list(default_result.metrics.index.names)
                if not group_val:
                    assert group_name not in group_list
                else:
                    assert group_name in group_list
                    assert default_result.metrics.index.get_level_values(
                        group_name) is not None


def test_evaluate_by_group():

    groupings = context.supported_groupings()
    metrics = context.supported_metrics()

    timestamps = copy.deepcopy(LONG_DATE_INDEXES)

    obs_series = pd.Series(data=np.random.randn(timestamps.size).astype(float),
                           index=timestamps)
    fx_series = pd.Series(data=np.random.randn(timestamps.size).astype(float),
                          index=timestamps)

    for group in groupings:
        for metric in metrics:

            metric_func = context._METRICS_MAP[metric]
            result = evaluator.evaluate_by_group(obs_series,
                                                 fx_series,
                                                 group,
                                                 metric_func)

            # Checks
            if metric == 'month':
                assert result.size == 12
            if metric == 'weekday':
                assert result.size == 7
            if metric == 'hour':
                assert result.size == 24
            if metric == 'date':
                assert result.size == 365

            if metric in ['mae', 'rsme']:
                assert (result > 0.).all()
            if metric in ['mbe']:
                assert (abs(result) < 3*2).all()


def test_evaluate_with_timezone(random_observation_dataframe,
                                random_forecast_series):

    # Test America/New_York - Eastern with DST
    context_eastern = context.get_default_deterministic_context()
    context_eastern['timezone'] = 'America/New_York'
    for group, _ in context_eastern['results']['groupings'].items():
        context_eastern['results']['groupings'][group] = True
    result_eastern = evaluator.evaluate(random_observation_dataframe,
                                        random_forecast_series,
                                        context_eastern)

    assert result_eastern.processed_observations.index.tz.zone == \
        context_eastern['timezone']
    assert result_eastern.processed_forecasts.index.tz.zone == \
        context_eastern['timezone']

    # Test America/Phoenix - Mountain without DST
    context_mountain = context.get_default_deterministic_context()
    context_mountain['timezone'] = 'America/Phoenix'
    for group, _ in context_mountain['results']['groupings'].items():
        context_mountain['results']['groupings'][group] = True
    result_mountain = evaluator.evaluate(random_observation_dataframe,
                                         random_forecast_series,
                                         context_mountain)

    assert result_mountain.processed_observations.index.tz.zone == \
        context_mountain['timezone']
    assert result_mountain.processed_forecasts.index.tz.zone == \
        context_mountain['timezone']

    # Compare groupings
    e_metrics = result_eastern.metrics
    m_metrics = result_mountain.metrics

    # Check values
    for e_row, m_row in zip(e_metrics.iterrows(), m_metrics.iterrows()):
        e_idx, e_vals = e_row[0], e_row[1]
        m_idx, m_vals = m_row[0], m_row[1]
        e_idx_ok = [i for i in e_idx if not np.isnan(i)]
        m_idx_ok = [i for i in m_idx if not np.isnan(i)]
        # TODO provide checks 
