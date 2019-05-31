import copy
import numpy as np
import pandas as pd
import pytest

from solarforecastarbiter.metrics import evaluator, context


SHORT_DATE_INDEXES = pd.date_range(start='2019-02-20T12:00:00.0000000000',
                                   end='2019-02-27T16:00:00.0000000000',
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
    series = pd.Series(data=np.arange(series_index.size).astype(float)+1.0,
                       index=series_index)
    return series


def test_evaluator_with_single_deterministic(observation_dataframe,
                                             forecast_series):
    # Default context
    default_context = context.get_default_deterministic_context()
    default_result = evaluator.evaluate(observation_dataframe,
                                        forecast_series,
                                        default_context)

    assert np.array_equal(default_result['timeseries']['observations'],
                          observation_dataframe.value)
    assert np.array_equal(default_result['timeseries']['forecasts'],
                          forecast_series)

    # Check Metrics
    for metric_name, metric_val in default_context['metrics'].items():
        # Total
        metric_key_list = default_result['metrics']['total'].keys()
        if not metric_val:
            assert metric_name not in metric_key_list
        else:
            assert metric_name in metric_key_list

            # Groupings
            res_groups = default_context['results']['groupings']
            for group_name, group_val in res_groups.items():
                print(group_name, group_val, metric_name, metric_val)
                group_key_list = default_result['metrics'].keys()
                if not group_val:
                    assert group_name not in group_key_list
                else:
                    assert group_name in group_key_list
                    mg_key_list = default_result['metrics'][group_name].keys()
                    assert metric_name in mg_key_list


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

            if metric in ['mae', 'rsme']:
                assert (result > 0.).all()
            if metric in ['mbe']:
                assert (result < 0.5).all()
