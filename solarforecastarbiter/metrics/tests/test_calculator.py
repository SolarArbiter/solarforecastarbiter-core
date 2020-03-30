import numpy as np
import pandas as pd
import pytest
import itertools
import calendar
import datetime

from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (calculator, deterministic,
                                          probabilistic)


DETERMINISTIC_METRICS = list(deterministic._MAP.keys())
DET_NO_NORM = (set(DETERMINISTIC_METRICS) - set(deterministic._REQ_NORM))
DET_NO_REF = (set(DETERMINISTIC_METRICS) - set(deterministic._REQ_REF_FX))
DET_NO_REF_NO_NORM = (set(DET_NO_NORM) - set(deterministic._REQ_REF_FX))
PROBABILISTIC_METRICS = list(probabilistic._MAP.keys())
PROB_NO_NORM = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_NORM))
PROB_NO_REF = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_REF_FX))
PROB_NO_2DFX = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_2DFX))
LIST_OF_CATEGORIES = list(datamodel.ALLOWED_CATEGORIES.keys())


@pytest.fixture()
def create_processed_fxobs(create_datetime_index):
    def _create_processed_fxobs(fxobs, fx_values, obs_values,
                                ref_values=None,
                                interval_label=None):

        if not interval_label:
            interval_label = fxobs.forecast.interval_label

        if (isinstance(fx_values, pd.Series) or
            isinstance(fx_values, pd.DataFrame)):
            conv_fx_values = fx_values
        else:
            conv_fx_values = pd.Series(fx_values,
                index=create_datetime_index(len(fx_values)))
        if isinstance(obs_values, pd.Series):
            conv_obs_values = obs_values
        else:
            conv_obs_values = pd.Series(obs_values,
                index=create_datetime_index(len(obs_values)))

        return datamodel.ProcessedForecastObservation(
            fxobs.forecast.name,
            fxobs,
            fxobs.forecast.interval_value_type,
            fxobs.forecast.interval_length,
            interval_label,
            valid_point_count=len(fx_values),
            forecast_values=conv_fx_values,
            observation_values=conv_obs_values)

    return _create_processed_fxobs


@pytest.fixture()
def create_datetime_index():
    def _create_datetime_index(n_periods):
        return pd.date_range(start='20190801', periods=n_periods, freq='1h',
                             tz='MST', name='timestamp')

    return _create_datetime_index


@pytest.fixture()
def copy_prob_forecast_with_axis():
    def _copy_prob_forecast_with_axis(probfx, axis):
        cvs = probfx.constant_values
        new_cvs = []
        new_probfx = probfx.replace(constant_values=())
        new_probfx = new_probfx.replace(axis=axis)
        for cv in cvs:
            cv = cv.replace(axis=axis)
            new_cvs.append(cv)
        new_probfx = new_probfx.replace(constant_values=tuple(new_cvs))
        return new_probfx

    return _copy_prob_forecast_with_axis


@pytest.fixture(params=['fxobs', 'fxagg'])
def single_forecast_data_obj(
        request, single_forecast_aggregate,
        single_forecast_observation):
    if request.param == 'fxobs':
        return single_forecast_observation
    else:
        return single_forecast_aggregate


@pytest.fixture(params=['probfxobs', 'probfxagg'])
def probabilistic_forecasts_data_obj(
        request, prob_forecasts,
        many_prob_forecasts):
    if request.param == 'probfxobs':
        return prob_forecasts
    else:
        return many_prob_forecasts


@pytest.fixture()
def proc_fx_obs(create_processed_fxobs, many_forecast_observation):
    proc_fx_obs = []

    for fx_obs in many_forecast_observation:
        proc_fx_obs.append(
            create_processed_fxobs(fx_obs,
                                   np.random.randn(10)+10,
                                   np.random.randn(10)+10)
        )
    return proc_fx_obs


@pytest.fixture()
def ref_fx_obs(create_processed_fxobs, many_forecast_observation):
    ref_fx_obs = create_processed_fxobs(many_forecast_observation[0],
                                        np.random.randn(10)+10,
                                        np.random.randn(10)+10)
    return ref_fx_obs


# Suppress RuntimeWarnings b/c in some metrics will divide by zero or
# don't handle single values well
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories,metrics', [
    pytest.param(
        [], [], marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    pytest.param(
        ['date', 'month'], [],
        marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    pytest.param(
        LIST_OF_CATEGORIES, DETERMINISTIC_METRICS,
        marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    pytest.param(
        LIST_OF_CATEGORIES, DET_NO_NORM,
        marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    (LIST_OF_CATEGORIES, DET_NO_REF),
    (LIST_OF_CATEGORIES, DET_NO_REF_NO_NORM)
])
def test_calculate_metrics_no_reference(
        categories, metrics, proc_fx_obs):
    result = calculator.calculate_metrics(proc_fx_obs,
                                          categories, metrics)

    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == len(proc_fx_obs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories,metrics', [
    pytest.param(
        [], [], marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    pytest.param(
        ['date', 'month'], [],
        marks=pytest.mark.xfail(strict=True, type=RuntimeError)),
    (LIST_OF_CATEGORIES, DETERMINISTIC_METRICS),
    (LIST_OF_CATEGORIES, DET_NO_REF),
])
def test_calculate_metrics_with_reference(
        categories, metrics, proc_fx_obs, ref_fx_obs):
    result = calculator.calculate_metrics(proc_fx_obs,
                                          categories, metrics,
                                          ref_pair=ref_fx_obs)

    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == len(proc_fx_obs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_single(single_forecast_data_obj,
                                  create_processed_fxobs, ref_fx_obs):
    inp = [create_processed_fxobs(single_forecast_data_obj,
                                  np.random.randn(10)+10,
                                  np.random.randn(10)+10)]
    result = calculator.calculate_metrics(inp, LIST_OF_CATEGORIES,
                                          DETERMINISTIC_METRICS,
                                          ref_pair=ref_fx_obs)
    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == 1


@pytest.mark.skip('')
@pytest.mark.parametrize('categories,metrics', [
    ([], []),
    (LIST_OF_CATEGORIES, DETERMINISTIC_METRICS),
    (LIST_OF_CATEGORIES, PROBABILISTIC_METRICS)
])
def test_calculate_metrics_with_probablistic(categories, metrics,
                                        create_processed_fxobs,
                                        single_observation,
                                        prob_forecasts):
    prfxobs = datamodel.ForecastObservation(prob_forecasts, single_observation)
    fx_values = [np.random.randn(10)+10 for _ in range(7)]
    obs_values = np.random.randn(10)+10
    proc_prfx_obs = create_processed_fxobs(prfxobs,
                                           fx_values,
                                           obs_values)
    calculator.calculate_metrics(proc_prfx_obs,
                                 categories, metrics)


def test_calculate_deterministic_metrics_no_metrics(ref_fx_obs):
    with pytest.raises(RuntimeError):
        calculator.calculate_deterministic_metrics(
            ref_fx_obs, LIST_OF_CATEGORIES, []
        )


def test_calculate_deterministic_metrics_no_reference(ref_fx_obs):
    with pytest.raises(RuntimeError):
        calculator.calculate_deterministic_metrics(
            ref_fx_obs, LIST_OF_CATEGORIES, DETERMINISTIC_METRICS
        )


def test_calculate_deterministic_metrics_no_reference_data(
        create_processed_fxobs, single_forecast_observation):
    pair = create_processed_fxobs(single_forecast_observation,
                                  np.random.randn(10),
                                  np.random.randn(10))
    ref = create_processed_fxobs(single_forecast_observation,
                                 np.random.randn(0),
                                 np.random.randn(0))

    with pytest.raises(RuntimeError):
        calculator.calculate_deterministic_metrics(
            pair, LIST_OF_CATEGORIES, DETERMINISTIC_METRICS,
            ref_fx_obs=ref
        )


@pytest.mark.parametrize('fx_vals,obs_vals', [
    (np.random.randn(0), np.random.randn(0)),
    (np.random.randn(10), np.random.randn(0)),
    (np.random.randn(0), np.random.randn(10)),
])
def test_calculate_deterministic_metrics_missing_values(
        create_processed_fxobs, single_forecast_observation,
        fx_vals, obs_vals):
    pair = create_processed_fxobs(single_forecast_observation,
                                  fx_vals, obs_vals)
    with pytest.raises(RuntimeError):
        calculator.calculate_deterministic_metrics(
            pair, LIST_OF_CATEGORIES, DET_NO_REF
        )


def test_calculate_deterministic_metrics_normalizer(
        create_processed_fxobs, single_forecast_observation):
    pair = create_processed_fxobs(single_forecast_observation,
                                  np.random.randn(10), np.random.randn(10))
    s0 = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_NORM, normalizer=1.0)
    s1 = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_NORM, normalizer=1.0)
    s2 = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_NORM, normalizer=2.0)
    for s in [s0, s1, s2]:
        assert isinstance(s, datamodel.MetricResult)
    assert s0 == s1
    assert s1 != s2


def test_calculate_deterministic_metrics_reference(
        create_processed_fxobs, single_forecast_observation):
    pair = create_processed_fxobs(single_forecast_observation,
                                  np.random.randn(10), np.random.randn(10))
    ref0 = create_processed_fxobs(single_forecast_observation,
                                  np.random.randn(10), np.random.randn(10))
    ref1 = create_processed_fxobs(single_forecast_observation,
                                  np.random.randn(10), np.random.randn(10))
    s0 = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_REF_FX,
        ref_fx_obs=ref0)
    s1 = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_REF_FX,
        ref_fx_obs=ref1)
    for s in [s0, s1]:
        assert isinstance(s, datamodel.MetricResult)
    assert s0 != s1


# Suppress RuntimeWarnings b/c in some metrics will divide by zero or
# don't handle single values well
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES
])
@pytest.mark.parametrize('metrics', [
    *list(itertools.combinations(DETERMINISTIC_METRICS, 1)),
    DETERMINISTIC_METRICS[0:1],
    DETERMINISTIC_METRICS[0:2],
    DETERMINISTIC_METRICS
])
def test_calculate_deterministic_metrics(categories, metrics,
                                         single_forecast_data_obj,
                                         single_forecast_observation,
                                         create_processed_fxobs):
    pair = create_processed_fxobs(single_forecast_data_obj,
                                  np.random.randn(10)+10,
                                  np.random.randn(10)+10)
    ref = create_processed_fxobs(single_forecast_observation,
                                 np.random.randn(10)+10,
                                 np.random.randn(10)+10)
    result = calculator.calculate_deterministic_metrics(
        pair, categories, metrics, ref_fx_obs=ref)
    # Check results
    assert isinstance(result, datamodel.MetricResult)
    assert result.forecast_id == pair.original.forecast.forecast_id
    assert result.name == pair.original.forecast.name
    assert len(result.values) % len(metrics) == 0
    cats = {val.category for val in result.values}
    assert cats == set(categories)
    fx_values = pair.forecast_values
    for cat in categories:
        cat_grps = {v.index for v in result.values if v.category == cat}
        assert len(
            {v.metric for v in result.values if v.category == cat}
        ) == len(metrics)

        # has expected groupings
        if cat == 'month':
            grps = fx_values.groupby(
                fx_values.index.month).groups
            grps = [calendar.month_abbr[g] for g in grps]
        elif cat == 'hour':
            grps = fx_values.groupby(
                fx_values.index.hour).groups
        elif cat == 'year':
            grps = fx_values.groupby(
                fx_values.index.year).groups
        elif cat == 'date':
            grps = fx_values.groupby(
                fx_values.index.date).groups
        elif cat == 'weekday':
            grps = fx_values.groupby(
                fx_values.index.weekday).groups
            grps = [calendar.day_abbr[g] for g in grps]
        elif cat == 'total':
            grps = ['0']
        assert {str(g) for g in grps} == cat_grps
    for val in result.values:
        assert (
            np.isnan(val.value) or
            np.issubdtype(type(val.value), np.number)
        )


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_no_metrics():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_no_reference():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_no_reference_data():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_bad_reference_axis():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_missing_values():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_reference():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics_all_forecasts():
    raise NotImplementedError


@pytest.mark.skip('')
def test_calculate_probabilistic_metrics():
    raise NotImplementedError


@pytest.mark.parametrize('metric,fx,obs,ref_fx,norm,expect', [
    ('mae', [], [], None, None, np.NaN),
    ('mae', [1, 1, 1], [0, 1, -1], None, None, 1.0),
    ('mbe', [1, 1, 1], [0, 1, -1], None, None, 1.0),
    ('rmse', [1, 0, 1], [0, -1, 2], None, None, 1.0),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, None, None),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, 2.0, 100/2),
    ('mape', [2, 3, 1], [4, 2, 2], None, None, 50.0),
    ('s', [1, 0, 1], [0, -1, 2], None, None, None),
    ('s', [1, 0, 1], [0, -1, 2], [2, 1, 0], None, 0.5),
    ('r', [3, 2, 1], [1, 2, 3], None, None, -1.0),
    ('r^2', [3, 2, 1], [1, 2, 3], None, None, -3.0),
    ('crmse', [1, 1, 1], [0, 1, 2], None, None, np.sqrt(2/3))
])
def test_apply_deterministic_metric_func(metric, fx, obs, ref_fx, norm, expect,
                                         create_datetime_index):
    fx_series = pd.Series(fx, index=create_datetime_index(len(fx)),
                          dtype=float)
    obs_series = pd.Series(obs, index=create_datetime_index(len(obs)),
                           dtype=float)
    # Check require reference forecast kwarg
    if metric in ['s']:
        if ref_fx is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_deterministic_metric_func(
                    metric, fx_series, obs_series)
        else:
            ref_fx_series = pd.Series(ref_fx,
                                      index=create_datetime_index(len(ref_fx)))
            metric_value = calculator._apply_deterministic_metric_func(
                metric, fx_series, obs_series, ref_fx=ref_fx_series)
            np.testing.assert_approx_equal(metric_value, expect)

    # Check requires normalization kwarg
    elif metric in ['nrmse']:
        if norm is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_deterministic_metric_func(
                    metric, fx_series, obs_series)
        else:
            metric_value = calculator._apply_deterministic_metric_func(
                metric, fx_series, obs_series, normalizer=norm)
            np.testing.assert_approx_equal(metric_value, expect)

    # Does not require kwarg
    else:
        metric_value = calculator._apply_deterministic_metric_func(
            metric, fx_series, obs_series)
        if np.isnan(expect):
            assert np.isnan(metric_value)
        else:
            np.testing.assert_approx_equal(metric_value, expect)


def test_apply_deterministic_bad_metric_func():
    with pytest.raises(KeyError):
        calculator._apply_deterministic_metric_func('BAD METRIC',
                                                    pd.Series(dtype=float),
                                                    pd.Series(dtype=float))


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('metric,fx,fx_prob,obs,ref_fx,ref_fx_prob,expect', [
    ('bs', [], [], [], None, None, np.NaN),
    ('bs', [1, 1, 1], [100, 100, 100], [1, 1, 1], None, None, 0.),

    # Briar Skill Score with no reference
    ('bss', [1, 1, 1], [100, 100, 100], [0, 0, 0],
            None, None, 1.),
    ('bss', [1, 1, 1], [100, 100, 100], [1, 1, 1],
            [0, 0, 0], [100, 100, 100], 1.),

    ('rel', [1, 1, 1], [100, 100, 100], [1, 1, 1], None, None, 0.),
    ('res', [1, 1, 1], [100, 100, 100], [1, 1, 1], None, None, 0.),
    ('unc', [1, 1, 1], [100, 100, 100], [1, 1, 1], None, None, 0.),

    # CRPS single forecast
    ('crps', [1], [100], [0], None, None, 0.),
    # CRPS mulitple forecasts
    ('crps', [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
             [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
             [0, 0, 0], None, None, 0.)
])
def test_apply_probabilistic_metric_func(metric, fx, fx_prob, obs,
                                         ref_fx, ref_fx_prob, expect,
                                         create_datetime_index):
    if metric == 'crps':
        fx_data = pd.DataFrame(fx,
            index=create_datetime_index(len(fx)), dtype=float)
        fx_prob_data = pd.DataFrame(fx_prob,
            index=create_datetime_index(len(fx_prob)), dtype=float)
    else:
        fx_data = pd.Series(fx,
            index=create_datetime_index(len(fx)), dtype=float)
        fx_prob_data = pd.Series(fx_prob,
            index=create_datetime_index(len(fx_prob)), dtype=float)
    obs_series = pd.Series(obs,
        index=create_datetime_index(len(obs)), dtype=float)

    # Check metrics that require reference forecast kwarg
    if metric in ['bss']:
        if ref_fx is None or ref_fx_prob is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_probabilistic_metric_func(
                    metric, fx_data, fx_prob_data, obs_series)
        else:
            ref_fx_data = pd.Series(ref_fx,
                index=create_datetime_index(len(ref_fx)))
            ref_fx_prob_data = pd.Series(ref_fx_prob,
                index=create_datetime_index(len(ref_fx_prob)))
            metric_value = calculator._apply_probabilistic_metric_func(
                metric, fx_data, fx_prob_data, obs_series,
                ref_fx=ref_fx_data,
                ref_fx_prob=ref_fx_prob_data)
            np.testing.assert_approx_equal(metric_value, expect)

    # Does not require kwarg
    else:
        metric_value = calculator._apply_probabilistic_metric_func(
            metric, fx_data, fx_prob_data, obs_series)
        if np.isnan(expect):
            assert np.isnan(metric_value)
        else:
            np.testing.assert_approx_equal(metric_value, expect)


def test_apply_probabilistic_bad_metric_func():
    with pytest.raises(KeyError):
        calculator._apply_probabilistic_metric_func('BAD METRIC',
                                                    pd.Series(dtype=float),
                                                    pd.Series(dtype=float),
                                                    pd.Series(dtype=float))


@pytest.mark.parametrize('ts,tz,interval_label,category,result', [
    # category: hour
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "UTC",
        'ending',
        'hour',
        {('12', -1.0), ('13', 0.5)}
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "UTC",
        'beginning',
        'hour',
        {('13', -0.5), ('14', 1.0)}
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "US/Pacific",
        'ending',
        'hour',
        {('4', -1.0), ('5', 0.5)}
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "US/Pacific",
        'beginning',
        'hour',
        {('5', -0.5), ('6', 1.0)}
    ),



    # category: month
    (
        ['20191130T2330', '20191201T0000', '20191201T0030'],
        "UTC",
        'ending',
        'month',
        {('Nov', -0.5), ('Dec', 1.0)}
    ),
    (
        ['20191130T2330', '20191201T0000', '20191201T0030'],
        "UTC",
        'beginning',
        'month',
        {('Nov', -1.0), ('Dec', 0.5)}
    ),

    # category: year
    (
        ['20191231T2330', '20200101T0000', '20200101T0030'],
        "UTC",
        'ending',
        'year',
        {('2019', -0.5), ('2020', 1.0)}
    ),
    (
        ['20191231T2330', '20200101T0000', '20200101T0030'],
        "UTC",
        'beginning',
        'year',
        {('2019', -1.0), ('2020', 0.5)}
    ),
])
def test_interval_label(ts, tz, interval_label, category, result,
                        create_processed_fxobs):

    index = pd.DatetimeIndex(ts, tz="UTC")

    # Custom metadata to keep all timestamps in UTC for tests
    site = datamodel.Site(
        name='Albuquerque Baseline',
        latitude=35.05,
        longitude=-106.54,
        elevation=1657.0,
        timezone="UTC",
        provider='Sandia'
    )

    fx_series = pd.Series([0, 1, 2], index=index)
    obs_series = pd.Series([1, 1, 1], index=index)

    # convert to local timezone
    fx_series = fx_series.tz_convert(tz)
    obs_series = obs_series.tz_convert(tz)

    fxobs = datamodel.ForecastObservation(
        observation=datamodel.Observation(
            site=site, name='dummy obs', variable='ghi',
            interval_value_type='instantaneous', uncertainty=1,
            interval_length=pd.Timedelta(obs_series.index.freq),
            interval_label=interval_label
        ),
        forecast=datamodel.Forecast(
            site=site, name='dummy fx', variable='ghi',
            interval_value_type='instantaneous',
            interval_length=pd.Timedelta(fx_series.index.freq),
            interval_label=interval_label,
            issue_time_of_day=datetime.time(hour=5),
            lead_time_to_start=pd.Timedelta('1h'),
            run_length=pd.Timedelta('1h')
        )
    )

    proc_fx_obs = create_processed_fxobs(fxobs, fx_series, obs_series)

    res = calculator.calculate_deterministic_metrics(proc_fx_obs, [category],
                                                     ['mbe'])
    assert {(v.index, v.value) for v in res.values} == result


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('label_fx,label_ref', [
    ("beginning", "beginning"),
    ("ending", "ending"),
    pytest.param("beginning", "ending",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    pytest.param("ending", "beginning",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
])
def test_interval_label_match(site_metadata, label_fx, label_ref,
                              create_processed_fxobs,
                              many_forecast_observation):

    categories = LIST_OF_CATEGORIES
    metrics = list(deterministic._MAP.keys())

    proc_fx_obs = []
    for fx_obs in many_forecast_observation:
        proc_fx_obs.append(
            create_processed_fxobs(
                fx_obs,
                np.random.randn(10) + 10,
                np.random.randn(10) + 10,
                interval_label=label_fx,
            )
        )

    proc_ref_obs = create_processed_fxobs(
        many_forecast_observation[0],
        np.random.randn(10) + 10,
        np.random.randn(10) + 10,
        interval_label=label_ref,
    )

    all_results = calculator.calculate_metrics(
        proc_fx_obs,
        categories,
        metrics,
        ref_pair=proc_ref_obs,
    )

    assert isinstance(all_results, list)
    assert len(all_results) == len(proc_fx_obs)


@pytest.mark.parametrize('prob_fx_df,axis,exp_fx_fx_prob', [
    (pd.DataFrame({'25': [1.]*5, '50': [2.]*5, '75': [3.]*5}),
     'y',
     [(pd.Series([1.]*5, name='25'), pd.Series([25.]*5)),
      (pd.Series([2.]*5, name='50'), pd.Series([50.]*5)),
      (pd.Series([3.]*5, name='75'), pd.Series([75.]*5))]),
])
def test_transform_prob_forecast_value_and_prob(copy_prob_forecast_with_axis,
                                                create_processed_fxobs,
                                                prob_forecasts,
                                                single_observation,
                                                prob_fx_df, axis, exp_fx_fx_prob):
    conv_prob_fx = copy_prob_forecast_with_axis(prob_forecasts, axis)
    fxobs = datamodel.ForecastObservation(conv_prob_fx, single_observation)
    proc_fxobs = create_processed_fxobs(fxobs, prob_fx_df,
                                        pd.Series([0.]*prob_fx_df.shape[0]))
    result = calculator._transform_prob_forecast_value_and_prob(proc_fxobs)
    assert len(result) == len(exp_fx_fx_prob)
    for res, exp in zip(result, exp_fx_fx_prob):
        assert len(res) == len(res)
        pd.testing.assert_series_equal(res[0], exp[0])
        pd.testing.assert_series_equal(res[1], exp[1])
