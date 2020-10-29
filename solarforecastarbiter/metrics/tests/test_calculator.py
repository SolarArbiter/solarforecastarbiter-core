import calendar
import datetime
import itertools
import re

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import (calculator, deterministic,
                                          probabilistic, event)


DETERMINISTIC_METRICS = list(deterministic._MAP.keys())
DET_NO_NORM = (set(DETERMINISTIC_METRICS) - set(deterministic._REQ_NORM))
DET_NO_REF = (set(DETERMINISTIC_METRICS) - set(deterministic._REQ_REF_FX))
DET_NO_REF_NO_NORM = (set(DET_NO_NORM) - set(deterministic._REQ_REF_FX))
PROBABILISTIC_METRICS = list(probabilistic._MAP.keys())
# PROB_NO_NORM = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_NORM))
PROB_NO_REF = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_REF_FX))
PROB_NO_DIST = (set(PROBABILISTIC_METRICS) - set(probabilistic._REQ_DIST))
PROB_NO_REF_NO_DIST = (set(PROB_NO_REF) - set(probabilistic._REQ_DIST))
LIST_OF_CATEGORIES = list(datamodel.ALLOWED_CATEGORIES.keys())
EVENT_METRICS = list(event._MAP.keys())


@pytest.fixture()
def create_processed_fxobs(create_dt_index, banded_cost_params):
    def _create_processed_fxobs(fxobs, fx_values, obs_values,
                                ref_values=None,
                                interval_label=None):

        if not interval_label:
            interval_label = fxobs.forecast.interval_label

        if (isinstance(fx_values, pd.Series) or
           isinstance(fx_values, pd.DataFrame)):
            conv_fx_values = fx_values
        else:
            conv_fx_values = pd.Series(
                fx_values, index=create_dt_index(len(fx_values)))
        if isinstance(obs_values, pd.Series):
            conv_obs_values = obs_values
        else:
            conv_obs_values = pd.Series(
                obs_values, index=create_dt_index(len(obs_values)))

        return datamodel.ProcessedForecastObservation(
            fxobs.forecast.name,
            fxobs,
            fxobs.forecast.interval_value_type,
            fxobs.forecast.interval_length,
            interval_label,
            valid_point_count=len(fx_values),
            forecast_values=conv_fx_values,
            observation_values=conv_obs_values,
            reference_forecast_values=ref_values,
            normalization_factor=fxobs.normalization,
            uncertainty=fxobs.uncertainty,
            cost=banded_cost_params
            )

    return _create_processed_fxobs


@pytest.fixture()
def create_dt_index():
    def _create_dt_index(n_periods):
        return pd.date_range(start='20190801', periods=n_periods, freq='1h',
                             tz='MST', name='timestamp')
    return _create_dt_index


@pytest.fixture()
def copy_prob_forecast_with_axis():
    def _copy_prob_forecast_with_axis(probfx, axis, constant_values=None):
        if constant_values:
            cvs = [probfx.constant_values[0].replace(constant_value=cv)
                   for cv in constant_values]
        else:
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


@pytest.fixture(params=['fxobs', 'fxagg', 'fxobs_ref', 'fxagg_ref'])
def single_forecast_data_obj(
        request, single_forecast_aggregate,
        single_forecast_observation,
        single_forecast_aggregate_reffx,
        single_forecast_observation_reffx):
    if request.param == 'fxobs':
        return single_forecast_observation
    elif request.param == 'fxagg':
        return single_forecast_aggregate
    if request.param == 'fxobs_ref':
        return single_forecast_observation_reffx
    elif request.param == 'fxagg_ref':
        return single_forecast_aggregate_reffx


@pytest.fixture(params=['probfxobs', 'probfxagg',
                        'probfx_ref', 'probfxagg_ref', 'probfxobsy'])
def prob_forecasts_data_obj(
        request, single_prob_forecast_observation,
        single_prob_forecast_aggregate,
        single_prob_forecast_observation_y,
        single_prob_forecast_observation_reffx,
        single_prob_forecast_aggregate_reffx):
    if request.param == 'probfxobs':
        return single_prob_forecast_observation
    elif request.param == 'probfxobsy':
        return single_prob_forecast_observation_y
    elif request.param == 'probfxagg':
        return single_prob_forecast_aggregate
    elif request.param == 'probfx_ref':
        return single_prob_forecast_observation_reffx
    elif request.param == 'probfxagg_ref':
        return single_prob_forecast_aggregate_reffx


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
        categories, metrics, proc_fx_obs):
    result = calculator.calculate_metrics(proc_fx_obs,
                                          categories, metrics)

    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == len(proc_fx_obs)


def _random_ref_values_if_not_None(data_obj):
    if data_obj.reference_forecast is not None:
        return np.random.randn(10) + 10
    else:
        return None


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_single(single_forecast_data_obj,
                                  create_processed_fxobs):
    ref_values = _random_ref_values_if_not_None(single_forecast_data_obj)
    inp = [create_processed_fxobs(single_forecast_data_obj,
                                  np.random.randn(10)+10,
                                  np.random.randn(10)+10,
                                  ref_values=ref_values)]
    result = calculator.calculate_metrics(inp, LIST_OF_CATEGORIES,
                                          DETERMINISTIC_METRICS)
    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == 1


def test_calculate_metrics_single_uncert(single_forecast_observation_uncert,
                                         create_processed_fxobs):
    inp = [create_processed_fxobs(single_forecast_observation_uncert,
                                  np.array([1.9, 1, 1.0005]),
                                  np.array([1, 1, 1]))]
    result = calculator.calculate_metrics(inp, ['total'], ['mae'])
    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == 1
    expected_values = {None: (0.9 + 0.0005)/3, 100.: 0., 0.1: 0.3}
    expected = expected_values[single_forecast_observation_uncert.uncertainty]
    assert result[0].values[0].value == expected


def test_calculate_metrics_explicit_cost(single_forecast_observation_uncert,
                                         create_processed_fxobs,
                                         constant_cost):
    inp = [create_processed_fxobs(single_forecast_observation_uncert,
                                  np.array([1.9, 1, 1.0005]),
                                  np.array([1, 1, 1]))]
    result = calculator.calculate_metrics(inp, ['total'], ['cost'])
    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == 1
    expected_values = {None: (0.9 + 5e-4), 100.: 0., 0.1: 0.3 * 3}
    expected = expected_values[single_forecast_observation_uncert.uncertainty]
    # float precision issues
    assert abs(result[0].values[0].value - expected) < 1e-8


def test_calculate_deterministic_metrics_sorting(single_forecast_observation,
                                                 create_processed_fxobs,
                                                 create_dt_index):
    index = pd.DatetimeIndex(
        # sunday, monday, tuesday
        ['20200531 1900Z', '20200601 2000Z', '20200602 2100Z'])
    inp = create_processed_fxobs(
        single_forecast_observation,
        pd.Series([2, 1, 0], index=index),
        pd.Series([1, 1, 1], index=index))
    categories = ('hour', 'total', 'date', 'month', 'weekday')
    metrics = ('rmse', 'ksi', 'mbe', 'mae')
    result = calculator.calculate_deterministic_metrics(
        inp, categories, metrics)
    expected = {
        0: ('total', 'mae', '0', 2/3),
        1: ('total', 'mbe', '0', 0.),
        4: ('month', 'mae', 'May', 1.),
        5: ('month', 'mae', 'Jun', 0.5),
        12: ('hour', 'mae', '19', 1.),
        14: ('hour', 'mae', '21', 1.),
        18: ('hour', 'rmse', '19', 1.),
        39: ('weekday', 'mbe', 'Mon', 0.),
        40: ('weekday', 'mbe', 'Tue', -1.),
        41: ('weekday', 'mbe', 'Sun', 1.),
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(result.values[k], attr) == expected_val


def test_calculate_event_metrics_sorting(single_event_forecast_observation,
                                         create_processed_fxobs,
                                         create_dt_index):
    inp = create_processed_fxobs(
        single_event_forecast_observation,
        pd.Series([True, False, True], index=create_dt_index(3)),  # fx
        pd.Series([True, True, True], index=create_dt_index(3)))   # obs
    categories = ('hour', 'total')
    metrics = ('far', 'pod', 'ea', 'pofd')
    result = calculator.calculate_event_metrics(
        inp, categories, metrics)
    expected = {
        0: ('total', 'pod', '0', 2 / 3),
        1: ('total', 'far', '0', 0.0),
        2: ('total', 'pofd', '0', 0.0),
        3: ('total', 'ea', '0', 2 / 3),
        4: ('hour', 'pod', '0', 1.0),
        6: ('hour', 'pod', '2', 1.0),
        8: ('hour', 'far', '1', 0.0),
        10: ('hour', 'pofd', '0', 0.0),
        15: ('hour', 'ea', '2', 1.0),
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(result.values[k], attr) == expected_val


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_with_probablistic(single_observation,
                                             prob_forecasts,
                                             create_processed_fxobs,
                                             create_dt_index,
                                             copy_prob_forecast_with_axis,
                                             caplog):
    const_values = [10, 20, 30]
    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    prfxobs = datamodel.ForecastObservation(
        conv_prob_fx, single_observation)
    fx_values = pd.DataFrame(np.array((
        np.linspace(0, 9., 10),
        np.linspace(1, 10., 10),
        np.linspace(2, 11., 10))).T,
        columns=const_values,
        index=create_dt_index(10))
    obs_values = pd.Series(
        np.linspace(1.5, 10.5, 10),
        index=create_dt_index(10))
    proc_prfx_obs = create_processed_fxobs(prfxobs, fx_values, obs_values)

    # Without reference
    result = calculator.calculate_metrics([proc_prfx_obs],
                                          LIST_OF_CATEGORIES,
                                          PROB_NO_REF)
    assert len(result) == 1
    assert isinstance(result[0], datamodel.MetricResult)
    verify_metric_result(
        result[0], proc_prfx_obs, LIST_OF_CATEGORIES,
        set(probabilistic._REQ_DIST) - set(probabilistic._REQ_REF_FX))

    # With reference
    ref_fx_values = fx_values + .5
    conv_ref_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    ref_prfxobs = datamodel.ForecastObservation(
        conv_ref_prob_fx,
        single_observation,
        reference_forecast=conv_ref_prob_fx)
    proc_ref_prfx_obs = create_processed_fxobs(ref_prfxobs,
                                               fx_values,
                                               obs_values,
                                               ref_values=ref_fx_values)
    dist_results = calculator.calculate_metrics(
        [proc_ref_prfx_obs], LIST_OF_CATEGORIES,
        # reverse to ensure order output is independent
        PROBABILISTIC_METRICS[::-1])

    assert 'Failed' not in caplog.text
    assert isinstance(dist_results, list)
    assert len(dist_results) == 1
    assert isinstance(dist_results[0], datamodel.MetricResult)
    verify_metric_result(dist_results[0],
                         proc_ref_prfx_obs,
                         LIST_OF_CATEGORIES,
                         probabilistic._REQ_DIST)

    expected = {
        0: ('total', 'crps', '0', 17.247),
        2: ('year', 'crps', '2019', 17.247),
        4: ('season', 'crps', 'JJA', 17.247),
        6: ('month', 'crps', 'Aug', 17.247),
        8: ('hour', 'crps', '0', 19.801000000000002),
        9: ('hour', 'crps', '1', 19.405)
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(dist_results[0].values[k], attr) == expected_val

    # Constant Values
    cv_proc_ref_prfx_obs = []
    zip_cvs = zip(conv_prob_fx.constant_values,
                  conv_ref_prob_fx.constant_values)
    for i, (cv, ref_cv) in enumerate(zip_cvs):
        ref_fx_values = fx_values.iloc[:, i] + .5
        cv_ref_prfxobs = datamodel.ForecastObservation(
            cv,
            single_observation,
            reference_forecast=ref_cv)
        cv_proc_ref_prfx_obs.append(
            create_processed_fxobs(cv_ref_prfxobs,
                                   fx_values.iloc[:, i],
                                   obs_values,
                                   ref_values=ref_fx_values))

    cv_results = calculator.calculate_metrics(
        cv_proc_ref_prfx_obs, LIST_OF_CATEGORIES,
        # reverse to ensure order output is independent
        list(PROB_NO_DIST)[::-1])

    # test MetricValues contents and order
    single_result = cv_results[0]
    expected = {
        0: ('total', 'bs', '0', 0.8308500000000001),
        1: ('total', 'bss', '0', -0.010366947374821578),
        7: ('year', 'bs', '2019', 0.8308500000000001),
        11: ('year', 'unc', '2019', 0.08999999999999998),
        21: ('month', 'bs', 'Aug', 0.8308500000000001)
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(single_result.values[k], attr) == expected_val

    # Distribution and constant values
    all_results = calculator.calculate_metrics(
        [proc_ref_prfx_obs] + cv_proc_ref_prfx_obs, LIST_OF_CATEGORIES,
        # reverse to ensure order output is independent
        PROBABILISTIC_METRICS[::-1])

    assert all_results[0] == dist_results[0]
    assert len(all_results[1:]) == len(cv_results)
    for a, b in zip(all_results[1:], cv_results):
        assert a == b


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_probablistic_one_interval(
        single_observation, prob_forecasts,
        create_processed_fxobs, create_dt_index,
        copy_prob_forecast_with_axis, caplog):
    const_values = [10]
    fx_values = pd.DataFrame(np.array((
        np.linspace(2, 11., 10))).T,
        columns=const_values,
        index=create_dt_index(10))
    obs_values = pd.Series(
        np.linspace(1.5, 10.5, 10),
        index=create_dt_index(10))
    ref_fx_values = fx_values + .5
    conv_ref_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    ref_prfxobs = datamodel.ForecastObservation(
        conv_ref_prob_fx,
        single_observation,
        reference_forecast=conv_ref_prob_fx)
    proc_ref_prfx_obs = create_processed_fxobs(ref_prfxobs,
                                               fx_values,
                                               obs_values,
                                               ref_values=ref_fx_values)
    dist_results = calculator.calculate_metrics(
        [proc_ref_prfx_obs], LIST_OF_CATEGORIES,
        # reverse to ensure order output is independent
        PROBABILISTIC_METRICS[::-1])

    assert len(dist_results) == 0
    assert 'Failed' in caplog.text


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_with_probablistic_dist_missing_ref(
        single_observation, prob_forecasts, create_processed_fxobs,
        create_dt_index, copy_prob_forecast_with_axis, caplog):
    const_values = [10, 20, 30]
    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    prfxobs = datamodel.ForecastObservation(conv_prob_fx, single_observation)

    fx_values = pd.DataFrame(np.random.randn(10, 3)+10,
                             index=create_dt_index(10))
    obs_values = pd.Series(np.random.randn(10)+10,
                           index=create_dt_index(10))
    pairs = []
    for i, cv in enumerate(prfxobs.forecast.constant_values):
        cv_fxobs = datamodel.ForecastObservation(
            cv, single_observation)
        pairs.append(create_processed_fxobs(
            cv_fxobs, fx_values[i], obs_values))
    results = calculator.calculate_metrics(
        pairs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX)
    assert len(results) == 3
    for res in results:
        for val in res.values:
            assert np.isnan(val.value)


def test_calculate_deterministic_metrics_no_metrics(
        create_processed_fxobs, single_forecast_observation):
    proc_fx_obs = create_processed_fxobs(single_forecast_observation,
                                         np.array([1]), np.array([1]))
    with pytest.raises(RuntimeError):
        calculator.calculate_deterministic_metrics(
            proc_fx_obs, LIST_OF_CATEGORIES, []
        )


def test_calculate_deterministic_metrics_no_cost_param(
        create_processed_fxobs, single_forecast_observation):
    proc_fx_obs = create_processed_fxobs(single_forecast_observation,
                                         np.array([1]), np.array([1])).replace(
                                             cost=None)

    out = calculator.calculate_deterministic_metrics(
            proc_fx_obs, LIST_OF_CATEGORIES, ['cost']
    )
    for o in out.values:
        assert np.isnan(o.value)


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
        create_processed_fxobs, single_forecast_observation_norm):
    pair = create_processed_fxobs(single_forecast_observation_norm,
                                  np.random.randn(10), np.random.randn(10))
    s_normed = calculator.calculate_deterministic_metrics(
        pair, ['total'], deterministic._REQ_NORM)
    unnormed = [x.lstrip('n') for x in deterministic._REQ_NORM]
    s_unnormed = calculator.calculate_deterministic_metrics(
        pair, ['total'], unnormed)
    norm = single_forecast_observation_norm.normalization
    for v_normed, v_unnormed in zip(s_normed.values, s_unnormed.values):
        if np.isnan(norm):
            assert np.isnan(v_normed.value)
        else:
            assert_allclose(v_normed.value, v_unnormed.value * 100 / norm)


def test_calculate_deterministic_metrics_reference(
        create_processed_fxobs, single_forecast_observation_reffx):
    pair0 = create_processed_fxobs(single_forecast_observation_reffx,
                                   np.random.randn(10), np.random.randn(10),
                                   ref_values=np.random.randn(10))
    pair1 = create_processed_fxobs(single_forecast_observation_reffx,
                                   np.random.randn(10), np.random.randn(10),
                                   ref_values=np.random.randn(10))
    s0 = calculator.calculate_deterministic_metrics(
        pair0, ['total'], deterministic._REQ_REF_FX)
    s1 = calculator.calculate_deterministic_metrics(
        pair1, ['total'], deterministic._REQ_REF_FX)
    for s in [s0, s1]:
        assert isinstance(s, datamodel.MetricResult)
    assert s0 != s1


def verify_metric_result(result, pair, categories, metrics):
    assert result.name == pair.original.forecast.name
    if not categories or not metrics:
        return
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
        elif cat == 'season':
            grps = ['JJA']
        assert {str(g) for g in grps} == cat_grps
    for val in result.values:
        assert (
            np.isnan(val.value) or
            np.issubdtype(type(val.value), np.number)
        )


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
    ref_values = _random_ref_values_if_not_None(single_forecast_data_obj)
    pair = create_processed_fxobs(single_forecast_data_obj,
                                  np.random.randn(10)+10,
                                  np.random.randn(10)+10,
                                  ref_values=ref_values)
    result = calculator.calculate_deterministic_metrics(
        pair, categories, metrics)
    # Check results
    assert isinstance(result, datamodel.MetricResult)
    verify_metric_result(result, pair, categories, metrics)


def test_calculate_probabilistic_metrics_no_metrics(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series(np.random.randn(10)))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, []
        )


# numpy < operator warns on comparison with nan
@pytest.mark.filterwarnings("ignore:invalid value")
def test_calculate_probabilistic_metrics_missing_ref(
        single_prob_forecast_observation, create_processed_fxobs,
        create_dt_index):
    proc_fxobs = create_processed_fxobs(
        single_prob_forecast_observation,
        pd.DataFrame(np.random.randn(10, 3), index=create_dt_index(10)),
        pd.Series(np.random.randn(10), index=create_dt_index(10)))
    result = calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES,
            set(probabilistic._REQ_REF_FX) - set(probabilistic._REQ_DIST)
        )
    assert isinstance(result, datamodel.MetricResult)
    assert result.values == ()


# numpy < operator warns on comparison with nan
@pytest.mark.filterwarnings("ignore:invalid value")
def test_calculate_probabilistic_metrics_no_reference_data(
        single_prob_forecast_observation_reffx, create_processed_fxobs,
        create_dt_index):
    proc_fxobs = create_processed_fxobs(
        single_prob_forecast_observation_reffx,
        pd.DataFrame(np.random.randn(10, 3), index=create_dt_index(10)),
        pd.Series(np.random.randn(10), index=create_dt_index(10)),
        ref_values=None)
    result = calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES,
            set(probabilistic._REQ_REF_FX) - set(probabilistic._REQ_DIST)
        )
    assert isinstance(result, datamodel.MetricResult)
    assert result.values == ()


def test_calculate_probabilistic_metrics_interval_label_ending(
        single_prob_forecast_observation, create_processed_fxobs,
        create_dt_index):
    proc_fxobs = create_processed_fxobs(
        single_prob_forecast_observation,
        pd.DataFrame(np.random.randn(10, 3), index=create_dt_index(10)),
        pd.Series(np.random.randn(10), index=create_dt_index(10))
    )
    proc_fxobs = proc_fxobs.replace(interval_label='ending')
    result = calculator.calculate_probabilistic_metrics(
        proc_fxobs, LIST_OF_CATEGORIES, PROB_NO_REF)
    assert result


def test_calculate_probabilistic_metrics_bad_reference_axis(
        single_prob_forecast_observation, prob_forecasts, single_observation,
        create_processed_fxobs,
        copy_prob_forecast_with_axis):
    conv_fx = copy_prob_forecast_with_axis(prob_forecasts, axis='y')
    fxob = datamodel.ForecastObservation(
        prob_forecasts,
        single_observation,
        conv_fx)
    proc_fxobs = create_processed_fxobs(
        fxob,
        pd.DataFrame(np.random.randn(10, 3)),
        pd.Series(np.random.randn(10)),
        ref_values=pd.DataFrame(np.random.randn(10, 3)),)
    with pytest.raises(ValueError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX
        )


def test_calculate_probabilistic_metrics_missing_values(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(),
                                        pd.Series(np.random.randn(10)))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, PROB_NO_REF
        )


def test_calculate_probabilistic_metrics_missing_observation(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series([], dtype=float))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, PROB_NO_REF
        )


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_probabilistic_metrics_dist_simple(prob_forecasts_data_obj,
                                                     create_processed_fxobs,
                                                     create_dt_index):
    if prob_forecasts_data_obj.reference_forecast is None:
        ref_values = None
    else:
        ref_values = pd.DataFrame(
            np.random.randn(10, 3)+10, index=create_dt_index(10))
    pair = create_processed_fxobs(
        prob_forecasts_data_obj,
        pd.DataFrame(np.random.randn(10, 3)+10, index=create_dt_index(10)),
        pd.Series(np.random.randn(10)+10, index=create_dt_index(10)),
        ref_values=ref_values)
    result = calculator.calculate_probabilistic_metrics(
        pair, LIST_OF_CATEGORIES, PROBABILISTIC_METRICS)
    assert isinstance(result, datamodel.MetricResult)
    verify_metric_result(
        result, pair, LIST_OF_CATEGORIES, probabilistic._REQ_DIST)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_probabilistic_metrics_with_constant_value_simple(
        prob_forecasts_data_obj,
        create_processed_fxobs,
        create_dt_index):
    if prob_forecasts_data_obj.reference_forecast is None:
        ref_values = None
    else:
        ref_values = pd.DataFrame(
            np.random.randn(10, 3)+10, index=create_dt_index(10))
    pair = create_processed_fxobs(
        prob_forecasts_data_obj,
        pd.DataFrame(np.random.randn(10, 3)+10, index=create_dt_index(10)),
        pd.Series(np.random.randn(10)+10, index=create_dt_index(10)),
        ref_values=ref_values)

    for i, cv in enumerate(pair.original.forecast.constant_values):
        data_object = prob_forecasts_data_obj.data_object.replace(
            **{'interval_length': cv.interval_length})
        if isinstance(data_object, datamodel.Aggregate):
            pair_type = datamodel.ForecastAggregate
        else:
            pair_type = datamodel.ForecastObservation
        fxobs = pair_type(
            cv,
            data_object,
            pair.original.reference_forecast)
        fx = pair.forecast_values[i]
        fx.name = 'value'
        cv_pair = create_processed_fxobs(
            fxobs,
            fx,
            pair.observation_values,
            ref_values=ref_values)
        result = calculator.calculate_probabilistic_metrics(
            cv_pair, LIST_OF_CATEGORIES, PROBABILISTIC_METRICS)
        assert isinstance(result, datamodel.MetricResult)
        verify_metric_result(result, cv_pair, LIST_OF_CATEGORIES, PROB_NO_DIST)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES
])
@pytest.mark.parametrize('metrics', [
    *list(itertools.combinations(PROBABILISTIC_METRICS, 1)),
    PROBABILISTIC_METRICS[0:1],
    PROBABILISTIC_METRICS[0:2],
    PROBABILISTIC_METRICS
])
@pytest.mark.parametrize('axis,prob_fx_df,ref_fx_df,obs', [
    ('y',
     pd.DataFrame({'25': np.random.randn(10)*np.sqrt(20)+20,
                   '50': np.random.randn(10)*np.sqrt(20)+20,
                   '75': np.random.randn(10)*np.sqrt(20)+20}, dtype=float),
     pd.DataFrame({'25': np.random.randn(10)*np.sqrt(20)+20,
                   '50': np.random.randn(10)*np.sqrt(20)+20,
                   '75': np.random.randn(10)*np.sqrt(20)+20}, dtype=float),
     pd.Series(np.random.randn(10)*np.sqrt(20)+20, dtype=float)),
    ('x',
     pd.DataFrame({'10': np.random.randn(10)*5+25,
                   '20': np.random.randn(10)*5+50,
                   '30': np.random.randn(10)*5+75}, dtype=float),
     pd.DataFrame({'10': np.random.randn(10)*5+25,
                   '20': np.random.randn(10)*5+50,
                   '30': np.random.randn(10)*5+75}, dtype=float),
     pd.Series(np.random.randn(10)*np.sqrt(20)+20, dtype=float))
])
def test_calculate_probabilistic_metrics(categories, metrics,
                                         axis, prob_fx_df, ref_fx_df, obs,
                                         prob_forecasts, single_observation,
                                         copy_prob_forecast_with_axis,
                                         create_processed_fxobs,
                                         create_dt_index):
    # add index to data
    dt_index = create_dt_index(len(prob_fx_df))
    prob_fx_df.index = dt_index
    ref_fx_df.index = dt_index
    obs.index = dt_index

    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, axis, prob_fx_df.columns.tolist())
    conv_ref_fx = copy_prob_forecast_with_axis(
        prob_forecasts, axis, prob_fx_df.columns.tolist())

    # Full distributions
    if any(x for x in PROB_NO_DIST if x in set(metrics)):
        # create processed pairs
        prob_fxobs = datamodel.ForecastObservation(
            conv_prob_fx, single_observation, reference_forecast=conv_ref_fx)
        pair = create_processed_fxobs(
            prob_fxobs, prob_fx_df, obs, ref_values=ref_fx_df)

        # calculation
        dist_result = calculator.calculate_probabilistic_metrics(
            pair, categories, metrics)
        metrics_sans_dist = list(set(metrics) - set(PROB_NO_DIST))
        assert isinstance(dist_result, datamodel.MetricResult)
        verify_metric_result(dist_result, pair, categories, metrics_sans_dist)

    # Each constant value
    if any(x for x in PROB_NO_DIST if x in set(metrics)):
        for i, cv in enumerate(conv_prob_fx.constant_values):
            # create processed pairs
            cv_series = prob_fx_df[str(cv.constant_value)]
            cv_ref_series = ref_fx_df[str(cv.constant_value)]
            cv_prob_fxobs = datamodel.ForecastObservation(
                cv, single_observation, reference_forecast=cv)
            cv_pair = create_processed_fxobs(
                cv_prob_fxobs, cv_series, obs, ref_values=cv_ref_series)

            # calculation
            cv_result = calculator.calculate_probabilistic_metrics(
                cv_pair, categories, metrics)
            assert isinstance(cv_result, datamodel.MetricResult)
            cv_metrics = list(set(metrics) - set(probabilistic._REQ_DIST))
            verify_metric_result(cv_result, cv_pair, categories, cv_metrics)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES
])
@pytest.mark.parametrize('interval_label', ['beginning', 'ending'])
@pytest.mark.parametrize('df,metrics', [
    (pd.DataFrame({
        ('50', 'forecast'): np.random.randn(10),
        ('50', 'probability'): np.random.randn(10),
        ('50', 'reference_forecast'): np.random.randn(10),
        ('50', 'reference_probability'): np.random.randn(10),
        (None, 'observation'): np.random.randn(10)
        }),
     PROB_NO_DIST),
    (pd.DataFrame({
        ('50', 'forecast'): np.random.randn(10),
        ('50', 'probability'): np.random.randn(10),
        (None, 'observation'): np.random.randn(10)
        }),
     PROB_NO_REF_NO_DIST),
    (pd.DataFrame({
        ('25', 'forecast'): np.random.randn(10),
        ('50', 'forecast'): np.random.randn(10),
        ('75', 'forecast'): np.random.randn(10),
        ('25', 'probability'): np.random.randn(10),
        ('50', 'probability'): np.random.randn(10),
        ('75', 'probability'): np.random.randn(10),
        ('25', 'reference_forecast'): np.random.randn(10),
        ('50', 'reference_forecast'): np.random.randn(10),
        ('75', 'reference_forecast'): np.random.randn(10),
        ('25', 'reference_probability'): np.random.randn(10),
        ('50', 'reference_probability'): np.random.randn(10),
        ('75', 'reference_probability'): np.random.randn(10),
        (None, 'observation'): np.random.randn(10)
        }),
     probabilistic._REQ_DIST),
    (pd.DataFrame({
        ('25', 'forecast'): np.random.randn(10),
        ('50', 'forecast'): np.random.randn(10),
        ('75', 'forecast'): np.random.randn(10),
        ('25', 'probability'): np.random.randn(10),
        ('50', 'probability'): np.random.randn(10),
        ('75', 'probability'): np.random.randn(10),
        (None, 'observation'): np.random.randn(10)
        }),
     probabilistic._REQ_DIST)
])
def test_calculate_probabilistic_metrics_from_df(categories, df, metrics,
                                                 interval_label,
                                                 create_dt_index):
    df.index = create_dt_index(len(df))
    results = calculator._calculate_probabilistic_metrics_from_df(
        df, categories, metrics, interval_label)
    assert all(isinstance(r, datamodel.MetricValue) for r in results)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('metric,fx,obs,ref_fx,norm,dead,expect', [
    ('mae', [], [], None, None, None, np.NaN),
    ('mae', [], [], None, None, 1, np.NaN),
    ('mae', [1, 1, 1], [0, 1, -1], None, None, None, 1.0),
    ('mae', [0, 1, 4], [1, 1, 1], None, None, 100, 1.),
    ('mbe', [1, 1, 1], [0, 1, -1], None, None, None, 1.0),
    ('rmse', [1, 0, 1], [0, -1, 2], None, None, None, 1.0),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, None, None, None),
    ('nrmse', [1, 0, 1], [0, -1, 2], None, 2.0, None, 100/2),
    ('nrmse', [0.9, 1, 0.1], [1, 0, 1], None, 2.0, 100, 100*np.sqrt(1/3)/2),
    ('mape', [2, 3, 1], [4, 2, 2], None, None, None, 50.0),
    ('s', [1, 0, 1], [0, -1, 2], None, None, None, None),
    ('s', [1, 0, 1], [0, -1, 2], [2, 1, 0], None, None, 0.5),
    ('s', [1, 0, 1], [0, -1, 2], [2, 1, 0], None, 100, 0.6464466094067263),
    ('r', [3, 2, 1], [1, 2, 3], None, None, None, -1.0),
    ('r^2', [3, 2, 1], [1, 2, 3], None, None, None, -3.0),
    ('crmse', [1, 1, 1], [0, 1, 2], None, None, None, np.sqrt(2/3))
])
def test_apply_deterministic_metric_func(metric, fx, obs, ref_fx, norm,
                                         dead, expect):
    fx_series = np.array(fx)
    obs_series = np.array(obs)
    # Check require reference forecast kwarg
    if metric in ['s']:
        if ref_fx is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_deterministic_metric_func(
                    metric, fx_series, obs_series)
        else:
            ref_fx_series = np.array(ref_fx)
            metric_value = calculator._apply_deterministic_metric_func(
                metric, fx_series, obs_series, ref_fx=ref_fx_series,
                deadband=dead)
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
                metric, fx_series, obs_series, normalization=norm,
                deadband=dead)
            np.testing.assert_approx_equal(metric_value, expect)

    # Does not require kwarg
    else:
        metric_value = calculator._apply_deterministic_metric_func(
            metric, fx_series, obs_series, deadband=dead)
        if np.isnan(expect):
            assert np.isnan(metric_value)
        else:
            np.testing.assert_approx_equal(metric_value, expect)


def test_apply_deterministic_bad_metric_func():
    with pytest.raises(KeyError):
        calculator._apply_deterministic_metric_func('BAD METRIC',
                                                    np.array([]),
                                                    np.array([]))


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
    ('crps', [[1, 1]], [[100, 100]], [[0, 0]], None, None, 0.),
    # CRPS mulitple forecasts
    ('crps', [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
             [[100, 100, 100], [100, 100, 100], [100, 100, 100]],
             [0, 0, 0], None, None, 0.)
])
def test_apply_probabilistic_metric_func(metric, fx, fx_prob, obs,
                                         ref_fx, ref_fx_prob, expect,
                                         create_dt_index):
    fx_data = np.array(fx)
    fx_prob_data = np.array(fx_prob)
    obs_series = np.array(obs)

    # Check metrics that require reference forecast kwarg
    if metric in probabilistic._REQ_REF_FX:
        if ref_fx is None or ref_fx_prob is None:
            with pytest.raises(KeyError):
                # Missing positional argument
                calculator._apply_probabilistic_metric_func(
                    metric, fx_data, fx_prob_data, obs_series)
        else:
            ref_fx_data = np.array(ref_fx)
            ref_fx_prob_data = np.array(ref_fx_prob)
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
                                                    np.array([]),
                                                    np.array([]),
                                                    np.array([]))


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


@pytest.mark.parametrize('prob_fx_df,axis,exp_fx_fx_prob', [
    (pd.DataFrame({'25': [1.]*5, '50': [2.]*5, '75': [3.]*5}),
     'y',
     [(pd.Series([1.]*5, name='25'), pd.Series([25.]*5, name='25')),
      (pd.Series([2.]*5, name='50'), pd.Series([50.]*5, name='50')),
      (pd.Series([3.]*5, name='75'), pd.Series([75.]*5, name='75'))]),
])
def test_transform_prob_forecast_value_and_prob(prob_forecasts,
                                                single_observation,
                                                prob_fx_df, axis,
                                                exp_fx_fx_prob,
                                                copy_prob_forecast_with_axis,
                                                create_processed_fxobs):
    conv_prob_fx = copy_prob_forecast_with_axis(prob_forecasts, axis)
    fxobs = datamodel.ForecastObservation(conv_prob_fx, single_observation)
    proc_fxobs = create_processed_fxobs(fxobs, prob_fx_df,
                                        pd.Series([0.]*prob_fx_df.shape[0]))
    result = calculator._transform_prob_forecast_value_and_prob(proc_fxobs)
    assert len(result) == len(exp_fx_fx_prob)
    for res, exp in zip(result, exp_fx_fx_prob):
        assert len(res) == len(exp)
        pd.testing.assert_series_equal(res[0], exp[0])
        pd.testing.assert_series_equal(res[1], exp[1])


@pytest.mark.parametrize('obs,fx_fx_prob,ref_fx_fx_prob,exp_df', [
    (pd.Series([0.]*5),
     [(pd.Series([1.]*5, name='25'), pd.Series([25.]*5, name='25')),
      (pd.Series([2.]*5, name='50'), pd.Series([50.]*5, name='50')),
      (pd.Series([3.]*5, name='75'), pd.Series([75.]*5, name='75'))],
     [(pd.Series([4.]*5, name='25'), pd.Series([25.]*5, name='25')),
      (pd.Series([5.]*5, name='50'), pd.Series([50.]*5, name='50')),
      (pd.Series([6.]*5, name='75'), pd.Series([75.]*5, name='75'))],
     pd.DataFrame({
        (None, 'observation'): pd.Series([0.]*5),
        ('25', 'forecast'): pd.Series([1.]*5),
        ('25', 'probability'): pd.Series([25.]*5),
        ('25', 'reference_forecast'): pd.Series([4.]*5),
        ('25', 'reference_probability'): pd.Series([25.]*5),
        ('50', 'forecast'): pd.Series([2.]*5),
        ('50', 'probability'): pd.Series([50.]*5),
        ('50', 'reference_forecast'): pd.Series([5.]*5),
        ('50', 'reference_probability'): pd.Series([50.]*5),
        ('75', 'forecast'): pd.Series([3.]*5),
        ('75', 'probability'): pd.Series([75.]*5),
        ('75', 'reference_forecast'): pd.Series([6.]*5),
        ('75', 'reference_probability'): pd.Series([75.]*5)}))
])
def test_create_prob_dataframe(obs, fx_fx_prob, ref_fx_fx_prob, exp_df):
    result = calculator._create_prob_dataframe(obs, fx_fx_prob, ref_fx_fx_prob)
    pd.testing.assert_frame_equal(result, exp_df)


@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES,
])
@pytest.mark.parametrize('metrics', [
    *list(itertools.combinations(EVENT_METRICS, 1)),
    EVENT_METRICS[0:1],
    EVENT_METRICS[0:2],
    EVENT_METRICS,
    pytest.param([],
                 marks=pytest.mark.xfail(raises=RuntimeError, strict=True)),
])
def test_calculate_event_metrics(single_event_forecast_observation, categories,
                                 metrics):

    index = pd.DatetimeIndex(
        ["20200301T0000Z", "20200301T0100Z", "20200301T0200Z"]
    )
    obs_values = np.random.randint(0, 2, size=len(index), dtype=bool)
    fx_values = np.random.randint(0, 2, size=len(index), dtype=bool)
    obs_series = pd.Series(obs_values, index=index)
    fx_series = pd.Series(fx_values, index=index)

    fxobs = single_event_forecast_observation
    proc_fx_obs = datamodel.ProcessedForecastObservation(
        name=fxobs.forecast.name,
        original=fxobs,
        interval_value_type=fxobs.forecast.interval_value_type,
        interval_length=fxobs.forecast.interval_length,
        interval_label=fxobs.forecast.interval_label,
        valid_point_count=len(fx_series),
        forecast_values=fx_series,
        observation_values=obs_series,
    )

    result = calculator.calculate_event_metrics(
        proc_fx_obs, categories, metrics
    )

    # Check results
    assert isinstance(result, datamodel.MetricResult)
    assert result.forecast_id == proc_fx_obs.original.forecast.forecast_id
    assert result.name == proc_fx_obs.original.forecast.name
    assert len(result.values) % len(metrics) == 0
    cats = {val.category for val in result.values}
    assert cats == set(categories)
    fx_values = proc_fx_obs.forecast_values
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
        elif cat == 'season':
            grps = ["MAM"]
        assert {str(g) for g in grps} == cat_grps
    for val in result.values:
        assert (
            np.isnan(val.value) or
            np.issubdtype(type(val.value), np.number)
        )


@pytest.mark.parametrize('obs_values,fx_values', [
    ([], [True, False]),
    ([False, True], []),
])
def test_calculate_event_metrics_no_data(single_event_forecast_observation,
                                         obs_values, fx_values):

    categories = LIST_OF_CATEGORIES
    metrics = EVENT_METRICS
    obs_series = pd.Series(obs_values, dtype=bool)
    fx_series = pd.Series(fx_values, dtype=bool)

    # processed fx-obs pair
    fxobs = single_event_forecast_observation
    proc_fx_obs = datamodel.ProcessedForecastObservation(
        name=fxobs.forecast.name,
        original=fxobs,
        interval_value_type=fxobs.forecast.interval_value_type,
        interval_length=fxobs.forecast.interval_length,
        interval_label=fxobs.forecast.interval_label,
        valid_point_count=len(fx_series),
        forecast_values=fx_series,
        observation_values=obs_series,
    )

    with pytest.raises(RuntimeError):
        calculator.calculate_event_metrics(
            proc_fx_obs, categories, metrics
        )


@pytest.mark.parametrize('categories', [
    [],
    *list(itertools.combinations(LIST_OF_CATEGORIES, 1)),
    LIST_OF_CATEGORIES[0:1],
    LIST_OF_CATEGORIES[0:2],
    LIST_OF_CATEGORIES,
])
@pytest.mark.parametrize('metrics', [
    *list(itertools.combinations(EVENT_METRICS, 1)),
    EVENT_METRICS[0:1],
    EVENT_METRICS[0:2],
    EVENT_METRICS,
])
def test_calculate_metrics_with_event(single_event_forecast_observation,
                                      categories, metrics):
    index = pd.DatetimeIndex(
        ["20200301T0000Z", "20200301T0100Z", "20200301T0200Z"]
    )

    fxobs = single_event_forecast_observation

    # processed fx-obs pairs
    proc_fx_obs = []
    for i in range(4):
        obs_values = np.random.randint(0, 2, size=len(index), dtype=bool)
        fx_values = np.random.randint(0, 2, size=len(index), dtype=bool)
        obs_series = pd.Series(obs_values, index=index)
        fx_series = pd.Series(fx_values, index=index)

        proc_fx_obs.append(
            datamodel.ProcessedForecastObservation(
                name=fxobs.forecast.name,
                original=fxobs,
                interval_value_type=fxobs.forecast.interval_value_type,
                interval_length=fxobs.forecast.interval_length,
                interval_label=fxobs.forecast.interval_label,
                valid_point_count=len(fx_series),
                forecast_values=fx_series,
                observation_values=obs_series,
            )
        )

    # compute metrics
    result = calculator.calculate_metrics(
        proc_fx_obs, categories, metrics
    )

    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == len(proc_fx_obs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize("categories,metrics", [
    (LIST_OF_CATEGORIES, EVENT_METRICS),
])
def test_calculate_metrics_with_event_empty(single_event_forecast_observation,
                                            categories, metrics, caplog):
    index = pd.DatetimeIndex(
        ["20200301T0000Z", "20200301T0100Z", "20200301T0200Z"]
    )

    fxobs = single_event_forecast_observation

    # processed fx-obs pairs
    proc_fx_obs = []
    obs_values = np.random.randint(0, 2, size=len(index), dtype=bool)
    obs_series = pd.Series(obs_values, index=index)
    fx_series = pd.Series(dtype=bool)

    proc_fx_obs.append(
        datamodel.ProcessedForecastObservation(
            name=fxobs.forecast.name,
            original=fxobs,
            interval_value_type=fxobs.forecast.interval_value_type,
            interval_length=fxobs.forecast.interval_length,
            interval_label=fxobs.forecast.interval_label,
            valid_point_count=len(fx_series),
            forecast_values=fx_series,
            observation_values=obs_series,
        )
    )

    result = calculator.calculate_metrics(proc_fx_obs, categories, metrics)
    assert len(result) == 0
    assert "ERROR" == caplog.text[0:5]
    failure_log_text = caplog.text[re.search(r'.py:\d+ ', caplog.text).end():]
    assert (f"Failed to calculate event metrics for {proc_fx_obs[0].name}: "
            "No Forecast timeseries data.\n") == failure_log_text


def test_calculate_summary_statistics(single_forecast_data_obj,
                                      create_processed_fxobs):
    ref_values = _random_ref_values_if_not_None(single_forecast_data_obj)
    inp = create_processed_fxobs(single_forecast_data_obj,
                                 np.random.randn(10)+10,
                                 np.random.randn(10)+10,
                                 ref_values=ref_values)
    result = calculator.calculate_summary_statistics(inp, LIST_OF_CATEGORIES)
    assert isinstance(result, datamodel.MetricResult)


def test_calculate_summary_statistics_exact(single_forecast_observation,
                                            create_processed_fxobs):
    index = pd.DatetimeIndex(
        # sunday, monday, tuesday
        ['20200531 1900Z', '20200601 2000Z', '20200602 2100Z'])
    inp = create_processed_fxobs(
        single_forecast_observation,
        pd.Series([2, 1, 0], index=index),
        pd.Series([1, 1, 1], index=index))
    categories = ('hour', 'total', 'date', 'month', 'weekday')
    result = calculator.calculate_summary_statistics(inp, categories)
    expected = {
        0: ('total', 'forecast_mean', '0', 1.),
        1: ('total', 'observation_mean', '0', 1.),
        10: ('month', 'forecast_mean', 'May', 2.),
        13: ('month', 'observation_mean', 'Jun', 1.),
        36: ('hour', 'forecast_min', '19', 2.),
        46: ('hour', 'observation_max', '20', 1.),
        -8: ('weekday', 'observation_median', 'Tue', 1.0)
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(result.values[k], attr) == expected_val


def test_calculate_summary_statistics_no_data(single_forecast_observation,
                                              create_processed_fxobs):
    inp = create_processed_fxobs(
        single_forecast_observation,
        pd.Series(dtype=float), pd.Series(dtype=float),)
    categories = ('hour', 'total', 'date', 'month', 'weekday')
    with pytest.raises(RuntimeError):
        calculator.calculate_summary_statistics(inp, categories)


@pytest.mark.parametrize('ts,tz,interval_label,category,fxresult,obsresult', [
    # category: hour
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "UTC",
        'ending',
        'hour',
        {('12', 0.), ('13', 1.5)},
        {('12', 1.), ('13', 1.)}
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "UTC",
        'beginning',
        'hour',
        {('13', 0.5), ('14', 2.0)},
        {('13', 1.0), ('14', 1.0)},
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "US/Pacific",
        'ending',
        'hour',
        {('4', 0.), ('5', 1.5)},
        {('4', 1.), ('5', 1.0)}
    ),
    (
        ['20191210T1300', '20191210T1330', '20191210T1400'],
        "US/Pacific",
        'beginning',
        'hour',
        {('5', 0.5), ('6', 2.0)},
        {('5', 1.0), ('6', 1.0)},
    ),

    # category: month
    (
        ['20191130T2330', '20191201T0000', '20191201T0030'],
        "UTC",
        'ending',
        'month',
        {('Nov', 0.5), ('Dec', 2.0)},
        {('Nov', 1.0), ('Dec', 1.0)},
    ),
    (
        ['20191130T2330', '20191201T0000', '20191201T0030'],
        "UTC",
        'beginning',
        'month',
        {('Nov', 0.), ('Dec', 1.5)},
        {('Nov', 1.), ('Dec', 1.0)},
    ),

    # category: year
    (
        ['20191231T2330', '20200101T0000', '20200101T0030'],
        "UTC",
        'ending',
        'year',
        {('2019', 0.5), ('2020', 2.0)},
        {('2019', 1.0), ('2020', 1.0)},
    ),
    (
        ['20191231T2330', '20200101T0000', '20200101T0030'],
        "UTC",
        'beginning',
        'year',
        {('2019', 0.), ('2020', 1.5)},
        {('2019', 1.), ('2020', 1.0)},
    ),
])
def test_calculate_summary_statistics_interval_label(
        ts, tz, interval_label, category, fxresult,
        obsresult,
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

    res = calculator.calculate_summary_statistics(proc_fx_obs, [category])
    assert {(v.index, v.value) for v in res.values
            if v.metric == 'forecast_mean'} == fxresult
    assert {(v.index, v.value) for v in res.values
            if v.metric == 'observation_mean'} == obsresult


@pytest.mark.parametrize('inp,expected', [
    (
        pd.DataFrame({'forecast': [0, 1, 2], 'observation': [1, 1, 1]},
                     index=pd.date_range(start='20200101T0000Z', freq='1h',
                                         periods=3), dtype=float),
        pd.DataFrame({'mean': [1, 1], 'min': [0, 1], 'max': [2, 1],
                      'median': [1, 1], 'std': [1, 0]},
                     index=['forecast', 'observation'], dtype=float).T
     ),
    (
        pd.DataFrame({'reference_forecast': [0, 1, 2], 'observation':
                      [1, 1, np.nan]},
                     index=pd.date_range(start='20200101T0000Z', freq='1h',
                                         periods=3), dtype=float),
        pd.DataFrame({'mean': [1, 1], 'min': [0, 1], 'max': [2, 1],
                      'median': [1, 1], 'std': [1, 0]},
                     index=['reference_forecast', 'observation'],
                     dtype=float).T
     )
])
def test_calculate_summary_for_frame(inp, expected):
    out = calculator._calculate_summary_for_frame(inp)
    pd.testing.assert_frame_equal(out, expected)


def test_is_deterministic_forecast(single_forecast, single_event_forecast,
                                   prob_forecast_constant_value,
                                   prob_forecasts, single_observation):
    def make(obj):
        class Fx:
            forecast = obj

        class Obj:
            original = Fx()

        return Obj()

    assert calculator._is_deterministic_forecast(make(single_forecast))
    assert not calculator._is_deterministic_forecast(
        make(single_event_forecast))
    assert not calculator._is_deterministic_forecast(make(prob_forecasts))
    assert not calculator._is_deterministic_forecast(
        make(prob_forecast_constant_value))
    assert not calculator._is_deterministic_forecast(make(single_observation))


def test_calculate_all_summary_statistics(
        mocker, single_forecast_data_obj, create_processed_fxobs):
    log = mocker.spy(calculator, 'logger')
    ref_values = _random_ref_values_if_not_None(single_forecast_data_obj)
    inp = [create_processed_fxobs(single_forecast_data_obj,
                                  np.random.randn(10)+10,
                                  np.random.randn(10)+10,
                                  ref_values=ref_values),
           create_processed_fxobs(single_forecast_data_obj,
                                  np.array([]), np.array([]))
           ]
    result = calculator.calculate_all_summary_statistics(
        inp, LIST_OF_CATEGORIES)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == 1
    log.error.assert_called_once()
