import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
import itertools
import calendar
import datetime
import re


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
def create_processed_fxobs(create_dt_index):
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
            normalization_factor=fxobs.normalization,
            uncertainty=fxobs.uncertainty
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


@pytest.fixture(params=['fxobs', 'fxagg'])
def single_forecast_data_obj(
        request, single_forecast_aggregate,
        single_forecast_observation):
    if request.param == 'fxobs':
        return single_forecast_observation
    else:
        return single_forecast_aggregate


@pytest.fixture(params=['probfxobs', 'probfxagg'])
def prob_forecasts_data_obj(
        request, single_prob_forecast_observation,
        single_prob_forecast_aggregate):
    if request.param == 'probfxobs':
        return single_prob_forecast_observation
    else:
        return single_prob_forecast_aggregate


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
def test_calculate_metrics_no_reference(categories, metrics, proc_fx_obs,
                                        caplog):
    result = calculator.calculate_metrics(proc_fx_obs, categories, metrics)
    assert isinstance(result, list)
    assert isinstance(result[0], datamodel.MetricResult)
    assert len(result) == len(proc_fx_obs)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('categories,metrics', [
    (LIST_OF_CATEGORIES, deterministic._REQ_REF_FX)
])
def test_calculate_metrics_ref_metric_no_ref(categories, metrics, proc_fx_obs,
                                             caplog):
    result = calculator.calculate_metrics(
        [proc_fx_obs[0]], categories, metrics)
    assert len(result) == 0
    assert "ERROR" == caplog.text[0:5]
    failure_log_text = caplog.text[re.search(r'.py:\d+ ', caplog.text).end():]
    assert ("Failed to calculate deterministic metrics for "
            f"{proc_fx_obs[0].name}: No reference forecast provided but it is "
            "required for desired metrics.\n") == failure_log_text


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


def test_calculate_deterministic_metrics_sorting(single_forecast_observation,
                                                 create_processed_fxobs,
                                                 create_dt_index):
    inp = create_processed_fxobs(
        single_forecast_observation,
        pd.Series([2, 1, 0], index=create_dt_index(3)),
        pd.Series([1, 1, 1], index=create_dt_index(3)))
    categories = ('hour', 'total', 'date')
    metrics = ('rmse', 'ksi', 'mbe', 'mae')
    proc_fx_obs
    result = calculator.calculate_deterministic_metrics(
        inp, categories, metrics)
    expected = {
        0: ('total', 'mae', '0', 2/3),
        1: ('total', 'mbe', '0', 0.),
        4: ('hour', 'mae', '0', 1.),
        6: ('hour', 'mae', '2', 1.),
        10: ('hour', 'rmse', '0', 1.)
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
                                             copy_prob_forecast_with_axis):
    const_values = [10, 20, 30]
    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    prfxobs = datamodel.ForecastObservation(
        conv_prob_fx, single_observation)
    fx_values = pd.DataFrame(np.array((
        np.linspace(0, 9., 10),
        np.linspace(1, 10., 10),
        np.linspace(2, 11., 10))).T,
        index=create_dt_index(10))
    obs_values = pd.Series(
        np.linspace(1.5, 10.5, 10),
        index=create_dt_index(10))
    proc_prfx_obs = create_processed_fxobs(prfxobs, fx_values,
                                           obs_values)

    # Without reference
    results = calculator.calculate_metrics([proc_prfx_obs], LIST_OF_CATEGORIES,
                                           PROB_NO_REF)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    single_results, dist_results = results[0]
    assert len(single_results) == len(const_values)
    assert all(isinstance(r, datamodel.MetricResult) for r in single_results)
    assert isinstance(dist_results, datamodel.MetricResult)

    # With reference
    ref_fx_values = fx_values + .5
    conv_ref_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    ref_prfxobs = datamodel.ForecastObservation(conv_ref_prob_fx,
                                                single_observation)
    proc_ref_prfx_obs = create_processed_fxobs(ref_prfxobs,
                                               ref_fx_values,
                                               obs_values)
    results = calculator.calculate_metrics(
        [proc_prfx_obs], LIST_OF_CATEGORIES,
        # reverse to ensure order output is independent
        PROBABILISTIC_METRICS[::-1],
        ref_pair=proc_ref_prfx_obs)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], tuple)
    single_results, dist_results = results[0]
    assert len(single_results) == len(const_values)
    assert all(isinstance(r, datamodel.MetricResult) for r in single_results)
    assert isinstance(dist_results, datamodel.MetricResult)
    result = single_results[0]
    expected = {
        0: ('total', 'bs', '0', 0.00285),
        1: ('total', 'bss', '0', 0.1428571428571429),
        5: ('year', 'bs', '2019', 0.00285),
        9: ('year', 'unc', '2019', 0.),
        10: ('month', 'bs', 'Aug', 0.00285)
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(result.values[k], attr) == expected_val
    result = dist_results
    expected = {
        0: ('total', 'crps', '0', 0.0067),
        1: ('year', 'crps', '2019', 0.0067),
        2: ('month', 'crps', 'Aug', 0.0067),
        3: ('hour', 'crps', '0', 0.0001),
        4: ('hour', 'crps', '1', 0.0005)
    }
    attr_order = ('category', 'metric', 'index', 'value')
    for k, expected_attrs in expected.items():
        for attr, expected_val in zip(attr_order, expected_attrs):
            assert getattr(result.values[k], attr) == expected_val


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_calculate_metrics_with_probablistic_no_ref(single_observation,
    prob_forecasts, create_processed_fxobs, create_dt_index,
    copy_prob_forecast_with_axis, caplog):  # NOQA
    const_values = [10, 20, 30]
    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, prob_forecasts.axis, constant_values=const_values)
    prfxobs = datamodel.ForecastObservation(conv_prob_fx, single_observation)

    fx_values = pd.DataFrame(np.random.randn(10, 3)+10,
                             index=create_dt_index(10))
    obs_values = pd.Series(np.random.randn(10)+10,
                           index=create_dt_index(10))
    proc_prfx_obs = create_processed_fxobs(prfxobs, fx_values, obs_values)
    result = calculator.calculate_metrics([proc_prfx_obs], LIST_OF_CATEGORIES,
                                          probabilistic._REQ_REF_FX)
    assert len(result) == 0
    assert "ERROR" == caplog.text[0:5]
    failure_log_text = caplog.text[re.search(r'.py:\d+ ', caplog.text).end():]
    assert ("Failed to calculate probabilistic metrics for "
            f"{proc_prfx_obs.name}: No reference forecast provided but it is "
            "required for desired metrics.\n") == failure_log_text


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


def verify_metric_result(result, pair, categories, metrics):
    assert result.name == pair.original.forecast.name
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


def test_calculate_probabilistic_metrics_no_ref(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series(np.random.randn(10)))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX
        )


def test_calculate_probabilistic_metrics_no_reference(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series(np.random.randn(10)))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX,
        )


def test_calculate_probabilistic_metrics_no_reference_data(
        single_prob_forecast_observation, create_processed_fxobs):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series(np.random.randn(10)))
    proc_ref = create_processed_fxobs(single_prob_forecast_observation,
                                      pd.DataFrame(),
                                      pd.Series(np.random.randn(10)))
    with pytest.raises(RuntimeError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX,
            proc_ref
        )


def test_calculate_probabilistic_metrics_bad_reference_interval_label(
        single_prob_forecast_observation, create_processed_fxobs,
        create_dt_index):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3),
                                                     index=create_dt_index(10)),  # NOQA
                                        pd.Series(np.random.randn(10),
                                                  index=create_dt_index(10)))
    proc_ref = create_processed_fxobs(single_prob_forecast_observation,
                                      pd.DataFrame(np.random.randn(10, 3),
                                                   index=create_dt_index(10)),
                                      pd.Series(np.random.randn(10),
                                                index=create_dt_index(10)))
    proc_fxobs = proc_fxobs.replace(interval_label='beginning')
    proc_ref = proc_ref.replace(interval_label='ending')
    with pytest.raises(ValueError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX,
            proc_ref
        )


def test_calculate_probabilistic_metrics_interval_label_ending(
        single_prob_forecast_observation, create_processed_fxobs,
        create_dt_index):
    proc_fxobs = create_processed_fxobs(
        single_prob_forecast_observation,
        pd.DataFrame(np.random.randn(10, 3), index=create_dt_index(10)),
        pd.Series(np.random.randn(10), index=create_dt_index(10))
    )
    proc_fxobs = proc_fxobs.replace(interval_label='ending')
    calculator.calculate_probabilistic_metrics(proc_fxobs, LIST_OF_CATEGORIES,
                                               PROB_NO_REF)


def test_calculate_probabilistic_metrics_bad_reference_axis(
        single_prob_forecast_observation, prob_forecasts, single_observation,
        create_processed_fxobs,
        copy_prob_forecast_with_axis):
    proc_fxobs = create_processed_fxobs(single_prob_forecast_observation,
                                        pd.DataFrame(np.random.randn(10, 3)),
                                        pd.Series(np.random.randn(10)))
    conv_fx = copy_prob_forecast_with_axis(prob_forecasts, axis='y')
    ref_fxobs = datamodel.ForecastObservation(conv_fx, single_observation)
    proc_ref = create_processed_fxobs(ref_fxobs,
                                      pd.DataFrame(np.random.randn(10, 3)),
                                      pd.Series(np.random.randn(10)))
    with pytest.raises(ValueError):
        calculator.calculate_probabilistic_metrics(
            proc_fxobs, LIST_OF_CATEGORIES, probabilistic._REQ_REF_FX,
            proc_ref
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
def test_calculate_metrics_with_probablistic_simple(prob_forecasts_data_obj,
                                                    create_processed_fxobs,
                                                    create_dt_index):
    pair = create_processed_fxobs(
        prob_forecasts_data_obj,
        pd.DataFrame(np.random.randn(10, 3)+10, index=create_dt_index(10)),
        pd.Series(np.random.randn(10)+10, index=create_dt_index(10)))
    ref = create_processed_fxobs(
        prob_forecasts_data_obj,
        pd.DataFrame(np.random.randn(10, 3)+10, index=create_dt_index(10)),
        pd.Series(np.random.randn(10)+10, index=create_dt_index(10)))
    results = calculator.calculate_probabilistic_metrics(
        pair, LIST_OF_CATEGORIES, PROBABILISTIC_METRICS, ref_fx_obs=ref)
    assert isinstance(results, tuple)
    single_results, dist_results = results
    assert len(single_results) == len(pair.original.forecast.constant_values)
    assert all(isinstance(r, datamodel.MetricResult) for r in single_results)
    assert isinstance(dist_results, datamodel.MetricResult)


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

    # create processed pairs
    conv_prob_fx = copy_prob_forecast_with_axis(
        prob_forecasts, axis, prob_fx_df.columns.tolist())
    conv_ref_fx = copy_prob_forecast_with_axis(
        prob_forecasts, axis, prob_fx_df.columns.tolist())
    prob_fxobs = datamodel.ForecastObservation(
        conv_prob_fx, single_observation)
    ref_fxobs = datamodel.ForecastObservation(
        conv_ref_fx, single_observation)
    pair = create_processed_fxobs(prob_fxobs, prob_fx_df, obs)
    ref = create_processed_fxobs(ref_fxobs, ref_fx_df, obs)

    single_results, dist_result = calculator.calculate_probabilistic_metrics(
        pair, categories, metrics, ref_fx_obs=ref)

    # Check single forecast results
    if any(x for x in PROB_NO_DIST if x in set(metrics)):
        assert len(single_results) == len(prob_fx_df.columns)
        assert all([isinstance(x, datamodel.MetricResult)
                   for x in single_results])
        cvs = pair.original.forecast.constant_values
        assert (set([r.forecast_id for r in single_results]) ==
                set([p.forecast_id for p in cvs]))
        assert (set([r.observation_id for r in single_results]) ==
                set([single_observation.observation_id]))
        metrics_sans_dist = list(set(metrics) - set(probabilistic._REQ_DIST))
        for r in single_results:
            verify_metric_result(r, pair, categories, metrics_sans_dist)
    else:
        assert len(single_results) == 0

    # Check distribution forecast results
    if any(x for x in probabilistic._REQ_DIST if x in set(metrics)):
        assert isinstance(dist_result, datamodel.MetricResult)
        assert dist_result.forecast_id == prob_forecasts.forecast_id
        assert dist_result.observation_id == single_observation.observation_id
        metrics_dist_only = list(set(metrics) - set(PROB_NO_DIST))
        verify_metric_result(dist_result, pair, categories, metrics_dist_only)
    else:
        assert dist_result is None


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
    # unc shouldn't effect s result since it isn't supported
    ('s', [1, 0, 1], [0, -1, 2], [2, 1, 0], None, 100, 0.5),
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
    ('crps', [[1]], [[100]], [[0]], None, None, 0.),
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
])
def test_calculate_event_metrics(categories, metrics):

    # data
    index = pd.DatetimeIndex(
        ["20200301T0000Z", "20200301T0100Z", "20200301T0200Z"]
    )
    obs_series = pd.Series([True, False, True], index=index)
    fx_series = pd.Series([True, True, False], index=index)

    # Custom metadata to keep all timestamps in UTC for tests
    site = datamodel.Site(
        name='Albuquerque Baseline',
        latitude=35.05,
        longitude=-106.54,
        elevation=1657.0,
        timezone="UTC",
        provider='Sandia'
    )

    obs = datamodel.Observation(
        site=site,
        name="dummy obs",
        uncertainty=1,
        interval_length=pd.Timedelta(obs_series.index.freq),
        interval_value_type="instantaneous",
        variable="event",
        interval_label="event",
    )

    fx = datamodel.EventForecast(
        site=site,
        name="dummy fx",
        interval_length=pd.Timedelta(fx_series.index.freq),
        interval_value_type="instantaneous",
        issue_time_of_day=datetime.time(hour=5),
        lead_time_to_start=pd.Timedelta("1h"),
        run_length=pd.Timedelta("1h"),
        variable="event",
        interval_label="event",
    )

    # fx-obs pair
    fxobs = datamodel.ForecastObservation(observation=obs, forecast=fx)

    # processed fx-obs pair
    proc_fx_obs = datamodel.ProcessedForecastObservation(
        name=fxobs.forecast.name,
        original=fxobs,
        interval_value_type=fxobs.forecast.interval_value_type,
        interval_length=fxobs.forecast.interval_length,
        interval_label="event",
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
        assert {str(g) for g in grps} == cat_grps
    for val in result.values:
        assert (
            np.isnan(val.value) or
            np.issubdtype(type(val.value), np.number)
        )
