from dataclasses import fields, MISSING, dataclass
import json
from typing import Union


import numpy as np
import pandas as pd
import pytest


from solarforecastarbiter import datamodel


@pytest.fixture(params=['site', 'fixed', 'single', 'observation',
                        'forecast', 'forecastobservation',
                        'forecastaggregate',
                        'eventobservation', 'eventforecast',
                        'probabilisticforecastconstantvalue',
                        'probabilisticforecast', 'aggregate',
                        'aggregateforecast', 'aggregateprobforecast',
                        'aggregateobservation', 'report', 'quality_filter',
                        'timeofdayfilter', 'valuefilter', 'metricvalue',
                        'metricresult', 'validationresult',
                        'preprocessing_result', 'reportparameters',
                        'plotlyreportfigure', 'reportmessage',
                        'bokehreportfigure', 'constantcost',
                        'timeofdaycost', 'datetimecost', 'errorbandcost',
                        'cost'])
def pdid_params(request, many_sites, many_sites_text,
                single_observation, single_observation_text,
                single_site, single_forecast_text, single_forecast,
                single_event_observation_text, single_event_observation,
                single_event_forecast_text, single_event_forecast,
                prob_forecast_constant_value,
                prob_forecast_constant_value_text, prob_forecasts,
                prob_forecast_text, aggregate, aggregate_observations,
                aggregate_text, aggregate_forecast_text,
                aggregateforecast, aggregate_prob_forecast,
                aggregate_prob_forecast_text,
                agg_prob_forecast_constant_value,
                single_aggregate_observation,
                single_aggregate_observation_text, report_objects,
                report_dict, quality_filter, quality_filter_dict,
                timeofdayfilter, timeofdayfilter_dict, valuefilter,
                valuefilter_dict, metric_value_dict, metric_value,
                metric_result_dict, metric_result,
                validation_result_dict, validation_result,
                preprocessing_result_dict, preprocessing_result,
                plotly_report_figure_dict, plotly_report_figure,
                bokeh_report_figure_dict, bokeh_report_figure,
                report_message_dict, report_message,
                report_params_dict, report_params,
                constant_cost, timeofday_cost, datetime_cost,
                errorband_cost, banded_cost_params, cost_dicts):
    if request.param == 'site':
        return (many_sites[0], json.loads(many_sites_text)[0],
                datamodel.Site)
    elif request.param == 'fixed':
        return (many_sites[1].modeling_parameters,
                json.loads(many_sites_text)[1]['modeling_parameters'],
                datamodel.FixedTiltModelingParameters)
    elif request.param == 'single':
        return (many_sites[2].modeling_parameters,
                json.loads(many_sites_text)[2]['modeling_parameters'],
                datamodel.SingleAxisModelingParameters)
    elif request.param == 'observation':
        obs_dict = json.loads(single_observation_text)
        obs_dict['site'] = single_site
        return (single_observation, obs_dict,
                datamodel.Observation)
    elif request.param == 'forecast':
        fx_dict = json.loads(single_forecast_text)
        fx_dict['site'] = single_site
        return (single_forecast, fx_dict, datamodel.Forecast)
    elif request.param == 'eventobservation':
        obs_dict = json.loads(single_event_observation_text)
        obs_dict['site'] = single_site
        return (single_event_observation, obs_dict, datamodel.Observation)
    elif request.param == 'eventforecast':
        fx_dict = json.loads(single_event_forecast_text)
        fx_dict['site'] = single_site
        return (single_event_forecast, fx_dict, datamodel.EventForecast)
    elif request.param == 'probabilisticforecastconstantvalue':
        fx_dict = json.loads(prob_forecast_constant_value_text)
        fx_dict['site'] = single_site
        return (prob_forecast_constant_value, fx_dict,
                datamodel.ProbabilisticForecastConstantValue)
    elif request.param == 'probabilisticforecast':
        fx_dict = json.loads(prob_forecast_text)
        fx_dict['site'] = single_site
        fx_dict['constant_values'] = (prob_forecast_constant_value, )
        return (prob_forecasts, fx_dict, datamodel.ProbabilisticForecast)
    elif request.param == 'forecastobservation':
        fx_dict = json.loads(single_forecast_text)
        fx_dict['site'] = single_site
        obs_dict = json.loads(single_observation_text)
        obs_dict['site'] = single_site
        fxobs_dict = {'forecast': fx_dict, 'observation': obs_dict}
        fxobs = datamodel.ForecastObservation(
            single_forecast, single_observation)
        return (fxobs, fxobs_dict, datamodel.ForecastObservation)
    elif request.param == 'aggregate':
        agg_dict = json.loads(aggregate_text)
        agg_dict['observations'] = aggregate_observations
        return (aggregate, agg_dict, datamodel.Aggregate)
    elif request.param == 'aggregateforecast':
        aggfx_dict = json.loads(aggregate_forecast_text)
        aggfx_dict['aggregate'] = aggregate.to_dict()
        return (aggregateforecast, aggfx_dict, datamodel.Forecast)
    elif request.param == 'aggregateprobforecast':
        fx_dict = json.loads(aggregate_prob_forecast_text)
        fx_dict['aggregate'] = aggregate.to_dict()
        fx_dict['constant_values'] = (agg_prob_forecast_constant_value, )
        return (aggregate_prob_forecast, fx_dict,
                datamodel.ProbabilisticForecast)
    elif request.param == 'forecastaggregate':
        aggfx_dict = json.loads(aggregate_forecast_text)
        aggfx_dict['aggregate'] = aggregate.to_dict()
        agg_dict = json.loads(aggregate_text)
        agg_dict['observations'] = aggregate_observations
        fxobs_dict = {'forecast': aggfx_dict, 'aggregate': agg_dict}
        fxobs = datamodel.ForecastAggregate(aggregateforecast, aggregate)
        return (fxobs, fxobs_dict, datamodel.ForecastAggregate)
    elif request.param == 'aggregateobservation':
        aggobs_dict = json.loads(single_aggregate_observation_text)
        return (single_aggregate_observation, aggobs_dict,
                datamodel.AggregateObservation)
    elif request.param == 'report':
        report, *_ = report_objects
        return (report, report_dict.copy(), datamodel.Report)
    elif request.param == 'quality_filter':
        return (quality_filter, quality_filter_dict,
                datamodel.QualityFlagFilter)
    elif request.param == 'timeofdayfilter':
        return (timeofdayfilter, timeofdayfilter_dict,
                datamodel.TimeOfDayFilter)
    elif request.param == 'valuefilter':
        return (valuefilter, valuefilter_dict,
                datamodel.ValueFilter)
    elif request.param == 'metricvalue':
        return (metric_value, metric_value_dict, datamodel.MetricValue)
    elif request.param == 'metricresult':
        return (metric_result, metric_result_dict, datamodel.MetricResult)
    elif request.param == 'validationresult':
        return (validation_result, validation_result_dict,
                datamodel.ValidationResult)
    elif request.param == 'preprocessing_result':
        return (preprocessing_result, preprocessing_result_dict,
                datamodel.PreprocessingResult)
    elif request.param == 'reportparameters':
        return (report_params, report_params_dict,
                datamodel.ReportParameters)
    elif request.param == 'plotlyreportfigure':
        return (plotly_report_figure, plotly_report_figure_dict,
                datamodel.PlotlyReportFigure)
    elif request.param == 'bokehreportfigure':
        return (bokeh_report_figure, bokeh_report_figure_dict,
                datamodel.BokehReportFigure)
    elif request.param == 'reportmessage':
        return (report_message, report_message_dict, datamodel.ReportMessage)
    elif request.param == 'constantcost':
        return (constant_cost, cost_dicts['constant'], datamodel.ConstantCost)
    elif request.param == 'timeofdaycost':
        return (timeofday_cost, cost_dicts['timeofday'],
                datamodel.TimeOfDayCost)
    elif request.param == 'datetimecost':
        return (datetime_cost, cost_dicts['datetime'], datamodel.DatetimeCost)
    elif request.param == 'errorbandcost':
        return (errorband_cost, cost_dicts['errorband'],
                datamodel.ErrorBandCost)
    elif request.param == 'cost':
        return (banded_cost_params, cost_dicts['fullcost'], datamodel.Cost)


@pytest.mark.parametrize('extra', [
    {},
    {'extra': 'thing'},
])
def test_from_dict_into_datamodel(extra, pdid_params):
    expected, obj_dict, model = pdid_params
    obj_dict.update(extra)
    out = model.from_dict(obj_dict)
    assert out == expected


@pytest.fixture(params=[0, 1, 2])
def basefilter_params(
        request, valuefilter, valuefilter_dict, quality_filter,
        quality_filter_dict, timeofdayfilter, timeofdayfilter_dict):
    parameters = [
        (valuefilter, valuefilter_dict),
        (quality_filter, quality_filter_dict),
        (timeofdayfilter, timeofdayfilter_dict)]
    return parameters[request.param]


def test_base_filter_from_dict_into_datamodel(basefilter_params):
    expected, obj_dict = basefilter_params
    out = datamodel.BaseFilter.from_dict(obj_dict, raise_on_extra=True)
    assert out == expected


def test_filter_to_dict(basefilter_params):
    expected, obj_dict = basefilter_params
    out = expected.to_dict()
    assert out == obj_dict


def test_from_dict_into_datamodel_missing_field(pdid_params):
    _, obj_dict, model = pdid_params
    field_to_remove = None
    for field in fields(model):
        if field.default is MISSING and field.default_factory is MISSING:
            field_to_remove = field.name
            break
    if field_to_remove is not None:
        del obj_dict[field_to_remove]
        with pytest.raises(KeyError):
            model.from_dict(obj_dict)


def test_from_dict_into_datamodel_no_extra(pdid_params):
    expected, obj_dict, model = pdid_params
    obj_dict.pop('extra_parameters', '')
    out = model.from_dict(obj_dict)
    for field in fields(model):
        if field.name == 'extra_parameters':
            continue
        expected_value = getattr(expected, field.name)
        out_value = getattr(out, field.name)
        # normalization can be nan. isinstance ensures that isnan doesn't
        # try to coerce random strings or objects to a float and then error
        if isinstance(expected_value, float) and np.isnan(expected_value):
            assert np.isnan(out_value)
        else:
            assert out_value == expected_value


def test_from_dict_no_extra(pdid_params):
    expected, obj_dict, model = pdid_params
    names = [f.name for f in fields(model)]
    for key in list(obj_dict.keys()):
        if key not in names:
            del obj_dict[key]
    assert model.from_dict(obj_dict, raise_on_extra=True) == expected


def test_from_dict_extra_params_raise(pdid_params):
    _, obj_dict, model = pdid_params
    obj_dict['superextra'] = 'thing'
    with pytest.raises(KeyError):
        model.from_dict(obj_dict, raise_on_extra=True)


@pytest.mark.parametrize('site_num', [1, 2])
def test_from_dict_invalid_tracking_type(site_num, many_sites_text):
    model = datamodel.SolarPowerPlant
    obj_dict = json.loads(many_sites_text)[site_num]
    obj_dict['modeling_parameters']['tracking_type'] = 'invalid'
    with pytest.raises(ValueError):
        model.from_dict(obj_dict)


@pytest.mark.parametrize('model', [datamodel.Observation, datamodel.Forecast])
def test_from_dict_invalid_timedelta(model, many_observations_text,
                                     many_forecasts_text):
    if model == datamodel.Observation:
        obj_dict = json.loads(many_observations_text)[0]
    else:
        obj_dict = json.loads(many_forecasts_text)[0]
    obj_dict['interval_length'] = 'blah'
    with pytest.raises(ValueError):
        model.from_dict(obj_dict)


def test_from_dict_invalid_time_format(many_forecasts_text):
    obj_dict = json.loads(many_forecasts_text)[0]
    obj_dict['issue_time_of_day'] = '0000'
    with pytest.raises(ValueError):
        datamodel.Forecast.from_dict(obj_dict)


def test_from_dict_invalid_constant_values(prob_forecast_text, single_site):
    fx_dict = json.loads(prob_forecast_text)
    fx_dict['site'] = single_site
    fx_dict['constant_values'] = ('not a tuple of cv', 1)
    with pytest.raises(TypeError):
        datamodel.ProbabilisticForecast.from_dict(fx_dict)


def test_from_dict_invalid_axis(prob_forecast_text, single_site,
                                prob_forecast_constant_value):
    fx_dict = json.loads(prob_forecast_text)
    fx_dict['site'] = single_site
    fx_dict['constant_values'] = (prob_forecast_constant_value,)
    fx_dict['axis'] = 'z'
    with pytest.raises(ValueError):
        datamodel.ProbabilisticForecast.from_dict(fx_dict)


def test_from_dict_inconsistent_axis(prob_forecast_text, single_site,
                                     prob_forecast_constant_value_text,
                                     prob_forecast_constant_value):
    cv_dict = json.loads(prob_forecast_constant_value_text)
    cv_dict['site'] = single_site
    fx_dict = json.loads(prob_forecast_text)
    fx_dict['site'] = single_site
    fx_dict['constant_values'] = (prob_forecast_constant_value, cv_dict)
    fx_dict['axis'] = 'x'
    # check multiple constant values
    datamodel.ProbabilisticForecast.from_dict(fx_dict)
    cv_dict['axis'] = 'y'
    with pytest.raises(ValueError):
        datamodel.ProbabilisticForecast.from_dict(fx_dict)


def test_dict_roundtrip(pdid_params):
    expected, _, model = pdid_params
    dict_ = expected.to_dict()
    out = model.from_dict(dict_)
    assert out == expected


def test_to_dict_recurse(single_forecast):
    @dataclass
    class Special(datamodel.BaseModel):
        fx: datamodel.Forecast

    spec = Special(single_forecast)
    d_ = spec.to_dict()
    assert isinstance(d_['fx']['run_length'], float)
    assert isinstance(d_['fx']['issue_time_of_day'], str)
    assert isinstance(d_['fx'], dict)
    assert isinstance(d_['fx']['site'], dict)


def test_replace(single_forecast):
    run_length = single_forecast.run_length
    new_fx = single_forecast.replace(run_length=pd.Timedelta('3d'))
    assert new_fx.run_length != run_length
    assert new_fx != single_forecast
    assert new_fx.name == single_forecast.name


def test_invalid_variable(single_site):
    with pytest.raises(ValueError):
        datamodel.Observation(
            name='test', variable='noway',
            interval_value_type='interval_mean',
            interval_length=pd.Timedelta('1min'),
            interval_label='beginning',
            site=single_site,
            uncertainty=0.1,
        )


@pytest.fixture(params=[0, 1, 2])
def _sites(many_sites_text, many_sites, request):
    models = [datamodel.Site, datamodel.SolarPowerPlant,
              datamodel.SolarPowerPlant]
    param = request.param
    site_dict_list = json.loads(many_sites_text)
    return site_dict_list[param], many_sites[param], models[param]


def test_process_site_dict(_sites):
    site_dict, expected, model = _sites
    out = model.from_dict(site_dict)
    assert out == expected


def test_process_nested_objects(single_observation_text_with_site_text,
                                single_site, single_observation):
    obs_dict = json.loads(single_observation_text_with_site_text)
    obs = datamodel.Observation.from_dict(obs_dict)
    assert obs == single_observation
    assert obs.site == single_site
    assert obs.site == single_observation.site


@pytest.mark.parametrize('key,val', [
    ('interval_length', pd.Timedelta('2h')),
    ('interval_value_type', 'interval_max'),
    ('variable', 'ghi')
])
def test_aggregate_invalid(single_observation, key, val):
    obsd = single_observation.to_dict()
    obsd[key] = val
    obs = datamodel.Observation.from_dict(obsd)
    aggobs = datamodel.AggregateObservation(
        obs, pd.Timestamp.utcnow())
    with pytest.raises(ValueError):
        datamodel.Aggregate(
            'test', 'testd', 'dni', 'mean', pd.Timedelta('1h'),
            'ending', 'America/Denver',
            observations=(aggobs,)
        )


def test_forecast_invalid(single_forecast, single_site, aggregate):
    with pytest.raises(KeyError):
        single_forecast.replace(site=None, aggregate=None)
    with pytest.raises(KeyError):
        single_forecast.replace(site=single_site, aggregate=aggregate)


def test_probabilistic_forecast_float_constant_values(prob_forecasts):
    out = datamodel.ProbabilisticForecast(
        name=prob_forecasts.name,
        issue_time_of_day=prob_forecasts.issue_time_of_day,
        lead_time_to_start=prob_forecasts.lead_time_to_start,
        interval_length=prob_forecasts.interval_length,
        run_length=prob_forecasts.run_length,
        interval_label=prob_forecasts.interval_label,
        interval_value_type=prob_forecasts.interval_value_type,
        variable=prob_forecasts.variable,
        site=prob_forecasts.site,
        forecast_id=prob_forecasts.forecast_id,
        axis=prob_forecasts.axis,
        extra_parameters=prob_forecasts.extra_parameters,
        provider=prob_forecasts.provider,
        constant_values=tuple(cv.constant_value
                              for cv in prob_forecasts.constant_values),
    )
    object.__setattr__(prob_forecasts.constant_values[0], 'forecast_id', '')
    assert isinstance(out.constant_values[0],
                      datamodel.ProbabilisticForecastConstantValue)
    assert out == prob_forecasts


def test_probabilistic_forecast_float_constant_values_from_dict(
        prob_forecast_text, single_site, prob_forecast_constant_value,
        prob_forecasts):
    fx_dict = json.loads(prob_forecast_text)
    fx_dict['site'] = single_site
    fx_dict['constant_values'] = (
        prob_forecast_constant_value.constant_value, )
    out = datamodel.ProbabilisticForecast.from_dict(fx_dict)
    object.__setattr__(prob_forecasts.constant_values[0], 'forecast_id', '')
    assert out == prob_forecasts


def test_probabilistic_forecast_invalid_constant_value(prob_forecasts):
    with pytest.raises(TypeError):
        prob_forecasts.replace(constant_values=(1, 'a'))


def test__single_field_processing_union():
    @dataclass
    class Model(datamodel.BaseModel):
        myfield: Union[int, pd.Timestamp, None]

    out = datamodel._single_field_processing(Model, fields(Model)[0],
                                             '20190101')
    assert out == pd.Timestamp('20190101')

    out = datamodel._single_field_processing(Model, fields(Model)[0], None)
    assert out is None

    out = datamodel._single_field_processing(Model, fields(Model)[0], 10)
    assert out == 10

    with pytest.raises(TypeError):
        datamodel._single_field_processing(Model, fields(Model)[0], 'bad')


def test_forecast_from_union(single_forecast, single_forecast_text, site_text):
    @dataclass
    class Model(datamodel.BaseModel):
        myfield: Union[datamodel.Observation, datamodel.Forecast]

    fxdict = json.loads(single_forecast_text)
    fxdict['site'] = json.loads(site_text)
    out = Model.from_dict({'myfield': fxdict})
    assert out.myfield == single_forecast


def test___check_categories__():
    datamodel.__check_categories__(['total', 'weekday'])
    with pytest.raises(ValueError):
        datamodel.__check_categories__(['bad', 'very bad'])


@pytest.mark.parametrize('metrics', [
    (['rmse']),
    (list(datamodel.ALLOWED_DETERMINISTIC_METRICS.keys())),
    pytest.param(
        ["bss"],
        marks=pytest.mark.xfail(raises=ValueError, strict=True)
    ),
    pytest.param(
        list(datamodel.ALLOWED_PROBABILISTIC_METRICS.keys()),
        marks=pytest.mark.xfail(raises=ValueError, strict=True)
    ),
])
def test___check_metrics__(metrics, single_forecast):
    datamodel.__check_metrics__(single_forecast, metrics)


@pytest.mark.parametrize('metrics', [
    (["far"]),
    (list(datamodel.ALLOWED_EVENT_METRICS.keys())),
    pytest.param(
        ["rmse"],
        marks=pytest.mark.xfail(raises=ValueError, strict=True)
    ),
])
def test___check_metrics___event(single_event_forecast, metrics):
    datamodel.__check_metrics__(single_event_forecast, metrics)


@pytest.mark.parametrize('metrics', [
    (['crps']),
    (list(datamodel.ALLOWED_PROBABILISTIC_METRICS.keys())),
    pytest.param(
        ('rmse'),
        marks=pytest.mark.xfail(raises=ValueError, strict=True)
    ),
    pytest.param(
        list(datamodel.ALLOWED_DETERMINISTIC_METRICS.keys()),
        marks=pytest.mark.xfail(raises=ValueError, strict=True)
    ),
])
def test___check_metrics__probabilistic(metrics, prob_forecast_constant_value):
    datamodel.__check_metrics__(prob_forecast_constant_value, metrics)


@pytest.fixture
def objects_from_attrs(mocker):
    """Takes a list of lists with tuples of (attr_name, value)
       and creates a listed of Mock objects with their attributes
       set to those values
    """
    def fn(attr_list):
        things_with_attrs = []
        for attr_tuples in attr_list:
            with_attr = mocker.Mock()
            for (attr_name, attr_value) in attr_tuples:
                setattr(with_attr, attr_name, attr_value)
            things_with_attrs.append(with_attr)
        return things_with_attrs
    return fn


def test___check_units__(mocker, objects_from_attrs):
    things_with_units = objects_from_attrs(
        [[('units', u)] for u in ['W/M^2', 'W/M^2', 'W/M^2', 'W/M^2', 'W/M^2']]
    )
    datamodel.__check_units__(*things_with_units)


def test___check_units__no_args():
    datamodel.__check_units__()


def test___check_units___error(mocker, objects_from_attrs):
    things_with_units = objects_from_attrs(
        [[('units', u)] for u in ['W/M^2', 'different', 'W/M^2', 'W/M^2']]
    )
    with pytest.raises(ValueError):
        datamodel.__check_units__(*things_with_units)


@pytest.mark.parametrize('fx_int, fx_label, obs_int, obs_label', [
    (15, 'instant', 15, 'instant'), (1, 'ending', 1, 'ending'),
    (5, 'beginning', 5, 'beginning'), (15, 'beginning', 5, 'instant'),
])
def test___check_interval_compatibility__(
        fx_int, fx_label, obs_int, obs_label, objects_from_attrs):
    attrs = [(('interval_length', fx_int), ('interval_label', fx_label)),
             (('interval_length', obs_int), ('interval_label', obs_label))]
    forecast_and_observation = objects_from_attrs(attrs)
    datamodel.__check_interval_compatibility__(*forecast_and_observation)


def test___check_interval_compatibility__bad_labels(objects_from_attrs):
    attrs = [(('interval_length', 5), ('interval_label', 'instant')),
             (('interval_length', 5), ('interval_label', 'beginning'))]
    forecast_and_observation = objects_from_attrs(attrs)
    with pytest.raises(ValueError):
        datamodel.__check_interval_compatibility__(*forecast_and_observation)


def test___check_interval_compatibility__bad_length(objects_from_attrs):
    attrs = [(('interval_length', 5), ('interval_label', 'instant')),
             (('interval_length', 15), ('interval_label', 'instant'))]
    forecast_and_observation = objects_from_attrs(attrs)
    with pytest.raises(ValueError):
        datamodel.__check_interval_compatibility__(*forecast_and_observation)


def test_base_filter_from_dict_error():
    with pytest.raises(NotImplementedError):
        datamodel.BaseFilter.from_dict({'spicyness': 'very'})


def test_quality_flag_filter_post_init_error():
    with pytest.raises(ValueError):
        datamodel.QualityFlagFilter.from_dict(
            {'quality_flags': ['spicy_filter']})


def test_metric_result_post_init_error(metric_result_dict):
    metric_result_dict['aggregate_id'] = None
    metric_result_dict['observation_id'] = None
    with pytest.raises(ValueError):
        datamodel.MetricResult.from_dict(metric_result_dict)


# These 'sfp' tests check that behavior found in the _special_field_processing
# functions is maintained after removing the code.
@pytest.mark.parametrize('sitef', [
    lambda x: x,
    lambda x: x.to_dict(),
    pytest.param(lambda x: 'other', marks=pytest.mark.xfail(strict=True))
])
def test_forecast_sfp_site_dict(single_forecast_text, single_site, sitef):
    forecast_dict = json.loads(single_forecast_text)
    forecast_dict['site'] = sitef(single_site)
    forecast = datamodel.Forecast.from_dict(forecast_dict)
    assert isinstance(forecast.site, datamodel.Site)


@pytest.mark.parametrize('aggf', [
    lambda x: x,
    lambda x: x.to_dict(),
    pytest.param(lambda x: 'other', marks=pytest.mark.xfail(strict=True))
])
def test_forecast_sfp_aggregate_dict(single_forecast_text, aggregate, aggf):
    forecast_dict = json.loads(single_forecast_text)
    forecast_dict['aggregate'] = aggf(aggregate)
    forecast = datamodel.Forecast.from_dict(forecast_dict)
    assert isinstance(forecast.aggregate, datamodel.Aggregate)


@pytest.mark.parametrize('key,value,expected', [
    ('effective_until', None, None),
    ('effective_until', '2020-01-01T00:00Z',
     pd.Timestamp('2020-01-01T00:00Z')),
    ('observation_deleted_at', None, None),
    ('observation_deleted_at', '2020-01-01T00:00Z',
     pd.Timestamp('2020-01-01T00:00Z'))
])
def test_aggregate_observation_sfp(
        aggregate_observations, key, value, expected):
    aggobs = aggregate_observations[0]
    aggobs_dict = aggobs.to_dict()
    aggobs_dict[key] = value
    aggobs_from_dict = datamodel.AggregateObservation.from_dict(aggobs_dict)
    assert getattr(aggobs_from_dict, key) == expected


@pytest.mark.parametrize('key,value,expected', [
    ('effective_until', 'bad', None),
    ('observation_deleted_at', 'bad', None),
])
def test_aggregate_observation_sfp_invalid(
        aggregate_observations, key, value, expected):
    aggobs = aggregate_observations[0]
    aggobs_dict = aggobs.to_dict()
    aggobs_dict[key] = value
    with pytest.raises(TypeError):
        datamodel.AggregateObservation.from_dict(aggobs_dict)


@pytest.mark.parametrize('spec', [
    ('\"<script>alert("hello");</script>\"'),
    ('<script>alert("hello");</script>'),
    ('bad'),
    ('\"bad\"'),
    ('{"key": "value": "value"}'),
    ('{"key": "value"});</script><script>parseInt(1'),  # noqa
    ('{});</script><a onclick=alert("hello")></a><script>eval('),
])
def test_load_report_figure_invalid_spec(plotly_report_figure_dict, spec):
    json_dict = plotly_report_figure_dict.copy()
    json_dict.update(spec=spec)
    with pytest.raises(ValueError):
        datamodel.PlotlyReportFigure.from_dict(json_dict)


@pytest.mark.parametrize('spec', [
    ('\"<script>alert("hello");</script>\"'),
    ('<script>alert("hello");</script>'),
    ('bad'),
    ('\"bad\"'),
    ('{"key": "value": "value"}'),
    ('{"key": "value"});</script><script>parseInt(1'),  # noqa
    ('{});</script><a onclick=alert("hello")></a><script>eval('),
])
def test_update_report_figure_invalid_spec(plotly_report_figure_dict, spec):
    fig = datamodel.PlotlyReportFigure.from_dict(plotly_report_figure_dict)
    with pytest.raises(ValueError):
        fig.replace(spec=spec)


def test_invalid_empty_report_figure():
    with pytest.raises(ValueError):
        datamodel.ReportFigure()


@pytest.mark.parametrize('plotly_version,bokeh_version,script', [
    ('1.0.0', '1.0.0', '<script></script>'),
    (None, None, '<script></script>'),
    (None, '1.0.0', None),
])
def test_raw_report_plots_invalid_versions(
        plotly_report_figure, plotly_version, bokeh_version, script):
    with pytest.raises(KeyError):
        datamodel.RawReportPlots(
            figures=(plotly_report_figure,),
            plotly_version=plotly_version,
            bokeh_version=bokeh_version,
            script=script
        )


@pytest.mark.parametrize('plotly_version,bokeh_version,script', [
    ('1.0.0', None, None),
    (None, '1.0.0', '<script></script>')
])
def test_raw_report_plots(
        plotly_report_figure, plotly_version, bokeh_version, script):
    dm_obj = datamodel.RawReportPlots(
        figures=(plotly_report_figure,),
        plotly_version=plotly_version,
        bokeh_version=bokeh_version,
        script=script
    )
    assert dm_obj.figures == (plotly_report_figure,)
    assert dm_obj.plotly_version == plotly_version
    assert dm_obj.bokeh_version == bokeh_version
    assert dm_obj.script == script


def test_bokeh_report_figure(bokeh_report_figure_dict):
    dm_obj = datamodel.BokehReportFigure.from_dict(
        bokeh_report_figure_dict)
    for k, v in bokeh_report_figure_dict.items():
        assert getattr(dm_obj, k) == v


def test_plotly_report_figure(plotly_report_figure_dict):
    dm_obj = datamodel.PlotlyReportFigure.from_dict(
        plotly_report_figure_dict)
    for k, v in plotly_report_figure_dict.items():
        assert getattr(dm_obj, k) == v


def test_bokeh_report_figure_plotly_dict(plotly_report_figure_dict):
    with pytest.raises(KeyError):
        datamodel.BokehReportFigure.from_dict(plotly_report_figure_dict)


def test_plotly_report_figure_bokeh_dict(bokeh_report_figure_dict):
    with pytest.raises(KeyError):
        datamodel.PlotlyReportFigure.from_dict(bokeh_report_figure_dict)


def test_report_figure_from_dict_with_bokeh(bokeh_report_figure_dict):
    dm_obj = datamodel.ReportFigure.from_dict(bokeh_report_figure_dict)
    for key, value in bokeh_report_figure_dict.items():
        assert getattr(dm_obj, key) == value
    assert isinstance(dm_obj, datamodel.BokehReportFigure)


def test_report_figure_from_dict_with_plotly(plotly_report_figure_dict):
    dm_obj = datamodel.ReportFigure.from_dict(plotly_report_figure_dict)
    for key, value in plotly_report_figure_dict.items():
        assert getattr(dm_obj, key) == value
    assert isinstance(dm_obj, datamodel.PlotlyReportFigure)


@pytest.mark.parametrize('fig_dict', [
    {"no_class": "bad"},
    {"non-implemented": "d3"},
])
def test_report_figure_from_dict_invalid(fig_dict):
    with pytest.raises(NotImplementedError):
        datamodel.ReportFigure.from_dict(fig_dict)


def test_ForecastObservation_normalization_ac(single_forecast_ac_observation):
    assert single_forecast_ac_observation.normalization == 0.003


def test_ForecastObservation_normalization_dc(single_forecast_dc_observation):
    assert single_forecast_dc_observation.normalization == 0.0035


def test_ForecastObservation_normalization_wind_speed(
        single_forecast_wind_speed_observation):
    assert np.isnan(single_forecast_wind_speed_observation.normalization)


@pytest.mark.parametrize('uncertainty,expected', [
    (5, 5.),
    ('5', 5.),
    ('observation_uncertainty', 0.1),
    (None, None)
])
def test_ForecastObservation_uncertainty(
        single_forecast, single_observation, uncertainty, expected):
    fxobs = datamodel.ForecastObservation(
        single_forecast, single_observation, uncertainty=uncertainty)
    assert fxobs.uncertainty == expected


def test_ForecastObservation_uncertainty_invalid(
        single_forecast, single_observation):
    with pytest.raises(ValueError):
        datamodel.ForecastObservation(
            single_forecast, single_observation, uncertainty='nope')


@pytest.mark.parametrize('uncertainty,expected', [
    (5, 5.),
    ('5', 5.),
    (None, None)
])
def test_ForecastAggregate_uncertainty(
        aggregateforecast, aggregate, uncertainty, expected):
    fxobs = datamodel.ForecastAggregate(
        aggregateforecast, aggregate, uncertainty=uncertainty)
    assert fxobs.uncertainty == expected


def test_ForecastAggregate_uncertainty_invalid(
        aggregateforecast, aggregate):
    with pytest.raises(ValueError):
        datamodel.ForecastAggregate(
            aggregateforecast, aggregate, uncertainty='anystring')


def test_ForecastObservation_reffx(single_forecast_observation_reffx):
    assert isinstance(single_forecast_observation_reffx.reference_forecast,
                      datamodel.Forecast)


def test_ForecastAggregate_reffx(aggregateforecast, aggregate):
    fxobs = datamodel.ForecastAggregate(
        aggregateforecast, aggregate, reference_forecast=aggregateforecast)
    assert isinstance(fxobs.reference_forecast, datamodel.Forecast)


@pytest.mark.parametrize("interval_label,variable", [
    ("event", "event"),
    pytest.param("event", "ghi",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    pytest.param("beginning", "event",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
    pytest.param("beginning", "dni",
                 marks=pytest.mark.xfail(raises=ValueError, strict=True)),
])
def test_EventForecast_parameters(site_metadata, interval_label, variable):
    fx = datamodel.EventForecast(
        site=site_metadata,
        name="dummy fx",
        interval_length=pd.Timedelta("1h"),
        interval_value_type="instantaneous",
        issue_time_of_day='0500',
        lead_time_to_start=pd.Timedelta("1h"),
        run_length=pd.Timedelta("1h"),
        variable=variable,
        interval_label=interval_label,
    )

    assert fx.interval_label == "event"
    assert fx.variable == "event"


@pytest.mark.parametrize('model', [
    datamodel.Forecast,
    datamodel.EventForecast,
])
def test_event_forecast_from_dict(model, single_event_forecast):
    event_fx_dict = single_event_forecast.to_dict()
    assert isinstance(model.from_dict(event_fx_dict), datamodel.EventForecast)


def test_raw_report_event_forecast_loading(raw_report_dict_with_event):
    raw_report = datamodel.RawReport.from_dict(raw_report_dict_with_event)
    for fxobs in raw_report.processed_forecasts_observations:
        assert isinstance(fxobs.original.forecast, datamodel.EventForecast)


@pytest.mark.parametrize('model', [
    datamodel.Forecast,
    datamodel.ProbabilisticForecast,
])
def test_prob_forecast_from_dict(model, prob_forecasts):
    fx_dict = prob_forecasts.to_dict()
    assert isinstance(
        model.from_dict(fx_dict), datamodel.ProbabilisticForecast)


@pytest.mark.parametrize('model', [
    datamodel.Forecast,
    datamodel.ProbabilisticForecastConstantValue,
])
def test_prob_forecast_constant_value_from_dict(
        model, prob_forecast_constant_value):
    fx_dict = prob_forecast_constant_value.to_dict()
    assert isinstance(
        model.from_dict(fx_dict), datamodel.ProbabilisticForecastConstantValue)


def test_raw_report_prob_forecast_loading(raw_report_dict_with_prob):
    raw_report = datamodel.RawReport.from_dict(raw_report_dict_with_prob)
    assert isinstance(
        raw_report.processed_forecasts_observations[0].original.forecast,
        datamodel.ProbabilisticForecast)
    assert isinstance(
        raw_report.processed_forecasts_observations[1].original.forecast,
        datamodel.ProbabilisticForecastConstantValue)


@pytest.mark.parametrize('a,b', [
    ('constant', 'datetime'),
    ('constant', 'timeofday'),
    ('constant', 'errorband'),
    ('timeofday', 'constant'),
    ('timeofday', 'datetime'),
    ('timeofday', 'errorband'),
    ('datetime', 'constant'),
    ('datetime', 'timeofday'),
    ('datetime', 'errorband'),
    pytest.param('erorband', 'constant',
                 marks=pytest.mark.xfail(type=ValueError, strict=True)),
])
def test_invalid_costband(constant_cost, datetime_cost, timeofday_cost,
                          errorband_cost, a, b):
    dd = {'constant': constant_cost, 'timeofday': timeofday_cost,
          'datetime': datetime_cost, 'errorband': errorband_cost}
    with pytest.raises(TypeError):
        datamodel.CostBand(
            error_range=(0, 2),
            cost_function=a,
            cost_function_parameters=dd[b]
        )


@pytest.mark.parametrize('which', [
    'constant', 'timeofday', 'datetime', 'errorband',
    pytest.param('fullcost', marks=pytest.mark.xfail(
        type=ValueError, strict=True))
])
def test_cost_from_dict(cost_dicts, which):
    out = datamodel.Cost.from_dict({'name': 'test', 'type': which,
                                    'parameters': cost_dicts[which]})
    assert isinstance(out, datamodel.Cost)
    assert str(out.parameters).split('(')[0].lower() == which + 'cost'


@pytest.mark.parametrize('which', [
    'constant', 'timeofday', 'datetime', 'errorband',
    pytest.param('other', marks=pytest.mark.xfail(type=ValueError,
                                                  strict=True))
])
def test_cost_types(which, constant_cost, datetime_cost, timeofday_cost,
                    errorband_cost):
    dd = {'constant': constant_cost, 'timeofday': timeofday_cost,
          'datetime': datetime_cost, 'errorband': errorband_cost,
          'other': constant_cost}
    out = datamodel.Cost(
        name='test',
        type=which,
        parameters=dd[which]
    )
    assert isinstance(out, datamodel.Cost)
    assert str(out.parameters).split('(')[0].lower() == which + 'cost'


@pytest.mark.parametrize('which,cls,new', [
    ('constant', datamodel.ConstantCost, {'aggregation': 'min'}),
    ('timeofday', datamodel.TimeOfDayCost, {'aggregation': 'what'}),
    ('timeofday', datamodel.TimeOfDayCost, {'fill': 'no'}),
    ('timeofday', datamodel.TimeOfDayCost, {'times': tuple()}),
    ('timeofday', datamodel.TimeOfDayCost, {'cost': tuple()}),
    ('timeofday', datamodel.TimeOfDayCost, {'cost': tuple(), 'fill': 0}),
    ('datetime', datamodel.DatetimeCost, {'datetimes': tuple()}),
    ('datetime', datamodel.DatetimeCost, {'cost': tuple()}),
    ('datetime', datamodel.DatetimeCost, {'aggregation': tuple()}),
    ('datetime', datamodel.DatetimeCost, {'fill': 1})
])
def test_cost_validation(which, cls, new, cost_dicts):
    cd = cost_dicts[which]
    cd.update(new)
    with pytest.raises(ValueError):
        cls.from_dict(cd)


@pytest.mark.parametrize('param', [
    pytest.param(
        'wont be', marks=pytest.mark.xfail(strict=True, type=ValueError)),
    'example cost',
    None
])
def test_reportparameters_cost_consistency(report_params, param):
    nop = report_params.object_pairs[0].replace(cost=param)
    op = list(report_params.object_pairs) + [nop]
    datamodel.ReportParameters(
        name=report_params.name,
        start=report_params.start,
        end=report_params.end,
        object_pairs=op,
        metrics=('cost', 'mae'),
        categories=report_params.categories,
        filters=report_params.filters,
        costs=report_params.costs
    )


@pytest.fixture(params=['constant', 'timeofday', 'datetime', 'errorband'])
def cost_json(constant_cost, constant_cost_json, timeofday_cost,
              timeofday_cost_json, datetime_cost, datetime_cost_json,
              banded_cost_params, banded_cost_params_json, request):
    if request.param == 'constant':
        return constant_cost_json, datamodel.Cost(
            name='constantcost', type='constant', parameters=constant_cost)
    elif request.param == 'timeofday':
        return timeofday_cost_json, datamodel.Cost(
            name='timeofdaycost', type='timeofday', parameters=timeofday_cost)
    elif request.param == 'datetime':
        return datetime_cost_json, datamodel.Cost(
            name='datetimecost', type='datetime', parameters=datetime_cost)
    elif request.param == 'errorband':
        return banded_cost_params_json, banded_cost_params


def test_cost_from_json(cost_json):
    cjson, exp = cost_json
    out = datamodel.Cost.from_dict(json.loads(cjson))
    assert out == exp


def test_forecastobservation_null_uncertainty(
        single_forecast, many_observations):
    obs = many_observations[-1]
    assert obs.uncertainty is None
    assert datamodel.ForecastObservation.from_dict({
        'forecast': single_forecast.to_dict(), 'observation': obs.to_dict(),
        'uncertainty': 'observation_uncertainty'}).uncertainty is None


@pytest.mark.parametrize('il,ivt', (
    ('fail', 'fail'),
    ('instantaneous', 'interval_mean'),
    ('beginning', 'mean'),
    pytest.param('instant', 'instantaneous',
                 marks=pytest.mark.xfail(strict=True))
))
@pytest.mark.parametrize('obj', ('fx', 'obs', 'cdf'))
def test_interval_validation(il, ivt, obj, ac_power_forecast_metadata,
                             single_observation, prob_forecasts):
    if obj == 'fx':
        params = ac_power_forecast_metadata.to_dict()
        obj = datamodel.Forecast
    elif obj == 'obs':
        params = single_observation.to_dict()
        obj = datamodel.Observation
    else:
        params = prob_forecasts.to_dict()
        obj = datamodel.ProbabilisticForecast
    params['interval_label'] = il
    params['interval_value_type'] = ivt
    with pytest.raises(ValueError, match='must be one of'):
        obj(**params)


@pytest.mark.parametrize('il,at', (
    ('fail', 'mean'),
    ('beginning', 'notna'),
    ('instant', 'mean'),
    pytest.param('ending', 'sum',
                 marks=pytest.mark.xfail(strict=True))
))
def test_aggregate_type(il, at, aggregate, aggregate_observations):
    params = aggregate.to_dict()
    params['interval_label'] = il
    params['interval_length'] = pd.Timedelta('60m')
    params['aggregate_type'] = at
    params['observations'] = aggregate_observations
    with pytest.raises(ValueError, match='must be one of'):
        datamodel.Aggregate(**params)


def test_probabilistic_units(prob_forecasts):
    params = prob_forecasts.to_dict()
    assert 'constant_value_units' not in params
    params['variable'] = 'ghi'
    params['axis'] = 'x'
    params['constant_values'] = [0, 1]

    pfx = datamodel.ProbabilisticForecast(**params)
    assert pfx.units == '%'
    assert pfx.constant_value_units == 'W/m^2'

    params['axis'] = 'y'

    pfx = datamodel.ProbabilisticForecast(**params)
    assert pfx.units == 'W/m^2'
    assert pfx.constant_value_units == '%'


def test_probabilistic_single_units(prob_forecast_constant_value):
    params = prob_forecast_constant_value.to_dict()
    assert 'constant_value_units' not in params
    params['variable'] = 'ghi'
    params['axis'] = 'x'

    pfx = datamodel.ProbabilisticForecastConstantValue(**params)
    assert pfx.units == '%'
    assert pfx.constant_value_units == 'W/m^2'

    params['axis'] = 'y'

    pfx = datamodel.ProbabilisticForecastConstantValue(**params)
    assert pfx.units == 'W/m^2'
    assert pfx.constant_value_units == '%'


def test_probabilistic_units_data_object_matching(
        prob_forecasts, single_observation):
    params = prob_forecasts.to_dict()
    assert 'constant_value_units' not in params
    params['variable'] = 'ghi'
    params['axis'] = 'x'

    pfx = datamodel.ProbabilisticForecast.from_dict(params)
    datamodel.ForecastObservation(pfx, single_observation)
