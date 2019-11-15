from dataclasses import fields, MISSING, dataclass
import json
from typing import Union


import pandas as pd
import pytest


from solarforecastarbiter import datamodel


@pytest.fixture(params=['site', 'fixed', 'single', 'observation',
                        'forecast', 'forecastobservation',
                        'probabilisticforecastconstantvalue',
                        'probabilisticforecast', 'aggregate',
                        'aggregateforecast', 'aggregateprobforecast',
                        'report', 'quality_filter',
                        'timeofdayfilter'])
def pdid_params(request, many_sites, many_sites_text, single_observation,
                single_observation_text, single_site,
                single_forecast_text, single_forecast,
                prob_forecast_constant_value,
                prob_forecast_constant_value_text,
                prob_forecasts, prob_forecast_text,
                aggregate, aggregate_observations,
                aggregate_text, aggregate_forecast_text,
                aggregateforecast, aggregate_prob_forecast,
                aggregate_prob_forecast_text,
                agg_prob_forecast_constant_value,
                report_objects, report_dict, quality_filter,
                quality_filter_dict, timeofdayfilter,
                timeofdayfilter_dict):
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
    elif request.param == 'report':
        report, *_ = report_objects
        return (report, report_dict.copy(), datamodel.Report)
    elif request.param == 'quality_filter':
        return (quality_filter, quality_filter_dict,
                datamodel.QualityFlagFilter)
    elif request.param == 'timeofdayfilter':
        return (timeofdayfilter, timeofdayfilter_dict,
                datamodel.TimeOfDayFilter)


@pytest.mark.parametrize('extra', [
    {},
    {'extra': 'thing'},
])
def test_from_dict_into_datamodel(extra, pdid_params):
    expected, obj_dict, model = pdid_params
    obj_dict.update(extra)
    out = model.from_dict(obj_dict)
    assert out == expected


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
        assert getattr(out, field.name) == getattr(expected, field.name)


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
            interval_value_type='mean',
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


def test_report_defaults(report_objects):
    report, *_ = report_objects
    report_defaults = datamodel.Report(
        name=report.name,
        start=report.start,
        end=report.end,
        forecast_observations=report.forecast_observations,
        report_id=report.report_id
    )
    assert isinstance(report_defaults.filters, tuple)


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
