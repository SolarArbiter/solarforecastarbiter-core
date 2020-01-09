from bokeh.models import ColumnDataSource

import numpy as np

import pytest


from solarforecastarbiter.reports import figures
from solarforecastarbiter import datamodel


@pytest.fixture
def report_with_raw(report_dict, raw_report):
    report_dict['raw_report'] = raw_report(True)
    report = datamodel.Report.from_dict(report_dict)
    return report


@pytest.fixture
def raw_report_pfxobs_values(raw_report):
    # Useful for testing the valid types for a ProcessedForecastObservation's
    # values fields which can be either pd.Series, str or None
    def replace_values(value):
        raw = raw_report(False)
        raw = raw.replace(
            processed_forecasts_observations=tuple(
                pfxobs.replace(
                    forecast_values=value,
                    observation_values=value,
                )
                for pfxobs in raw.processed_forecasts_observations
            )
        )
        return raw
    return replace_values


@pytest.fixture
def set_report_pfxobs_values(report_dict, raw_report_pfxobs_values):
    def set_pfxobs_values(value):
        report_dict['raw_report'] = raw_report_pfxobs_values(value)
        report = datamodel.Report.from_dict(report_dict)
        return report
    return set_pfxobs_values


def test_construct_metrics_cds(report_with_raw):
    report = report_with_raw
    metrics = report.raw_report.metrics
    cds = figures.construct_metrics_cds(metrics)
    names = cds.data['name']
    abbrev = cds.data['abbrev']
    categories = cds.data['category']
    metrics = cds.data['metric']
    values = cds.data['value']

    expected_length = (len(report.metrics) * len(report.categories) *
                       len(report.forecast_observations))
    assert all([len(v) == expected_length for k, v in cds.data.items()])

    original_names = [fxobs.forecast.name
                      for fxobs in report.forecast_observations]
    assert np.all(
        names == np.repeat(
            np.array(original_names),
            len(report.metrics) * len(report.categories))
    )
    assert np.all(names == abbrev)

    assert np.all(
        metrics == np.tile(
            np.repeat(np.array(report.metrics, dtype=object),
                      len(report.categories)),
            len(report.forecast_observations))
    )

    assert np.all(
        categories == np.tile(
            np.array(report.categories),
            len(report.metrics) * len(report.forecast_observations))
    )

    # this could maybe use value variance, but asserting the cds process
    # did not mangle values for now
    assert (values == 1).all()


def test_construct_metrics_cds_with_rename(report_with_raw):
    metrics = report_with_raw.raw_report.metrics
    cds = figures.construct_metrics_cds(metrics,
                                        rename=figures.abbreviate)
    original_names = [fxobs.forecast.name
                      for fxobs in report_with_raw.forecast_observations]
    abbreviated = list(map(figures.abbreviate, original_names))
    assert np.all(
        cds.data['abbrev'] == np.repeat(
            np.array(abbreviated, dtype=object),
            len(report_with_raw.metrics) * len(report_with_raw.categories))
    )


def test_construct_metric_cds_no_values():
    # Iterative metrics cds creation just build empty cds from an empty
    # dataframe if no MetricResults are found in the metrics tuple
    cds = figures.construct_metrics_cds(())
    assert cds.data['index'].size == 0


def test_construct_timeseries_cds(report_with_raw):
    report = report_with_raw
    raw_report = report.raw_report
    timeseries_cds, metadata_cds = figures.construct_timeseries_cds(report)

    ts_pair_index = timeseries_cds.data['pair_index']
    assert np.all(
        ts_pair_index == np.arange(len(report.forecast_observations)).repeat(
            [len(fxob.forecast_values)
             for fxob in raw_report.processed_forecasts_observations])
    )
    observation_values = timeseries_cds.data['observation_values']
    forecast_values = timeseries_cds.data['forecast_values']
    assert len(observation_values) == len(ts_pair_index)
    assert len(forecast_values) == len(ts_pair_index)
    # just testing for non-mangling behavior here
    assert np.all(observation_values == 100)
    assert np.all(forecast_values == 100)

    assert 'pair_index' in metadata_cds.data
    assert 'observation_name' in metadata_cds.data
    assert 'forecast_name' in metadata_cds.data
    assert 'interval_label' in metadata_cds.data
    assert 'observation_hash' in metadata_cds.data
    assert 'forecast_hash' in metadata_cds.data
    assert 'observation_color' in metadata_cds.data


@pytest.mark.parametrize('value', ['someid', None])
def test_construct_timeseries_cds_no_data(
        set_report_pfxobs_values, value):
    report = set_report_pfxobs_values(value)
    with pytest.raises(ValueError):
        timeseires_cds, metadata_cds = figures.construct_timeseries_cds(report)


@pytest.mark.parametrize('hash_key', [
    'observation_hash', 'forecast_hash'])
def test_extract_metadata_from_cds(report_with_raw, hash_key):
    timeseries_cds, metadata_cds = figures.construct_timeseries_cds(
        report_with_raw)
    for the_hash in metadata_cds.data[hash_key]:
        metadata = figures._extract_metadata_from_cds(
            metadata_cds, the_hash, hash_key)
        assert 'pair_index' in metadata
        assert 'observation_name' in metadata
        assert 'interval_label' in metadata
        assert 'observation_color' in metadata


@pytest.fixture
def fxobs_name_mock(mocker):
    def fn(obs_name, fx_name, agg=False):
        if agg:
            obspec = datamodel.Aggregate
        else:
            obspec = datamodel.Observation
        fxobs = mocker.Mock()
        obs = mocker.Mock(spec=obspec)
        obs.name = obs_name
        fx = mocker.Mock()
        fx.name = fx_name
        fxobs.forecast = fx
        fxobs.data_object = obs
        return fxobs
    return fn


def test_obs_name_same(fxobs_name_mock):
    name = 'ghi 1 hr'
    fxobs = fxobs_name_mock(name, name)
    fxagg = fxobs_name_mock(name, name, True)
    new_obsname = figures._obs_name(fxobs)
    new_aggname = figures._obs_name(fxagg)
    assert new_obsname == f'{name} Observation'
    assert new_aggname == f'{name} Aggregate'


def test_obs_name_same_diff(fxobs_name_mock):
    name = 'ghi 1 hr'
    fx_name = 'ghi 1 hr fx'
    fxobs = fxobs_name_mock(name, fx_name)
    fxagg = fxobs_name_mock(name, fx_name, True)
    new_obsname = figures._obs_name(fxobs)
    new_aggname = figures._obs_name(fxagg)
    assert new_obsname == name
    assert new_aggname == name


def test_fx_name_same(fxobs_name_mock):
    fxobs = fxobs_name_mock('same', 'same')
    new_fx_name = figures._fx_name(fxobs)
    assert new_fx_name == 'same Forecast'


def test_fx_name_diff(fxobs_name_mock):
    fxobs = fxobs_name_mock('same', 'diff')
    new_fx_name = figures._fx_name(fxobs)
    assert new_fx_name == 'diff'


@pytest.mark.parametrize('array,index,expected', [
    ([1, 2, 3, 4], 2, [False, True, False, False]),
    ([1, 1, 3, 4], 2, [False, False, False, False]),
    ([1, 1, 1, 1], 1, [True, True, True, True]),
])
def test_boolean_filter_indices_by_pair(mocker, array, index, expected):
    cds = mocker.Mock()
    cds.data = {'pair_index': np.array(array)}
    expected = np.array(expected)
    result = figures._boolean_filter_indices_by_pair(cds, index)
    assert np.all(result == expected)


@pytest.mark.parametrize('fmin,fmax,omin,omax,expected', [
    (-5, 5, -3, 3, (-5, 5)),
    (1, 5, 0, 4, (0, 5)),
    (np.nan, np.nan, np.nan, np.nan, (-999, 999)),
])
def test_get_scatter_limits(fmin, fmax, omin, omax, expected):
    cds = ColumnDataSource({
        'observation_values': np.array([omin, omax]),
        'forecast_values': np.array([fmin, fmax]),
    })
    assert figures._get_scatter_limits(cds) == expected


@pytest.mark.parametrize('y_min,y_max,pad,expected', [
    (1.0, 10.0, 0.5, (0.0, 5.0)),
    (-5.0, 10.0, 1.5, (-7.5, 15.0)),
    (-5.0, -1.0, 1.0, (-5.0, 0.0)),
    (-100.0, 100.0, 1.03, (-103.0, 103.0)),
])
def test_start_end(y_min, y_max, pad, expected):
    result = figures.calc_y_start_end(y_min, y_max, pad)
    assert result == expected


@pytest.mark.parametrize('arg,expected', [
    ('a', 'a'),
    ('abcd', 'abc.'),
    ('ABCDEFGHIJKLMNOP', 'ABCDEFGHIJKLMNOP'),
    ('University of Arizona OASIS Day Ahead GFS ghi',
     'Uni. of Ari. OASIS Day Ahe. GFS ghi')
])
def test_abbreviate(arg, expected):
    out = figures.abbreviate(arg)
    assert out == expected


def test_timeseries(report_with_raw):
    timeseries_cds, metadata_cds = figures.construct_timeseries_cds(
        report_with_raw)
    fig = figures.timeseries(
        timeseries_cds,
        metadata_cds,
        report_with_raw.start,
        report_with_raw.end,
        report_with_raw.forecast_observations[0].forecast.units
    )
    assert fig is not None


def test_scatter(report_with_raw):
    timeseries_cds, metadata_cds = figures.construct_timeseries_cds(
        report_with_raw)
    fig = figures.scatter(
        timeseries_cds,
        metadata_cds,
        report_with_raw.forecast_observations[0].forecast.units,
    )
    assert fig is not None


def test_timeseries_plots(report_with_raw):
    script, div = figures.timeseries_plots(report_with_raw)
    assert script is not None
    assert div is not None


def test_raw_report_plots(report_with_raw):
    metrics = report_with_raw.raw_report.metrics
    plots = figures.raw_report_plots(report_with_raw, metrics)
    assert plots is not None
