import datetime as dt
import json
import re
import uuid


import numpy as np
import pandas as pd
import pytest
from requests.exceptions import HTTPError


from solarforecastarbiter.datamodel import (
    Site, Observation, Forecast, ProbabilisticForecast)
from solarforecastarbiter.io.reference_observations import common
from solarforecastarbiter.io.reference_observations.tests.conftest import (
    expected_site,
    site_dicts,
    site_string_dicts,
    site_objects)


@pytest.fixture
def log(mocker):
    log = mocker.patch('solarforecastarbiter.io.reference_observations.'
                       'common.logger')
    return log


site_test_observation = Observation.from_dict({
    'name': 'site ghi',
    'variable': 'ghi',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 1,
    'site': site_objects[0],
    'uncertainty': 0.,
    'extra_parameters': ('{"network": "DOE ARM", '
                         '"observation_interval_length": 1,'
                         '"network_api_id": "api_id", '
                         '"network_api_abbreviation": "siteghi"}')
})
invalid_params = {
    'name': 'site-invalid-jsonparams',
    'latitude': 3,
    'longitude': -3,
    'elevation': 6,
    'timezone': 'Etc/GMT+8',
    'extra_parameters': '{{ mon$stertruck',
}
no_params = {
    'name': 'site-invalid-jsonparams',
    'latitude': 3,
    'longitude': -3,
    'elevation': 6,
    'timezone': 'Etc/GMT+8',
}


def test_decode_extra_parameters():
    metadata = Site.from_dict(site_string_dicts[0])
    params = common.decode_extra_parameters(metadata)
    assert params['network'] == 'DOE ARM'
    assert params['observation_interval_length'] == 1


@pytest.mark.parametrize('site', [
    (invalid_params),
    (no_params),
])
def test_decode_extra_parameters_error(site):
    with pytest.raises(ValueError):
        common.decode_extra_parameters(Site.from_dict(site))


@pytest.mark.parametrize('site,expected', [
    (site_objects[0], 'site'),
    (site_objects[1], 'site2'),
    (site_objects[2], 'site3'),
    (site_objects[3], 'site4'),
])
def test_site_name_no_network(site, expected):
    assert common.site_name_no_network(site) == expected


@pytest.mark.parametrize('name,expected', [
    ('{n}_(a)_/m\\_.e@_[1]-', 'n a m e 1')
])
def test_clean_name(name, expected):
    assert common.clean_name(name) == expected


bad_network = site_dicts()[0]
bad_network['extra_parameters']['network'] = 'BBQ DISCO'
bad_network = expected_site(bad_network)


@pytest.mark.parametrize('networks,site,expected', [
    (['DOE ARM'], site_string_dicts[0], True),
    ('DOE ARM', site_string_dicts[0], True),
    (['DOE ARM', 'NREL MIDC'], site_string_dicts[1], False),
    (['NOAA SURFRAD', 'DOE ARM'], site_string_dicts[3], False),
    (['NOAA SURFRAD', 'DOE ARM'], bad_network, False),
    (['ARMS'], site_string_dicts[0], False),
])
def test_check_network(networks, site, expected):
    metadata = Site.from_dict(site)
    assert common.check_network(networks, metadata) == expected


@pytest.mark.parametrize('networks,expected', [
    (['DOE ARM'], site_objects[:1]),
    ('DOE ARM', site_objects[:1]),
    (['NOAA SURFRAD', 'NOAA SOLRAD'], site_objects[1:3])
])
def test_filter_by_network(networks, expected):
    assert common.filter_by_networks(site_objects, networks) == expected


@pytest.mark.parametrize('site,variable', [
    (site_objects[0], 'ghi'),
])
def test_create_observation(
        mock_api, site, variable, observation_objects_param):
    mock_api.list_observations.return_value = []
    common.create_observation(mock_api, site, variable)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(
        observation_objects_param[0])


def test_create_observation_exists(
        mock_api, site_objects_param, observation_objects_param):
    variable = 'ghi'
    site = site_objects_param[0]
    common.create_observation(mock_api, site, variable)
    mock_api.create_observation.assert_not_called()


long_site_name = Site.from_dict({
    'name': 'ARM site with just abouts sixty four characters in its name oops',
    'latitude': 1,
    'longitude': 1,
    'elevation': 5,
    'timezone': 'Etc/GMT+8',
    'extra_parameters': ('{"network": "DOE ARM", "network_api_abbreviation": '
                         '"site_abbrev", "network_api_id": "thing", '
                         '"observation_interval_length": 1}'),
})
observation_long_site_name = Observation.from_dict({
    'name': 'site_abbrev air_temperature',
    'variable': 'air_temperature',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 1,
    'site': long_site_name,
    'uncertainty': None,
    'extra_parameters': ('{"network": "DOE ARM", "network_api_abbreviation": '
                         '"site_abbrev", "network_api_id": "thing", '
                         '"observation_interval_length": 1}'),
})


@pytest.mark.parametrize('site,variable,expected', [
    (long_site_name, 'air_temperature', observation_long_site_name),
])
def test_create_observation_long_site(mock_api, site, variable, expected):
    common.create_observation(mock_api, site, variable)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)


observation_with_extra_params = Observation.from_dict({
    'name': 'site ghi',
    'variable': 'ghi',
    'interval_label': 'ending',
    'interval_value_type': 'interval_mean',
    'interval_length': 5,
    'site': site_objects[0],
    'uncertainty': None,
    'extra_parameters': '{"network": "", "observation_interval_length": 5}'
})
observation_params = {
    'network': '',
    'observation_interval_length': 5,
}


@pytest.mark.parametrize('site,variable,expected,extra_params', [
    (site_objects[0], 'ghi', observation_with_extra_params,
     observation_params),
])
def test_create_observation_extra_parameters(
        mock_api, site, variable, expected, extra_params):
    mock_api.list_observations.return_value = []
    common.create_observation(mock_api, site, variable, extra_params)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)


observation_dict = {
    'name': 'just observation',
    'variable': 'ghi',
    'interval_label': 'beginning',
    'interval_value_type': 'instantaneous',
    'interval_length': 1,
    'site': site_objects[0],
    'uncertainty': 2.,
    'extra_parameters': ('{"network": "DOE ARM", '
                         '"network_api_id": "B13", '
                         '"network_api_abbreviation": "sgp", '
                         '"observation_interval_length": 1, '
                         '"datastreams": {'
                         '"qcrad": "sgpqcradlong1E13.c1", '
                         '"met": {'
                         '"sgpmetE13.b1": "2019-01-01/2020-01-01", '
                         '"sgpmetE13.b2": "2020-01-01/2021-01-01"}}}')
}
test_kwarg_observation = Observation.from_dict(observation_dict)
obs_kwargs = {
    'interval_label': 'beginning',
    'name': 'just observation',
    'interval_value_type': 'instantaneous',
    'uncertainty': 2.,
}

observation_dict_resample_mean = observation_dict.copy()
observation_dict_resample_mean['extra_parameters'] = (
    '{"network": "DOE ARM", '
    '"network_api_id": "qcradlong1", '
    '"network_api_abbreviation": "abbrv", '
    '"observation_interval_length": 1, '
    '"resample_how": "mean"}')
test_observation_mean = Observation.from_dict(observation_dict_resample_mean)

observation_dict_resample_first = observation_dict.copy()
observation_dict_resample_first['extra_parameters'] = (
    '{"network": "DOE ARM", '
    '"network_api_id": "qcradlong1", '
    '"network_api_abbreviation": "abbrv", '
    '"observation_interval_length": 1, '
    '"resample_how": "first"}')
test_observation_first = Observation.from_dict(observation_dict_resample_first)

observation_dict_resample_fail = observation_dict.copy()
observation_dict_resample_fail['extra_parameters'] = (
    '{"network": "DOE ARM", '
    '"network_api_id": "qcradlong1", '
    '"network_api_abbreviation": "abbrv", '
    '"observation_interval_length": 1, '
    '"resample_how": "nopethepope"}')
test_observation_fail = Observation.from_dict(observation_dict_resample_fail)


@pytest.mark.parametrize('site,variable,expected,kwargs', [
    (site_objects[0], 'ghi', test_kwarg_observation, obs_kwargs),
])
def test_create_observation_with_kwargs(
        mock_api, site, variable, expected, kwargs):
    common.create_observation(mock_api, site, variable, **kwargs)
    mock_api.create_observation.assert_called()
    mock_api.create_observation.assert_called_with(expected)


@pytest.mark.parametrize('observation,resample_how', [
    # all test data has same freq as observation.interval_length,
    # so resample method should not matter
    (test_kwarg_observation, None),
    (test_observation_mean, 'mean'),
    (test_observation_first, 'first')
])
@pytest.mark.parametrize('inp,expected', [
    # nan kept
    (pd.DataFrame({'ghi': [0, 1, 4.0, None, 5, 6]}, dtype='float',
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6)),
     pd.DataFrame({'value': [0, 1, 4.0, None, 5, 6],
                   'quality_flag': [0, 0, 0, 1, 0, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
    # nan filled in center
    (pd.DataFrame({'ghi': [0, 1, 4.0, 5, 6]}, dtype='float',
                  index=pd.DatetimeIndex(
                      ['2019-10-04T1200Z', '2019-10-04T1201Z',
                       '2019-10-04T1202Z', '2019-10-04T1204Z',
                       '2019-10-04T1205Z'])),
     pd.DataFrame({'value': [0, 1, 4.0, None, 5, 6],
                   'quality_flag': [0, 0, 0, 1, 0, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
    # new freq
    (pd.DataFrame({'ghi': [0, 1]}, dtype='float',
                  index=pd.DatetimeIndex(
                      ['2019-10-04T1200Z', '2019-10-04T1205Z'])),
     pd.DataFrame({'value': [0, None, None, None, None, 1.0],
                   'quality_flag': [0, 1, 1, 1, 1, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),

    # nans not extended to start end
    (pd.DataFrame({'ghi': [0, 1]}, dtype='float',
                  index=pd.DatetimeIndex(
                      ['2019-10-04T1201Z', '2019-10-04T1202Z'])),
     pd.DataFrame({'value': [0.0, 1.0],
                   'quality_flag': [0, 0]},
                  index=pd.DatetimeIndex(
                      ['2019-10-04T1201Z', '2019-10-04T1202Z']))),
])
def test_prepare_data_to_post(inp, expected, observation, resample_how):
    start = pd.Timestamp('2019-10-04T1200Z')
    end = pd.Timestamp('2019-10-04T1205Z')
    variable = 'ghi'
    out = common._prepare_data_to_post(inp, variable, observation, start, end,
                                       resample_how=resample_how)
    pd.testing.assert_frame_equal(out, expected)


@pytest.mark.parametrize('inp,expected', [
    (pd.DataFrame({'ghi': [0, 1, 4.0, None, 5, 6] * 2}, dtype='float',
                  index=pd.date_range('2019-10-04T1200Z', freq='30s',
                                      periods=12)),
     pd.DataFrame({'value': [0.5, 4.0, 5.5, 0.5, 4.0, 5.5],
                   'quality_flag': [0, 0, 0, 0, 0, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
    (pd.DataFrame({'ghi': [0, 1, None, None, 5, 6] * 2}, dtype='float',
                  index=pd.date_range('2019-10-04T1200Z', freq='30s',
                                      periods=12)),
     pd.DataFrame({'value': [0.5, None, 5.5, 0.5, None, 5.5],
                   'quality_flag': [0, 1, 0, 0, 1, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
])
def test_prepare_data_to_post_mean(inp, expected):
    start = pd.Timestamp('2019-10-04T1200Z')
    end = pd.Timestamp('2019-10-04T1205Z')
    variable = 'ghi'
    out = common._prepare_data_to_post(inp, variable, test_observation_mean,
                                       start, end, resample_how='mean')
    pd.testing.assert_frame_equal(out, expected)


@pytest.mark.parametrize('inp,expected', [
    (pd.DataFrame({'ghi': [0, 1, None, 4.0, 5, 6] * 2}, dtype='float',
                  index=pd.date_range('2019-10-04T1200Z', freq='30s',
                                      periods=12)),
     pd.DataFrame({'value': [0.0, 4.0, 5.0, 0.0, 4.0, 5.0],
                   'quality_flag': [0, 0, 0, 0, 0, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
    (pd.DataFrame({'ghi': [0, 1, None, None, 5, 6] * 2}, dtype='float',
                  index=pd.date_range('2019-10-04T1200Z', freq='30s',
                                      periods=12)),
     pd.DataFrame({'value': [0.0, None, 5.0, 0.0, None, 5.0],
                   'quality_flag': [0, 1, 0, 0, 1, 0]},
                  index=pd.date_range('2019-10-04T1200Z', freq='1min',
                                      periods=6))),
])
def test_prepare_data_to_post_first(inp, expected):
    start = pd.Timestamp('2019-10-04T1200Z')
    end = pd.Timestamp('2019-10-04T1205Z')
    variable = 'ghi'
    out = common._prepare_data_to_post(inp, variable, test_observation_first,
                                       start, end, resample_how='first')
    pd.testing.assert_frame_equal(out, expected)


def test_prepare_data_to_post_offset():
    # offset data is kept as is. If this is a problem we'll need to
    # probably resample the reference data to the appropriate index
    start = pd.Timestamp('2019-10-04T1200Z')
    end = pd.Timestamp('2019-10-04T1215Z')
    variable = 'ghi'
    inp = pd.DataFrame({'ghi': [0, 1, 4.0]},
                       index=pd.date_range('2019-10-04T1201Z', freq='5min',
                                           periods=3))
    expected = pd.DataFrame(
        {'value': [0, 1, 4.0], 'quality_flag': [0, 0, 0]},
        index=pd.date_range('2019-10-04T1201Z', freq='5min',
                            periods=3))
    out = common._prepare_data_to_post(
        inp, variable, observation_with_extra_params, start, end)
    pd.testing.assert_frame_equal(out, expected)


def test_prepare_data_to_post_empty():
    inp = pd.DataFrame(
        {'ghi': [0.0]}, index=pd.DatetimeIndex(['2019-01-01 00:00Z']))
    start = pd.Timestamp('2019-10-04T1200Z')
    end = pd.Timestamp('2019-10-04T1205Z')
    variable = 'ghi'
    out = common._prepare_data_to_post(inp, variable, test_kwarg_observation,
                                       start, end)
    assert out.empty


def test_prepare_data_to_post_no_var():
    start = pd.Timestamp('2109-10-04T1200Z')
    end = pd.Timestamp('2109-10-04T1300Z')
    data = pd.DataFrame({'notavar': [0, 1]})
    with pytest.raises(KeyError):
        common._prepare_data_to_post(data, 'GHI 7', test_kwarg_observation,
                                     start, end)


def test_post_observation_data_no_data(mock_api, log, start, end,):
    common.post_observation_data(
        mock_api, test_kwarg_observation,
        pd.DataFrame({'a': [1, 2, 3],
                      'b': ['a', 'b', 'c']}),
        start, end)
    log.error.assert_called()


def test_post_observation_data(mock_api, log, fake_ghi_data, start, end):
    common.post_observation_data(mock_api, site_test_observation,
                                 fake_ghi_data, start, end)

    args, _ = mock_api.post_observation_values.call_args
    # test observations never get assigned an id so the observation_id
    # argument should be an empty string
    assert args[0] == ''
    pd.testing.assert_frame_equal(
        args[1], fake_ghi_data.rename(columns={'ghi': 'value'}))


def test_post_observation_data_HTTPError(
        mock_api, log, fake_ghi_data, start, end):
    # mocking this error means that when the debugging logging call
    # fires, an AttributeError is thrown while looking for a response
    # but this also tests that we've followed the correct logic.
    mock_api.post_observation_values.side_effect = HTTPError
    with pytest.raises(AttributeError):
        common.post_observation_data(mock_api, site_test_observation,
                                     fake_ghi_data, start, end)
    log.error.assert_called()


def test_post_observation_data_all_nans(
        mock_api, log, fake_ghi_data, start, end):
    nan_data = fake_ghi_data.copy()
    nan_data['ghi'] = np.NaN
    ret = common.post_observation_data(mock_api, site_test_observation,
                                       nan_data, start, end)
    # post the nans
    log.warning.assert_not_called()
    assert ret is None


def test_post_observation_data_AttributeError(
        mock_api, log, fake_ghi_data, start, end):
    common.post_observation_data(mock_api, test_observation_fail,
                                 fake_ghi_data, start, end)
    log.error.assert_called()


@pytest.fixture()
def now(mocker):
    now = pd.Timestamp('20190108T11:32:00Z')
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.common._utcnow',
        return_value=now)
    return now


@pytest.fixture()
def site_obs():
    return [
        Observation.from_dict({
            'name': 'site ghi',
            'observation_id': str(uuid.uuid1()),
            'variable': 'ghi',
            'interval_label': 'ending',
            'interval_value_type': 'interval_mean',
            'interval_length': '5',
            'site': site,
            'uncertainty': 0.,
            'extra_parameters': site.extra_parameters
        }) for site in site_objects[:2]]


def test_get_last_site_timestamp(mock_api, now, site_obs):
    retvals = {site_obs[0].observation_id: (0, pd.Timestamp('20190105T0020Z')),
               site_obs[1].observation_id: (0, now)}
    mock_api.get_observation_time_range.side_effect = lambda x: retvals[x]
    ret = common.get_last_site_timestamp(mock_api, site_obs, now)
    assert ret == pd.Timestamp('20190105T0020Z')


def test_get_last_site_timestamp_small(mock_api, now, site_obs):
    retvals = {site_obs[0].observation_id: (0, pd.Timestamp('20181222T0020Z')),
               site_obs[1].observation_id: (0, pd.Timestamp('20190106T0020Z'))}
    mock_api.get_observation_time_range.side_effect = lambda x: retvals[x]
    ret = common.get_last_site_timestamp(mock_api, site_obs, now)
    assert ret == now - pd.Timedelta('7d')


def test_get_last_site_timestamp_none(mock_api, now, site_obs):
    mock_api.get_observation_time_range.return_value = (0, pd.NaT)
    ret = common.get_last_site_timestamp(mock_api, site_obs, now)
    assert ret == now - pd.Timedelta('7d')


def test_get_last_site_timestamp_some_nat(mock_api, now, site_obs):
    retvals = {site_obs[0].observation_id: (0, pd.Timestamp('20190105T0020Z')),
               site_obs[1].observation_id: (0, pd.NaT)}
    mock_api.get_observation_time_range.side_effect = lambda x: retvals[x]
    ret = common.get_last_site_timestamp(mock_api, site_obs, now)
    assert ret == pd.Timestamp('20190105T0020Z')


def test_get_last_site_timestamp_uptodate(mock_api, now, site_obs):
    mock_api.get_observation_time_range.return_value = (0, now)
    ret = common.get_last_site_timestamp(mock_api, site_obs, now)
    assert ret == now


def test_get_last_site_timestamp_empty(mock_api, now):
    ret = common.get_last_site_timestamp(mock_api, [], now)
    assert ret == now - pd.Timedelta('7d')


@pytest.mark.parametrize('site', site_objects)
def test_update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, fake_ghi_data):
    start = pd.Timestamp('20190101T1205Z')
    end = pd.Timestamp('20190101T1225Z')
    common.update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, start, end)
    args, _ = mock_api.post_observation_values.call_args
    assert args[0] == ''
    pd.testing.assert_frame_equal(
        args[1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[start:end].resample(
                args[1].index.freq).first())


def test_update_site_observations_no_start(
        mock_api, mock_fetch, observation_objects_param, fake_ghi_data,
        now):
    site = site_objects[0]
    start = None
    slimit = pd.Timestamp('20190101T1219Z')
    end = pd.Timestamp('20190108T1219Z')
    mock_api.get_observation_time_range.return_value = (0, slimit)
    common.update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, start, end)
    args, _ = mock_api.post_observation_values.call_args
    assert args[0] == ''
    pd.testing.assert_frame_equal(
        args[1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[slimit:end].resample(
                args[1].index.freq).first())


def test_update_site_observations_no_end(
        mock_api, mock_fetch, observation_objects_param, fake_ghi_data,
        now):
    site = site_objects[0]
    start = pd.Timestamp('20190101T1210Z')
    end = None
    common.update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, start, end)
    args, _ = mock_api.post_observation_values.call_args
    assert args[0] == ''
    pd.testing.assert_frame_equal(
        args[1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[start:].resample(
                args[1].index.freq).first())


def test_update_site_observations_uptodate(
        mock_api, mock_fetch, observation_objects_param, fake_ghi_data,
        now):
    site = site_objects[0]
    mock_api.get_observation_time_range.return_value = (0, now)
    common.update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, None, None)
    mock_api.post_observation_values.assert_not_called


def test_update_site_observations_no_data(
        mock_api, mocker, site_objects_param,
        observation_objects_param, log, start, end):
    fetch = mocker.MagicMock()
    fetch.return_value = pd.DataFrame()
    common.update_site_observations(
        mock_api, fetch, site_objects[1],
        observation_objects_param, start, end)
    mock_api.assert_not_called()


def test_update_site_observations_out_of_order(
        mock_api, site_objects_param, mocker,
        observation_objects_param, fake_ghi_data):
    start = pd.Timestamp('20190101T1200Z')
    end = pd.Timestamp('20190101T1230Z')
    fetch = mocker.MagicMock()
    fetch.return_value = fake_ghi_data.sample(frac=1)
    common.update_site_observations(
        mock_api, fetch, site_objects[1], observation_objects_param,
        start, end)
    args, _ = mock_api.post_observation_values.call_args
    assert args[0] == ''
    pd.testing.assert_frame_equal(
        args[1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[start:end].resample(
                args[1].index.freq).first())


def test_update_site_observations_gaps(
        mock_api, mock_fetch, observation_objects_param, fake_ghi_data,
        mocker):
    mocker.patch.object(
        mock_api, 'get_observation_value_gaps', return_value=[
            (pd.Timestamp('2019-01-01T12:10Z'),
             pd.Timestamp('2019-01-01T12:13Z')),
            (pd.Timestamp('2019-01-01T12:19Z'),
             pd.Timestamp('2019-01-01T12:20Z')),
        ])
    start = pd.Timestamp('20190101T1205Z')
    end = pd.Timestamp('20190101T1225Z')
    site = site_objects[0]
    common.update_site_observations(
        mock_api, mock_fetch, site, observation_objects_param, start, end,
        gaps_only=True)
    assert mock_api.post_observation_values.call_count == 2
    kargs = mock_api.post_observation_values.call_args_list
    assert kargs[0][0][0] == ''
    pd.testing.assert_frame_equal(
        kargs[0][0][1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[
                '2019-01-01T12:10Z':'2019-01-01T12:13Z'].resample(
                    kargs[0][0][1].index.freq).first())
    pd.testing.assert_frame_equal(
        kargs[1][0][1], fake_ghi_data.rename(
            columns={'ghi': 'value'})[
                '2019-01-01T12:19Z':'2019-01-01T12:20Z'].resample(
                    kargs[1][0][1].index.freq).first())


@pytest.fixture()
def template_fx(mock_api, mocker):
    mock_api.create_forecast = mocker.MagicMock(side_effect=lambda x: x)
    mock_api.create_probabilistic_forecast = mocker.MagicMock(
        side_effect=lambda x: x)
    site = site_objects[1].replace(latitude=32, longitude=-110)
    template = Forecast(
        name='Test Template',
        issue_time_of_day=dt.time(0),
        lead_time_to_start=pd.Timedelta('1d'),
        interval_length=pd.Timedelta('1h'),
        run_length=pd.Timedelta('24h'),
        interval_label='ending',
        interval_value_type='interval_mean',
        variable='ghi',
        site=Site('dummy', 0, 0, 0, 'MST'),
        extra_parameters=json.dumps(
            {'is_reference_forecast': True,
             'model': 'gfs_quarter_deg_hourly_to_hourly_mean'
             })
        )
    return mock_api, template, site


def test_create_one_forecast(template_fx):
    api, template, site = template_fx
    fx = common.create_one_forecast(api, site, template, 'ac_power')
    assert fx.name == 'site2 Test Template ac_power'
    assert fx.variable == 'ac_power'
    assert fx.site == site
    assert fx.issue_time_of_day == dt.time(8)
    ep = json.loads(fx.extra_parameters)
    assert ep['is_reference_forecast']
    assert ep['model'] == 'gfs_quarter_deg_hourly_to_hourly_mean'
    assert 'piggyback_on' not in ep


def test_create_one_forecast_invalid(template_fx):
    api, template, site = template_fx

    def fail(fx):
        raise ValueError('failed')
    fx = common.create_one_forecast(
        api, site, template, 'ac_power',
        creation_validation=fail)
    assert fx is None


@pytest.mark.parametrize('tz,expected', [
    ('Etc/GMT+8', dt.time(1)),
    ('MST', dt.time(0)),
    ('Etc/GMT+5', dt.time(4))
])
def test_create_one_forecast_issue_time(template_fx, tz, expected):
    api, template, site = template_fx
    template = template.replace(run_length=pd.Timedelta('6h'),
                                issue_time_of_day=dt.time(5),
                                lead_time_to_start=pd.Timedelta('1h'),
                                )
    site = site.replace(timezone=tz)
    fx = common.create_one_forecast(api, site, template, 'ac_power')
    assert fx.issue_time_of_day == expected


def test_create_one_forecast_long_name(template_fx):
    api, template, site = template_fx
    nn = 'a ' + ''.join(['n'] * 64)
    fx = common.create_one_forecast(api, site.replace(name=nn), template,
                                    'ac_power')
    assert fx.name == 'a Test Template ac_power'
    assert fx.variable == 'ac_power'
    assert fx.issue_time_of_day == dt.time(8)
    ep = json.loads(fx.extra_parameters)
    assert ep['is_reference_forecast']
    assert ep['model'] == 'gfs_quarter_deg_hourly_to_hourly_mean'
    assert 'piggyback_on' not in ep


@pytest.mark.parametrize('tmpl,exp', [
    ('Persistence 1hour ahead', 'The Site is really really really really Pers 1hour ahead GHI'),  # NOQA
    ('Persistence Fifteen-minute ahead', 'The Site is really really really really Pers 15 min ahead GHI'),  # NOQA
    ('n' * 46, 'The Site is ' + 'n' * 46 + ' GHI'),
    pytest.param(
        'This is too long to be a template forecast name and will raise a value error',   # NOQA
        '', marks=pytest.mark.xfail(strict=True, raises=ValueError))
])
def test_make_fx_name(tmpl, exp):
    out = common._make_fx_name(
        'The Site is really really really really long', tmpl, 'GHI')
    assert out == exp


def test_make_fx_name_long_site():
    with pytest.raises(ValueError):
        common._make_fx_name(
            'a' * 14,
            'r' * 49,
            '')


def test_create_one_forecast_piggy(template_fx):
    api, template, site = template_fx
    fx = common.create_one_forecast(api, site, template, 'ac_power',
                                    piggyback_on='other_fx')
    assert fx.name == 'site2 Test Template ac_power'
    assert fx.variable == 'ac_power'
    assert fx.site == site
    assert fx.issue_time_of_day == dt.time(8)
    ep = json.loads(fx.extra_parameters)
    assert ep['is_reference_forecast']
    assert ep['model'] == 'gfs_quarter_deg_hourly_to_hourly_mean'
    assert ep['piggyback_on'] == 'other_fx'


def test_create_one_forecast_existing(template_fx, mocker):
    api, template, site = template_fx
    newfx = template.replace(name='site2 Test Template ac_power', site=site)
    api.list_forecasts = mocker.MagicMock(return_value=[newfx])
    fx = common.create_one_forecast(api, site, template, 'ac_power',
                                    piggyback_on='other_fx')
    assert fx == newfx


@pytest.mark.parametrize('vars_,primary', [
    (('ac_power', 'dni'), 'ac_power'),
    (('ghi', 'dni'), 'ghi'),
    (('dni', 'wind_speed', 'relative_humidity'), False)
])
def test_create_nwp_forecasts(template_fx, mocker, vars_, primary):
    api, template, site = template_fx
    templates = [template.replace(name='one'), template.replace(name='two')]
    fxdict = template.to_dict()
    fxdict['constant_values'] = [0, 50, 100]
    fxdict['axis'] = 'y'
    templates += [ProbabilisticForecast.from_dict(fxdict)]

    fxs = common.create_nwp_forecasts(api, site, vars_, templates)
    assert len(fxs) == 6
    if primary:
        assert fxs[0].variable == primary
        assert fxs[2].variable == primary
    assert 'one' in fxs[0].name
    assert 'one' in fxs[1].name
    assert fxs[0].forecast_id in fxs[1].extra_parameters
    assert 'two' in fxs[2].name
    assert 'two' in fxs[3].name
    assert fxs[2].forecast_id in fxs[3].extra_parameters
    assert isinstance(fxs[-1], ProbabilisticForecast)


def test_create_nwp_forecasts_outside(template_fx, mocker, log):
    vars_ = ('ac_power', 'dni')
    api, template, site = template_fx
    site = site.replace(latitude=19, longitude=-159)
    templates = [template.replace(name='one'), template.replace(name='two')]
    with pytest.raises(ValueError):
        common.create_nwp_forecasts(api, site, vars_, templates)


def test_create_persistence_forecasts(template_fx, mocker):
    vars_ = ('ghi', 'dni', 'air_temperature', 'ac_power')
    api, template, site = template_fx
    templates = [
        template.replace(extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True}),
                         run_length=pd.Timedelta('6h')),
        template.replace(extra_parameters=json.dumps(
            {'is_reference_persistence_forecast': True}),
                         name='Longer template')
    ]
    obss = [site_test_observation.replace(site=site),
            site_test_observation.replace(site=site, variable='dni'),
            site_test_observation,
            site_test_observation.replace(
                site=site, variable='air_temperature'),
            site_test_observation.replace(
                site=site, variable='dhi'),
            ]
    api.list_observations = mocker.MagicMock(return_value=obss)
    fxs = common.create_persistence_forecasts(api, site, vars_, templates)
    assert len(fxs) == 6
    assert sorted([fx.variable for fx in fxs]) == [
        'air_temperature', 'air_temperature', 'dni', 'dni', 'ghi', 'ghi']
    assert 'Longer' in fxs[-1].name
    assert all(['observation_id' in fx.extra_parameters for fx in fxs])
    index_fxs = [re.search(r'index_persistence(["\s\:]*)true',
                           fx.extra_parameters, re.I) is not None
                 for fx in fxs if fx.variable != 'air_temperature'
                 and 'Longer' not in fx.name
                 ]
    assert len(index_fxs) == 2
    assert all(index_fxs)


def test_create_forecasts(template_fx, mocker):
    vars_ = ('ac_power', 'ghi')
    api, template, site = template_fx
    templates = [template, template.replace(extra_parameters=json.dumps(
        {'is_reference_persistence_forecast': True}))]
    api.list_observations = mocker.MagicMock(
        return_value=[site_test_observation.replace(site=site)])
    create_nwp = mocker.spy(common, 'create_nwp_forecasts')
    create_perst = mocker.spy(common, 'create_persistence_forecasts')
    fxs = common.create_forecasts(api, site, vars_, templates)
    assert len(fxs) == 3
    assert create_nwp.call_count == 1
    assert create_perst.call_count == 1
    assert create_nwp.call_args[0][-1] == templates[:1]
    assert create_perst.call_args[0][-1] == templates[-1:]


@pytest.mark.parametrize('params', [
    {'network_api_id': 2},
    {'network_api_id': '2'},
])
def test_apply_json_site_parameters_plant(test_json_site_file, params):
    new_site = common.apply_json_site_parameters(
        test_json_site_file,
        {'extra_parameters': params},
    )
    assert 'modeling_parameters' in new_site
    extra_params = new_site['extra_parameters']
    assert extra_params['network_api_abbreviation'] == 'SITE2'
    assert extra_params['attribution'] == ""
    assert extra_params['network'] == 'TEST'
    assert extra_params['observation_interval_length'] == 1.0


@pytest.mark.parametrize('params', [
    {'network_api_id': 'not_plant'},
])
def test_apply_json_site_parameters_no_params(test_json_site_file, params):
    new_site = common.apply_json_site_parameters(
        test_json_site_file,
        {'extra_parameters': params},
    )
    assert 'modeling_parameters' not in new_site
    assert list(new_site['extra_parameters'].keys()) == ['network_api_id']
