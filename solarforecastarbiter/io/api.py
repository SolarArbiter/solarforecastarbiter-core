"""
Functions to connect to and process data from SolarForecastArbiter API
"""
import datetime as dt
import json
import logging


import requests
from urllib3 import Retry
import numpy as np
import pandas as pd


from solarforecastarbiter import datamodel
from solarforecastarbiter.utils import merge_ranges
from solarforecastarbiter.io.utils import (
    json_payload_to_observation_df,
    json_payload_to_forecast_series,
    observation_df_to_json_payload,
    forecast_object_to_json,
    adjust_timeseries_for_interval_label,
    serialize_timeseries,
    HiddenToken, ensure_timestamps,
    load_report_values)


BASE_URL = 'https://api.solarforecastarbiter.org'
logger = logging.getLogger(__name__)

# Limit used to limit the amount of retrieved with a single request. Used
# to break up large requests into smaller requests to avoid timeout.
GET_VALUES_LIMIT = '365D'


def request_cli_access_token(user, password, **kwargs):
    """Request an API access token from Auth0.

    Parameters
    ----------
    user : str
        Username
    password : str
        Password
    kwargs
        Passed to requests.post. Useful for handling SSL certificates,
        navigating proxies, or other network complications. See requests
        documentation for details.

    Returns
    -------
    access_token : str
    """
    req = requests.post(
        'https://solarforecastarbiter.auth0.com/oauth/token',
        data={'grant_type': 'password', 'username': user,
              'audience': BASE_URL,
              'password': password,
              'client_id': 'c16EJo48lbTCQEhqSztGGlmxxxmZ4zX7'},
        **kwargs)
    req.raise_for_status()
    return req.json()['access_token']


class APISession(requests.Session):
    """
    Subclass of requests.Session to handle requets to the SolarForecastArbiter
    API. The Session provides connection pooling, automatic retries for certain
    types of requets, default timeouts, and a default base url. Responses are
    converted into the appropriate class from datamodel.py or a pandas object.

    Parameters
    ----------
    access_token : string or HiddenToken
        The base64 encoded Bearer token to authenticate with the API
    default_timeout : float or tuple, optional
        A default timeout to add to all requests. If a tuple, the first element
        is the connection timeout and the second is the read timeout.
        Default is 10 seconds for connection and 60 seconds to read from the
        server.
    base_url : string
        URL to use as the base for endpoints to APISession

    Notes
    -----
    To pass the API calls through a proxy server, set either the HTTP_PROXY or
    HTTPS_PROXY environment variable. If necessary, you can also specify a SSL
    certificate using the REQUESTS_CA_BUNDLE environment variable. For example,
    on a Linux machine:

    >>> export HTTPS_PROXY=https://some_corporate_proxy.com:8080
    >>> export REQUESTS_CA_BUNDLE=/path/to/certificates/cert.crt
    >>> python script_that_calls_api.py

    For more information, see the "Advanced Usage" documentation for the
    requests package: https://requests.readthedocs.io/en/master/user/advanced/

    """

    def __init__(self, access_token, default_timeout=(10, 60),
                 base_url=None):
        super().__init__()
        if isinstance(access_token, HiddenToken):
            access_token = access_token.token
        self.headers = {'Authorization': f'Bearer {access_token}',
                        'Accept': 'application/json',
                        'Accept-Encoding': 'gzip,deflate'}
        self.default_timeout = default_timeout
        self.base_url = base_url or BASE_URL
        # set requests to automatically retry
        retries = Retry(total=10, connect=3, read=3, status=3,
                        status_forcelist=[408, 423, 444, 500, 501, 502, 503,
                                          504, 507, 508, 511, 599],
                        backoff_factor=0.5,
                        raise_on_status=False,
                        remove_headers_on_redirect=[])
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.mount(self.base_url, adapter)

    def request(self, method, url, *args, **kwargs):
        """
        Modify the default Session.request to add in the default timeout
        and make requests relative to the base_url. Users will likely
        use the standard get and post methods instead of calling this directly.

        Raises
        ------
        requests.exceptions.HTTPError
            When an error is encountered in when making the request to the API
        """
        if url.startswith('/'):
            url = f'{self.base_url}{url}'
        else:
            url = f'{self.base_url}/{url}'
        # set a defautl timeout so we never hang indefinitely
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout

        result = super().request(method, url, *args, **kwargs)
        if result.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f'{result.status_code} API Request Error: {result.reason} for '
                f'url: {result.url} and text: {result.text}',
                response=result)

        return result

    def _process_site_dict(self, site_dict):
        if (
                site_dict.get('modeling_parameters', {}).get(
                    'tracking_type', '') in ('fixed', 'single_axis')
        ):
            return datamodel.SolarPowerPlant.from_dict(site_dict)
        else:
            return datamodel.Site.from_dict(site_dict)

    def get_site(self, site_id):
        """
        Retrieve site metadata for site_id from the API and process
        into the proper model.

        Parameters
        ----------
        site_id : string
            UUID of the site to retrieve metadata for

        Returns
        -------
        datamodel.Site or datamodel.SolarPowerPlant
           Dataclass with all the metadata for the site depending on if
           the Site is a power plant with modeling parameters or not.
        """
        req = self.get(f'/sites/{site_id}')
        site_dict = req.json()
        return self._process_site_dict(site_dict)

    def list_sites(self):
        """
        List all the sites available to a user.

        Returns
        -------
        list of datamodel.Sites and datamodel.SolarPowerPlants
        """
        req = self.get('/sites/')
        return [self._process_site_dict(site_dict)
                for site_dict in req.json()]

    def list_sites_in_zone(self, zone):
        """
        List all the sites available to a user in the given climate zone.

        Parameters
        ----------
        zone : str

        Returns
        -------
        list of datamodel.Sites and datamodel.SolarPowerPlants
        """
        req = self.get(f'/sites/in/{zone}')
        return [self._process_site_dict(site_dict)
                for site_dict in req.json()]

    def search_climatezones(self, latitude, longitude):
        """
        Find all climate zones that the location is in.

        Parameters
        ----------
        latitude : float, degrees North
        longitude : float, degrees East of the Prime Meridian

        Returns
        -------
        list
            A list of the climate zones the location is in
        """
        req = self.get('/climatezones/search',
                       params={'latitude': latitude,
                               'longitude': longitude})
        return [r['name'] for r in req.json()]

    def create_site(self, site):
        """
        Create a new site in the API with the given Site model

        Parameters
        ----------
        site : datamodel.Site or datamodel.SolarPowerPlant
            Site to create in the API

        Returns
        -------
        datamodel.Site or datamodel.SolarPowerPlant
            With the appropriate parameters such as site_id set by the API
        """
        site_dict = site.to_dict()
        for k in ('site_id', 'provider', 'climate_zones'):
            site_dict.pop(k, None)
        site_json = json.dumps(site_dict)
        req = self.post('/sites/', data=site_json,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        return self.get_site(new_id)

    def get_observation(self, observation_id):
        """
        Get the metadata from the API for the a given observation_id
        in an Observation object.

        Parameters
        ----------
        observation_id : string
            UUID of the observation to retrieve

        Returns
        -------
        datamodel.Observation
        """
        req = self.get(f'/observations/{observation_id}/metadata')
        obs_dict = req.json()
        site = self.get_site(obs_dict['site_id'])
        obs_dict['site'] = site
        return datamodel.Observation.from_dict(obs_dict)

    def list_observations(self):
        """
        List the observations a user has access to.

        Returns
        -------
        list of datamodel.Observation
        """
        req = self.get('/observations/')
        obs_dicts = req.json()
        if isinstance(obs_dicts, dict):
            obs_dicts = [obs_dicts]
        if len(obs_dicts) == 0:
            return []
        sites = {site.site_id: site for site in self.list_sites()}
        out = []
        for obs_dict in obs_dicts:
            obs_dict['site'] = sites.get(obs_dict['site_id'])
            out.append(datamodel.Observation.from_dict(obs_dict))
        return out

    def create_observation(self, observation):
        """
        Create a new observation in the API with the given Observation model

        Parameters
        ----------
        observation : datamodel.Observation
            Observation to create in the API

        Returns
        -------
        datamodel.Observation
            With the appropriate parameters such as observation_id set by the
            API
        """
        obs_dict = observation.to_dict()
        obs_dict.pop('observation_id')
        obs_dict.pop('provider')
        site = obs_dict.pop('site')
        obs_dict['site_id'] = site['site_id']
        obs_json = json.dumps(obs_dict)
        req = self.post('/observations/', data=obs_json,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        return self.get_observation(new_id)

    def _process_fx(self, fx_dict, sites={}):
        if fx_dict['site_id'] is not None:
            if fx_dict['site_id'] in sites:
                fx_dict['site'] = sites[fx_dict['site_id']]
            else:
                fx_dict['site'] = self.get_site(fx_dict['site_id'])
        elif fx_dict['aggregate_id'] is not None:
            fx_dict['aggregate'] = self.get_aggregate(fx_dict['aggregate_id'])

        if fx_dict['variable'] == "event":
            return datamodel.EventForecast.from_dict(fx_dict)
        else:
            return datamodel.Forecast.from_dict(fx_dict)

    def get_forecast(self, forecast_id):
        """
        Get Forecast metadata from the API for the given forecast_id

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to get metadata for

        Returns
        -------
        datamodel.Forecast
        """
        req = self.get(f'/forecasts/single/{forecast_id}/metadata')
        fx_dict = req.json()
        return self._process_fx(fx_dict)

    def list_forecasts(self):
        """
        List all Forecasts a user has access to.

        Returns
        -------
        list of datamodel.Forecast
        """
        req = self.get('/forecasts/single/')
        fx_dicts = req.json()
        if isinstance(fx_dicts, dict):
            fx_dicts = [fx_dicts]
        if len(fx_dicts) == 0:
            return []
        sites = {site.site_id: site for site in self.list_sites()}
        out = []
        for fx_dict in fx_dicts:
            out.append(self._process_fx(fx_dict, sites=sites))
        return out

    def create_forecast(self, forecast):
        """
        Create a new forecast in the API with the given Forecast model

        Parameters
        ----------
        forecast : datamodel.Forecast
            Forecast to create in the API

        Returns
        -------
        datamodel.Forecast
            With the appropriate parameters such as forecast_id set by the API

        """
        fx_dict = forecast.to_dict()
        fx_dict.pop('forecast_id')
        fx_dict.pop('provider')
        site = fx_dict.pop('site')
        agg = fx_dict.pop('aggregate')
        if site is None and agg is not None:
            fx_dict['aggregate_id'] = agg['aggregate_id']
        else:
            fx_dict['site_id'] = site['site_id']
        fx_json = json.dumps(fx_dict)
        req = self.post('/forecasts/single/', data=fx_json,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        return self.get_forecast(new_id)

    def _process_prob_forecast(self, fx_dict, sites={}):
        if fx_dict['site_id'] is not None:
            if fx_dict['site_id'] in sites:
                fx_dict['site'] = sites[fx_dict['site_id']]
            else:
                fx_dict['site'] = self.get_site(fx_dict['site_id'])
        elif fx_dict['aggregate_id'] is not None:
            fx_dict['aggregate'] = self.get_aggregate(fx_dict['aggregate_id'])
        cvs = []
        for constant_value_dict in fx_dict['constant_values']:
            # the API just gets the groups attributes for the
            # single constant value forecasts, so avoid
            # those excess calls
            cv_dict = fx_dict.copy()
            cv_dict.update(constant_value_dict)
            cvs.append(
                datamodel.ProbabilisticForecastConstantValue.from_dict(
                    cv_dict))
        fx_dict['constant_values'] = cvs
        return datamodel.ProbabilisticForecast.from_dict(fx_dict)

    def list_probabilistic_forecasts(self):
        """
        List all ProbabilisticForecasts a user has access to.

        Returns
        -------
        list of datamodel.ProbabilisticForecast
        """
        req = self.get('/forecasts/cdf/')
        fx_dicts = req.json()
        if isinstance(fx_dicts, dict):
            fx_dicts = [fx_dicts]
        if len(fx_dicts) == 0:
            return []
        sites = {site.site_id: site for site in self.list_sites()}
        out = []
        for fx_dict in fx_dicts:
            out.append(self._process_prob_forecast(fx_dict, sites))
        return out

    def get_probabilistic_forecast(self, forecast_id):
        """
        Get ProbabilisticForecast metadata from the API for the given
        forecast_id.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to get metadata for

        Returns
        -------
        datamodel.ProbabilisticForecast
        """
        # add /metadata after
        # https://github.com/SolarArbiter/solarforecastarbiter-api/issues/158
        req = self.get(f'/forecasts/cdf/{forecast_id}')
        fx_dict = req.json()
        return self._process_prob_forecast(fx_dict)

    def get_probabilistic_forecast_constant_value(self, forecast_id,
                                                  site=None, aggregate=None):
        """
        Get ProbabilisticForecastConstantValue metadata from the API for
        the given forecast_id.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to get metadata for
        site : datamodel.Site or None
            If provided, the object will be attached to the returned
            value (faster). If None, object will be created from site
            metadata obtained from the database (slower).
        aggregate : datamodel.Aggregate or None
            If provided and the forecast is of an aggregate, the object
            will be attached to the return value.

        Returns
        -------
        datamodel.ProbabilisticForecastConstantValue

        Raises
        ------
        ValueError
            If provided site.site_id does not match database record of
            forecast object's linked site_id.
        """
        # add /metadata after
        # https://github.com/SolarArbiter/solarforecastarbiter-api/issues/158
        req = self.get(f'/forecasts/cdf/single/{forecast_id}')
        fx_dict = req.json()

        agg_id = fx_dict['aggregate_id']
        site_id = fx_dict['site_id']
        if site_id is not None:
            if site is None:
                site = self.get_site(site_id)
            elif site.site_id != site_id:
                raise ValueError('Supplied site.site_id does not match site_id'
                                 f'from database. site.site_id: {site.site_id}'
                                 f' database site_id: {site_id}')
            fx_dict['site'] = site
        elif agg_id is not None:
            if aggregate is None:
                aggregate = self.get_aggregate(agg_id)
            elif aggregate.aggregate_id != agg_id:
                raise ValueError(
                    'Supplied aggregate.aggregate_id does not match '
                    'aggregate from database. aggregate.aggregate_id: '
                    f'{aggregate.aggregate_id}'
                    f' database aggregate_id: {agg_id}')
            fx_dict['aggregate'] = aggregate
        return datamodel.ProbabilisticForecastConstantValue.from_dict(fx_dict)

    def create_probabilistic_forecast(self, forecast):
        """
        Create a new forecast in the API with the given
        ProbabilisticForecast model

        Parameters
        ----------
        forecast : datamodel.ProbabilisticForecast
            Probabilistic forecast to create in the API

        Returns
        -------
        datamodel.ProbabilisticForecast
            With the appropriate parameters such as forecast_id set by the API
        """
        fx_dict = forecast.to_dict()
        fx_dict.pop('forecast_id')
        fx_dict.pop('provider')
        site = fx_dict.pop('site')
        agg = fx_dict.pop('aggregate')
        if site is None and agg is not None:
            fx_dict['aggregate_id'] = agg['aggregate_id']
        else:
            fx_dict['site_id'] = site['site_id']

        # fx_dict['constant_values'] is tuple of dict representations of
        # all ProbabilisticForecastConstantValue objects in the
        # ProbabilisticForecast. We need to extract just the numeric
        # values from these dicts and put them into a list for the API.
        constant_values_fxs = fx_dict.pop('constant_values')
        constant_values = [fx['constant_value'] for fx in constant_values_fxs]
        fx_dict['constant_values'] = constant_values
        fx_json = json.dumps(fx_dict)
        req = self.post('/forecasts/cdf/', data=fx_json,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        return self.get_probabilistic_forecast(new_id)

    def get_observation_time_range(self, observation_id):
        """
        Get the minimum and maximum timestamps for observation values.

        Parameters
        ----------
        observation_id : string
            UUID of the observation object.

        Returns
        -------
        tuple of (pandas.Timestamp, pandas.Timestamp)
            The minimum and maximum timestamps for values of the observation.
            Values without an explicit timezone from the API are assumed to be
            UTC.
        """
        req = self.get(f'/observations/{observation_id}/values/timerange')
        data = req.json()
        mint = pd.Timestamp(data['min_timestamp'])
        if mint.tzinfo is None and pd.notna(mint):
            mint = mint.tz_localize('UTC')
        maxt = pd.Timestamp(data['max_timestamp'])
        if maxt.tzinfo is None and pd.notna(maxt):
            maxt = maxt.tz_localize('UTC')
        return mint, maxt

    def _process_gaps(self, url, start, end):
        req = self.get(url,
                       params={'start': start,
                               'end': end})
        gaps = req.json()
        out = []
        for g in gaps['gaps']:
            tstamp = pd.Timestamp(g['timestamp'])
            nextstamp = pd.Timestamp(g['next_timestamp'])
            # results should never be null, but skip anyway
            if pd.isna(tstamp) or pd.isna(nextstamp):
                continue  # pragma: no cover
            if tstamp.tzinfo is None:
                tstamp = tstamp.tz_localize('UTC')
            if nextstamp.tzinfo is None:
                nextstamp = nextstamp.tz_localize('UTC')
            out.append((tstamp, nextstamp))
        return out

    def _fixup_gaps(self, timerange, gaps, start, end):
        out = []
        if pd.isna(timerange[0]) or pd.isna(timerange[1]):
            return [(start, end)]
        else:
            if timerange[0] > start:
                if end < timerange[0]:
                    return [(start, end)]
                else:
                    out.append((start, timerange[0]))
            if timerange[1] < end:
                if start > timerange[1]:
                    return [(start, end)]
                else:
                    out.append((timerange[1], end))
        if len(gaps) != 0:
            if gaps[0][0] < start:
                gaps[0] = (start, gaps[0][1])
            if gaps[-1][1] > end:
                gaps[-1] = (gaps[-1][0], end)
            out.extend(gaps)
        return list(merge_ranges(out))

    @ensure_timestamps('start', 'end')
    def get_observation_value_gaps(self, observation_id, start, end):
        """Get any gaps in observation data from start to end.

        In addition to querying the /observations/{observation_id}/values/gaps
        endpoint, this function also queries the observation timerange to
        return all gaps from start to end.

        Parameters
        ----------
        observation_id : string
            UUID of the observation object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval

        Returns
        -------
        list of (pd.Timestamp, pd.Timestamp)
           Of (start, end) gaps in the observations from the last timestamp
           of a valid observation to the next valid observation timestamp.
           Interval label is not accounted for.
        """
        gaps = self._process_gaps(
            f'/observations/{observation_id}/values/gaps', start, end)
        trange = self.get_observation_time_range(observation_id)
        return self._fixup_gaps(trange, gaps, start, end)

    @ensure_timestamps('start', 'end')
    def get_observation_values_not_flagged(
            self, observation_id, start, end, flag, timezone='UTC'):
        """
        Get the dates where the observation series is NOT flagged with
        the given flag/bitmask.

        Parameters
        ----------
        observation_id : string
            UUID of the observation object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval
        flag : int
            Days that are not flagged with this flag are returned. This can
            be a compound flag/bitmask of the flags found in
            :py:mod:`solarforecastarbiter.validation.quality_mapping`,
            in which case days that do not have all flags present
            are returned.
        timezone : str, default "UTC"
            The timezone to localize the data before computing the date

        Returns
        -------
        dates : numpy.array of type datetime64[D]
        """
        req = self.get(f'/observations/{observation_id}/values/unflagged',
                       params={'start': start,
                               'end': end,
                               'timezone': timezone,
                               'flag': flag})
        data = req.json()
        dates = data['dates']
        return np.array([dt.date.fromisoformat(d) for d in dates],
                        dtype='datetime64[D]')

    @ensure_timestamps('start', 'end')
    def get_observation_values(
            self, observation_id, start, end, interval_label=None,
            request_limit=GET_VALUES_LIMIT):
        """
        Get observation values from start to end for observation_id from the
        API

        Parameters
        ----------
        observation_id : string
            UUID of the observation object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval
        interval_label : str or None
            If beginning, ending, adjust the data to return only data that is
            valid between start and end. If None or instant, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.DataFrame
            With a datetime index and (value, quality_flag) columns

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """
        out = self.chunk_value_requests(
            f'/observations/{observation_id}/values',
            start,
            end,
            parse_fn=json_payload_to_observation_df,
            request_limit=request_limit,
        )
        return adjust_timeseries_for_interval_label(
            out, interval_label, start, end)

    def get_forecast_time_range(self, forecast_id):
        """
        Get the miniumum and maximum timestamps for forecast values.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object.

        Returns
        -------
        tuple of (pandas.Timestamp, pandas.Timestamp)
            The minimum and maximum timestamps for values of the forecast.
            Values without an explicit timezone from the API are assumed to be
            UTC.
        """
        req = self.get(f'/forecasts/single/{forecast_id}/values/timerange')
        data = req.json()
        mint = pd.Timestamp(data['min_timestamp'])
        if mint.tzinfo is None and pd.notna(mint):
            mint = mint.tz_localize('UTC')
        maxt = pd.Timestamp(data['max_timestamp'])
        if maxt.tzinfo is None and pd.notna(maxt):
            maxt = maxt.tz_localize('UTC')
        return mint, maxt

    @ensure_timestamps('start', 'end')
    def get_forecast_value_gaps(self, forecast_id, start, end):
        """Get any gaps in forecast data from start to end.

        In addition to querying the
        /forecasts/single/{forecast_id}/values/gaps endpoint, this
        function also queries the forecast timerange to return all
        gaps from start to end.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval

        Returns
        -------
        list of (pd.Timestamp, pd.Timestamp)
           Of (start, end) gaps in the forecasts
        """
        gaps = self._process_gaps(
            f'/forecasts/single/{forecast_id}/values/gaps',
            start, end)
        trange = self.get_forecast_time_range(forecast_id)
        return self._fixup_gaps(trange, gaps, start, end)

    @ensure_timestamps('start', 'end')
    def get_forecast_values(self, forecast_id, start, end, interval_label=None,
                            request_limit=GET_VALUES_LIMIT):
        """
        Get forecast values from start to end for forecast_id

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object
        start : timelike object
            Start of the interval to retrieve values for
        end : timelike object
            End of the interval
        interval_label : str or None
            If beginning, ending, adjust the data to return only data that is
            valid between start and end. If None or instant, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.Series
           With the forecast values and a datetime index

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """
        out = self.chunk_value_requests(
            f'/forecasts/single/{forecast_id}/values',
            start,
            end,
            json_payload_to_forecast_series,
            request_limit=request_limit
        )
        return adjust_timeseries_for_interval_label(
            out, interval_label, start, end)

    def get_probabilistic_forecast_constant_value_time_range(
            self, forecast_id):
        """
        Get the miniumum and maximum timestamps for forecast values.

        Parameters
        ----------
        forecast_id : string
            UUID of the constant value forecast object.

        Returns
        -------
        tuple of (pandas.Timestamp, pandas.Timestamp)
            The minimum and maximum timestamps for values of the forecast.
            Values without an explicit timezone from the API are assumed to be
            UTC.
        """
        req = self.get(f'/forecasts/cdf/single/{forecast_id}/values/timerange')
        data = req.json()
        mint = pd.Timestamp(data['min_timestamp'])
        if mint.tzinfo is None and pd.notna(mint):
            mint = mint.tz_localize('UTC')
        maxt = pd.Timestamp(data['max_timestamp'])
        if maxt.tzinfo is None and pd.notna(maxt):
            maxt = maxt.tz_localize('UTC')
        return mint, maxt

    @ensure_timestamps('start', 'end')
    def get_probabilistic_forecast_constant_value_value_gaps(
            self, forecast_id, start, end):
        """Get any gaps in forecast data from start to end.

        In addition to querying the
        /forecasts/cdf/single/{forecast_id}/values/gaps endpoint, this
        function also queries the forecast timerange to return all
        gaps from start to end.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval

        Returns
        -------
        list of (pd.Timestamp, pd.Timestamp)
           Of (start, end) gaps in the forecasts
        """
        gaps = self._process_gaps(
            f'/forecasts/cdf/single/{forecast_id}/values/gaps',
            start, end)
        trange = self.get_probabilistic_forecast_constant_value_time_range(
            forecast_id)
        return self._fixup_gaps(trange, gaps, start, end)

    @ensure_timestamps('start', 'end')
    def get_probabilistic_forecast_constant_value_values(
            self, forecast_id, start, end, interval_label=None,
            request_limit=GET_VALUES_LIMIT):
        """
        Get forecast values from start to end for forecast_id

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object
        constant_value : string
            The variable value or percentile.
        start : timelike object
            Start of the interval to retrieve values for
        end : timelike object
            End of the interval
        interval_label : str or None
            If beginning, ending, adjust the data to return only data that is
            valid between start and end. If None or instant, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.Series
           With the forecast values and a datetime index

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """
        out = self.chunk_value_requests(
            f'/forecasts/cdf/single/{forecast_id}/values',
            start,
            end,
            json_payload_to_forecast_series,
            request_limit=request_limit,
        )
        return adjust_timeseries_for_interval_label(
            out, interval_label, start, end)

    @ensure_timestamps('start', 'end')
    def get_probabilistic_forecast_value_gaps(
            self, forecast_id, start, end):
        """Get any gaps in forecast data from start to end.

        In addition to querying the
        /forecasts/cdf/{forecast_id}/values/gaps endpoint, this
        function also queries the forecast timerange of the first
        constant value only to return all gaps from start to end.

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval

        Returns
        -------
        list of (pd.Timestamp, pd.Timestamp)
           Of (start, end) gaps in the forecasts
        """
        gaps = self._process_gaps(
            f'/forecasts/cdf/{forecast_id}/values/gaps',
            start, end)
        prob_fx = self.get_probabilistic_forecast(forecast_id)
        trange = self.get_probabilistic_forecast_constant_value_time_range(
            prob_fx.constant_values[0].forecast_id)
        return self._fixup_gaps(trange, gaps, start, end)

    @ensure_timestamps('start', 'end')
    def get_probabilistic_forecast_values(
            self, forecast_id, start, end, interval_label=None,
            request_limit=GET_VALUES_LIMIT):
        """
        Get all probabilistic forecast values for each from start to end for
        forecast_id

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast object
        start : timelike object
            Start of the interval to retrieve values for
        end : timelike object
            End of the interval
        interval_label : str or None
            If beginning, ending, adjust the data to return only data that is
            valid between start and end. If None or instant, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.DataFrame
           With the forecast values in each column with column names as the
           constant value and a datetime index

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """
        df_dict = {}
        prob_fx = self.get_probabilistic_forecast(forecast_id)
        for cv in prob_fx.constant_values:
            df_dict[str(cv.constant_value)] = \
                self.get_probabilistic_forecast_constant_value_values(
                    forecast_id=cv.forecast_id, start=start, end=end,
                    interval_label=interval_label, request_limit=request_limit)
        return pd.DataFrame(df_dict)

    def post_observation_values(self, observation_id, observation_df,
                                params=None):
        """
        Upload the given observation values to the appropriate observation_id
        of the API.

        Parameters
        ----------
        observation_id : string
            UUID of the observation to add values for
        observation_df : pandas.DataFrame
            Dataframe with a datetime index and the (required) value and
            quality_flag columns to upload to the API.
        params : dict, list, string, default None
            Parameters passed through POST request. Types are the same as
            Requests <https://2.python-requests.org/en/master/api/#requests.Request>
        """  # NOQA
        json_vals = observation_df_to_json_payload(observation_df)
        self.post(f'/observations/{observation_id}/values',
                  data=json_vals, params=params,
                  headers={'Content-Type': 'application/json'})

    def post_forecast_values(self, forecast_id, forecast_series):
        """
        Upload the given forecast values to the appropriate forecast_id of the
        API

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to upload values to
        forecast_obj : pandas.Series
            Pandas series with a datetime index that contains the values to
            upload to the API
        """
        json_vals = forecast_object_to_json(forecast_series)
        self.post(f'/forecasts/single/{forecast_id}/values',
                  data=json_vals,
                  headers={'Content-Type': 'application/json'})

    def post_probabilistic_forecast_constant_value_values(self, forecast_id,
                                                          forecast_series):
        """
        Upload the given forecast values to the appropriate forecast_id of the
        API

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to upload values to
        forecast_obj : pandas.Series
            Pandas series with a datetime index that contains the values to
            upload to the API
        """
        json_vals = forecast_object_to_json(forecast_series)
        self.post(f'/forecasts/cdf/single/{forecast_id}/values',
                  data=json_vals,
                  headers={'Content-Type': 'application/json'})

    def process_report_dict(self, rep_dict):
        """
        Load parameters from rep_dict into a Report object, getting forecasts
        and observations as necessary

        Parameters
        ----------
        rep_dict : dict
            Report dictionary as posted to the the API. See the API schema for
            details

        Returns
        -------
        datamodel.Report
        """
        rep_params = rep_dict['report_parameters'].copy()
        req_dict = {}
        for key in ('report_id', 'status', 'provider'):
            req_dict[key] = rep_dict.get(key, '')
        pairs = []
        for o in rep_params['object_pairs']:
            fx_type = o.get('forecast_type', 'forecast')
            fx_method = self._forecast_get_by_type(fx_type)
            fx = fx_method(o['forecast'])
            norm = o.get('normalization')
            unc = o.get('uncertainty')
            cost = o.get('cost')
            ref_fx = o.get('reference_forecast')
            if ref_fx is not None:
                ref_fx = fx_method(ref_fx)
            if 'observation' in o:
                obs = self.get_observation(o['observation'])
                pair = datamodel.ForecastObservation(
                    fx, obs, normalization=norm, uncertainty=unc,
                    reference_forecast=ref_fx, cost=cost)
            elif 'aggregate' in o:
                agg = self.get_aggregate(o['aggregate'])
                pair = datamodel.ForecastAggregate(
                    fx, agg, normalization=norm, uncertainty=unc,
                    reference_forecast=ref_fx, cost=cost)
            else:
                raise ValueError('must provide observation or aggregate in all'
                                 'object_pairs')
            pairs.append(pair)
        rep_params['object_pairs'] = tuple(pairs)
        req_dict['report_parameters'] = rep_params
        return datamodel.Report.from_dict(req_dict)

    def get_report(self, report_id):
        """
        Get the metadata, and possible raw report if it has processed,
        from the API for the given report_id in a Report object.

        Parameters
        ----------
        report_id : string
            UUID of the report to retrieve

        Returns
        -------
        datamodel.Report
        """
        req = self.get(f'/reports/{report_id}')
        resp = req.json()
        raw = resp.pop('raw_report', None)
        report = self.process_report_dict(resp)
        if raw is not None:
            raw_report = datamodel.RawReport.from_dict(raw)
            processed_fxobs = self.get_raw_report_processed_data(
                report_id, raw_report, resp['values'])
            report = report.replace(raw_report=raw_report.replace(
                processed_forecasts_observations=processed_fxobs))
        return report

    def list_reports(self):
        """
        List the reports a user has access to.  Does not load the raw
        report data, use :py:meth:`~.APISession.get_report`.

        Returns
        -------
        list of datamodel.Report

        """
        req = self.get('/reports')
        rep_dicts = req.json()
        if isinstance(rep_dicts, dict):
            rep_dicts = [rep_dicts]
        if len(rep_dicts) == 0:
            return []
        out = []
        for rep_dict in rep_dicts:
            out.append(self.process_report_dict(rep_dict))
        return out

    def create_report(self, report):
        """
        Post the report request to the API. A completed report should post
        the raw_report with :py:meth:`~.APISession.post_raw_report`.

        Parameters
        ----------
        report : datamodel.Report

        Returns
        -------
        datamodel.Report
           As returned by the API

        """
        report_params = report.report_parameters.to_dict()
        fxobs = report_params.pop('object_pairs')
        object_pairs = []
        for _fo in fxobs:
            d = {'forecast': _fo['forecast']['forecast_id']}
            if 'aggregate' in _fo:
                d['aggregate'] = _fo['aggregate']['aggregate_id']
            else:
                d['observation'] = _fo['observation']['observation_id']
            if _fo['reference_forecast'] is not None:
                d['reference_forecast'] = \
                    _fo['reference_forecast']['forecast_id']
            if (_fo['normalization'] is not None and
                    ~np.isnan(_fo['normalization'])):
                d['normalization'] = str(_fo['normalization'])
            if _fo['uncertainty'] is not None:
                d['uncertainty'] = str(_fo['uncertainty'])
            if _fo['cost'] is not None:
                d['cost'] = _fo['cost']
            object_pairs.append(d)
        report_params['object_pairs'] = object_pairs
        params = {'report_parameters': report_params}
        req = self.post('/reports/', json=params,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        return self.get_report(new_id)

    def post_raw_report_processed_data(self, report_id, raw_report):
        """
        Post the processed data that was used to make the report to the
        API.

        Parameters
        ----------
        report_id : str
            ID of the report to post values to
        raw_report : datamodel.RawReport
            The raw report object with processed_forecasts_observations

        Returns
        -------
        tuple
            of datamodel.ProcessedForecastObservation with `forecast_values`
            and `observations_values` replaced with report value IDs for later
            retrieval
        """
        posted_fxobs = []
        for fxobs in raw_report.processed_forecasts_observations:
            fx_data = {
                'object_id': fxobs.original.forecast.forecast_id,
                'processed_values': serialize_timeseries(
                    fxobs.forecast_values)}
            fx_post = self.post(
                f'/reports/{report_id}/values',
                json=fx_data, headers={'Content-Type': 'application/json'})
            if isinstance(fxobs.original, datamodel.ForecastObservation):
                obj_id = fxobs.original.observation.observation_id
            else:
                obj_id = fxobs.original.aggregate.aggregate_id
            obs_data = {
                'object_id': obj_id,
                'processed_values': serialize_timeseries(
                    fxobs.observation_values)}
            obs_post = self.post(
                f'/reports/{report_id}/values',
                json=obs_data, headers={'Content-Type': 'application/json'})
            if fxobs.original.reference_forecast is not None:
                ref_fx_data = {
                    'object_id': fxobs.original.reference_forecast.forecast_id,
                    'processed_values': serialize_timeseries(
                        fxobs.reference_forecast_values)}
                ref_fx_post = self.post(
                    f'/reports/{report_id}/values',
                    json=ref_fx_data,
                    headers={'Content-Type': 'application/json'})
                processed_ref_fx_id = ref_fx_post.text
            else:
                processed_ref_fx_id = None
            processed_fx_id = fx_post.text
            processed_obs_id = obs_post.text
            new_fxobs = fxobs.replace(
                forecast_values=processed_fx_id,
                observation_values=processed_obs_id,
                reference_forecast_values=processed_ref_fx_id)
            posted_fxobs.append(new_fxobs)
        return tuple(posted_fxobs)

    def get_raw_report_processed_data(self, report_id, raw_report,
                                      values=None):
        """
        Load the processed forecast/observation data into the
        datamodel.ProcessedForecastObservation objects of the raw_report.

        Parameters
        ----------
        report_id : str
            ID of the report that values will be loaded from
        raw_report : datamodel.RawReport
            The raw report with processed_forecasts_observations to
            be replaced
        values : list or None
            The report values dict as returned by the API. If None, fetch
            the values from the API for the given report_id

        Returns
        -------
        tuple
           Of datamodel.ProcessedForecastObservation with values loaded into
           `forecast_values` and `observation_values`
        """
        if values is None:
            val_req = self.get(f'/reports/{report_id}/values')
            values = val_req.json()
        return load_report_values(raw_report, values)

    def post_raw_report(self, report_id, raw_report, status='complete'):
        """
        Update the report with the raw report and metrics

        Parameters
        ----------
        report_id : str
           ID of the report to update
        raw_report : datamodel.RawReport
           The raw report object to add to the report
        status : str, default 'complete'
           The new status of the report
        """
        posted_fxobs = self.post_raw_report_processed_data(
            report_id, raw_report)
        raw_dict = raw_report.replace(
            processed_forecasts_observations=posted_fxobs).to_dict()
        self.post(f'/reports/{report_id}/raw',
                  json=raw_dict,
                  headers={'Content-Type': 'application/json'})
        self.update_report_status(report_id, status)

    def update_report_status(self, report_id, status):
        """
        Update the status of the report

        Parameters
        ----------
        report_id : str
           ID of the report to update
        status : str
           New status of the report
        """
        self.post(f'/reports/{report_id}/status/{status}')

    def get_aggregate(self, aggregate_id):
        """
        Get Aggregate metadata from the API for the given aggregate_id

        Parameters
        ----------
        aggregate_id : string
            UUID of the aggregate to get metadata for

        Returns
        -------
        datamodel.Aggregate
        """
        req = self.get(f'/aggregates/{aggregate_id}/metadata')
        agg_dict = req.json()
        for o in agg_dict['observations']:
            o['observation'] = self.get_observation(o['observation_id'])
        return datamodel.Aggregate.from_dict(agg_dict)

    def list_aggregates(self):
        """
        List all Aggregates a user has access to.

        Returns
        -------
        list of datamodel.Aggregate
        """
        req = self.get('/aggregates/')
        agg_dicts = req.json()
        if isinstance(agg_dicts, dict):
            agg_dicts = [agg_dicts]
        if len(agg_dicts) == 0:
            return []
        observations = {obs.observation_id: obs
                        for obs in self.list_observations()}
        out = []
        for agg_dict in agg_dicts:
            for o in agg_dict['observations']:
                o['observation'] = observations.get(o['observation_id'])
            out.append(datamodel.Aggregate.from_dict(agg_dict))
        return out

    def create_aggregate(self, aggregate):
        """
        Create a new aggregate in the API with the given Aggregate model

        Parameters
        ----------
        aggregate : datamodel.Aggregate
            Aggregate to create in the API

        Returns
        -------
        datamodel.Aggregate
            With the parameters aggregate_id and provider set by the API.
        """
        agg_dict = aggregate.to_dict()
        agg_dict.pop('aggregate_id')
        agg_dict.pop('provider')
        agg_dict.pop('interval_value_type')
        observations = []
        for obs in agg_dict.pop('observations'):
            if obs['effective_from'] is not None:
                observations.append(
                    {'observation_id': obs['observation']['observation_id'],
                     'effective_from': obs['effective_from']})
            if obs['effective_until'] is not None:
                observations.append(
                    {'observation_id': obs['observation']['observation_id'],
                     'effective_until': obs['effective_until']})

        agg_json = json.dumps(agg_dict)
        req = self.post('/aggregates/', data=agg_json,
                        headers={'Content-Type': 'application/json'})
        new_id = req.text
        obs_json = json.dumps({'observations': observations})
        self.post(f'/aggregates/{new_id}/metadata', data=obs_json,
                  headers={'Content-Type': 'application/json'})
        return self.get_aggregate(new_id)

    @ensure_timestamps('start', 'end')
    def get_aggregate_values(
            self, aggregate_id, start, end, interval_label=None,
            request_limit=GET_VALUES_LIMIT):
        """
        Get aggregate values from start to end for aggregate_id from the
        API

        Parameters
        ----------
        aggregate_id : string
            UUID of the aggregate object.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval
        interval_label : str or None
            If beginning or ending, return only data that is
            valid between start and end. If None, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.DataFrame
            With a datetime index and (value, quality_flag) columns

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """
        out = self.chunk_value_requests(
            f'/aggregates/{aggregate_id}/values',
            start,
            end,
            json_payload_to_observation_df,
            request_limit=request_limit,
        )
        return adjust_timeseries_for_interval_label(
            out, interval_label, start, end)

    @ensure_timestamps('start', 'end')
    def get_values(self, obj, start, end, interval_label=None,
                   request_limit=GET_VALUES_LIMIT):
        """
        Get time series values from start to end for object from the API

        Parameters
        ----------
        obj : datamodel.Observation, datamodel.Aggregate, datamodel.Forecast, datamodel.ProbabilisticForecastConstantValues
            Data model object for which to obtain time series data.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval
        interval_label : str or None
            If beginning or ending, return only data that is
            valid between start and end. If None, return any data
            between start and end inclusive of the endpoints.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            With a datetime index. If DataFrame, (value, quality_flag)
            columns

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        """  # noqa: E501
        # order avoids possible issues with inheritance
        if isinstance(obj, datamodel.ProbabilisticForecastConstantValue):
            f = self.get_probabilistic_forecast_constant_value_values
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.ProbabilisticForecast):
            f = self.get_probabilistic_forecast_values
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.Forecast):
            f = self.get_forecast_values
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.Aggregate):
            f = self.get_aggregate_values
            obj_id = obj.aggregate_id
        elif isinstance(obj, datamodel.Observation):
            f = self.get_observation_values
            obj_id = obj.observation_id
        return f(obj_id, start, end, interval_label=interval_label,
                 request_limit=request_limit)

    @ensure_timestamps('start', 'end')
    def get_value_gaps(self, obj, start, end):
        """
        Get gaps in the time series values from start to end for object from the API.

        Parameters
        ----------
        obj : datamodel.Observation, datamodel.Forecast, datamodel.ProbabilisticForecastConstantValues, datamodel.ProbabilisticForecast
            Data model object for which to obtain time series data.
        start : timelike object
            Start time in interval to retrieve values for
        end : timelike object
            End time of the interval


        Returns
        -------
        list of (pd.Timestamp, pd.Timestamp)
           Of (start, end) gaps in the values from the last timestamp
           of a valid value to the next valid timestamp.
           Interval label is not accounted for.

        Raises
        ------
        ValueError
            If start or end cannot be converted into a Pandas Timestamp
        TypeError
            If an invalid type of obj is supplied
        """  # noqa: E501
        # order avoids possible issues with inheritance
        if isinstance(obj, datamodel.ProbabilisticForecastConstantValue):
            f = self.get_probabilistic_forecast_constant_value_value_gaps
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.ProbabilisticForecast):
            f = self.get_probabilistic_forecast_value_gaps
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.Forecast):
            f = self.get_forecast_value_gaps
            obj_id = obj.forecast_id
        elif isinstance(obj, datamodel.Observation):
            f = self.get_observation_value_gaps
            obj_id = obj.observation_id
        else:
            raise TypeError(
                'Supplied object needs to be an Observation, Forecast, '
                'ProbabilisticForecast, or ProbabilisticForecastConstantValue')
        return f(obj_id, start, end)

    def get_user_info(self):
        """
        Get information about the current user from the API

        Returns
        -------
        dict
        """
        req = self.get('/users/current')
        return req.json()

    def _forecast_get_by_type(self, forecast_type):
        """Returns the appropriate function for requesting forecast metadata
        based on `forecast_type`.

        Parameters
        ----------
        forecast_type: str

        Returns
        -------
        function
        """
        if forecast_type == 'forecast':
            return self.get_forecast
        elif forecast_type == 'event_forecast':
            return self.get_forecast
        elif forecast_type == 'probabilistic_forecast':
            return self.get_probabilistic_forecast
        elif forecast_type == 'probabilistic_forecast_constant_value':
            return self.get_probabilistic_forecast_constant_value
        else:
            raise ValueError('Invalid forecast type.')

    def chunk_value_requests(
            self, api_path, start, end, parse_fn, params={},
            request_limit=GET_VALUES_LIMIT):
        """Breaks up a get requests for values into multiple requests limited
        by the request_limit argument.

        Parameters
        ----------
        api_path : str
        start : pandas.Timestamp
        end : pandas.Timestamp
        parse_fn : function
            A function used to parse json api response into a pandas Series
            or DataFrame.
        params : dict
            Any additional parameters to be passed with the get function.
        request_limit : string
            Timedelta string describing maximum request length. Defaults to 365
            days.

        Returns
        -------
        all_data: pandas.DataFrame or pandas.Series
            The concatenated results of each request when parsed by parse_fn.
        """
        data_objects = []
        request_start = start
        if end - start.tz_convert(end.tz) > pd.Timedelta(request_limit):
            # Recurse toward the beginning of the requested period to avoid
            # needing to sort the result.
            request_start = end - pd.Timedelta(request_limit)
            data_objects.append(
                self.chunk_value_requests(
                    api_path,
                    start,
                    request_start,
                    parse_fn=parse_fn,
                    params=params,
                    request_limit=request_limit,
                )
            )
        # Request the remaining data, period < = request_limit
        parameters = {'start': request_start, 'end': end}
        parameters.update(params)
        req = self.get(api_path, params=parameters)
        data_objects.append(parse_fn(req.json()))
        all_data = pd.concat(data_objects)

        # drop duplicate indices
        all_data = all_data[~all_data.index.duplicated(keep='first')]
        return all_data
