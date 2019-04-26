"""
Functions to connect to and process data from SolarForecastArbiter API
"""
import requests
from urllib3 import Retry


from solarforecastarbiter import datamodel
from solarforecastarbiter.io.utils import (json_payload_to_observation_df,
                                           json_payload_to_forecast_series,
                                           observation_df_to_json_payload,
                                           forecast_object_to_json)


class APISession(requests.Session):
    """
    why subclass of session?
    """
    base_url = 'https://api.solarforecastarbiter.org'

    def __init__(self, access_token, default_timeout=(10, 60)):
        super().__init__()
        self.headers = {'Authentication': f'Bearer {access_token}',
                        'Content-Type': 'application/json'}
        self.default_timeout = default_timeout
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
        if url.startswith('/'):
            url = f'{self.base_url}{url}'
        else:
            url = f'{self.base_url}/{url}'
        # set a defautl timeout so we never hang indefinitely
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.default_timeout

        result = super().request(method, url, *args, **kwargs)
        return result

    def get_site(self, site_id):
        req = self.get(f'/sites/{site_id}')
        return datamodel.process_site_dict(req.json())

    def list_sites(self):
        req = self.get('/sites/')
        return [datamodel.process_site_dict(site_dict)
                for site_dict in req.json()]

    def _process_observation_dict(self, observation_dict):
        obs_dict = observation_dict.copy()
        site_id = obs_dict['site_id']
        site = self.get_site(site_id)
        obs_dict['site'] = site
        return datamodel.process_dict_into_datamodel(
            obs_dict, datamodel.Observation)

    def get_observation(self, observation_id):
        req = self.get(f'/observations/{observation_id}')
        return self._process_observation_dict(req.json())

    def list_observations(self):
        req = self.get('/observations/')
        return [self._process_observation_dict(obs_dict)
                for obs_dict in req.json()]

    def _process_forecast_dict(self, forecast_dict):
        fx_dict = forecast_dict.copy()
        site_id = forecast_dict['site_id']
        site = self.get_site(site_id)
        fx_dict['site'] = site
        return datamodel.process_dict_into_datamodel(
            fx_dict, datamodel.Forecast)

    def get_forecast(self, forecast_id):
        req = self.get(f'/forecasts/single/{forecast_id}')
        return self._process_forecast_dict(req.json())

    def list_forecasts(self):
        req = self.get('/forecasts/single/')
        return [self._process_forecast_dict(fx_dict)
                for fx_dict in req.json()]

    def get_observation_values(self, observation_id, start, end):
        # make sure response id and requested id match
        # json to df errors?
        req = self.get(f'/observations/{observation_id}/values',
                       params={'start_time': start, 'end_time': end})
        return json_payload_to_observation_df(req.json())

    def get_forecast_values(self, forecast_id, start, end):
        req = self.get(f'/forecasts/single/{forecast_id}/values',
                       params={'start_time': start, 'end_time': end})
        return json_payload_to_forecast_series(req.json())

    def post_observation_values(self, observation_id, observation_df,
                                value_column='value',
                                quality_flag_column='quality_flag'):
        json_vals = observation_df_to_json_payload(
            observation_df, value_column, quality_flag_column)
        self.post(f'/observations/{observation_id}/values',
                  data=json_vals)

    def post_forecast_values(self, forecast_id, forecast_obj,
                             value_column=None):
        json_vals = forecast_object_to_json(forecast_obj, value_column)
        self.post(f'/forecasts/single/{forecast_id}/values',
                  data=json_vals)
