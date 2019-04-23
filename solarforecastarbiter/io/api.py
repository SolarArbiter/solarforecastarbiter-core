"""
Functions to connect to and process data from SolarForecastArbiter API
"""
import requests


from solarforecastarbiter import datamodel
from solarforecastarbiter.io.utils import (observation_df_to_json_payload,
                                           json_payload_to_observation_df)


class APISession(requests.Session):
    """
    why subclass of session?
    """
    base_url = 'https://api.solarforecastarbiter.org'

    def __init__(self, access_token):
        super().__init__()
        self.headers = {'Authentication': f'Bearer {access_token}',
                        'Content-Type': 'application/json'}

    def request(self, method, url, *args, **kwargs):
        if url.startswith('/'):
            url = f'{self.base_url}{url}'
        else:
            url = f'{self.base_url}/{url}'
        # handle basic errors
        super().request(method, url, *args, **kwargs)

    def get_site(self, site_id):
        req = self.get(f'/sites/{site_id}')
        return datamodel.process_site_dict(req.json())

    def list_sites(self):
        req = self.get('/sites/')
        return [datamodel.process_site_dict(site_dict)
                for site_dict in req.json()]

    def get_observation(self, observation_id):
        req = self.get(f'/observations/{observation_id}')
        return datamodel.process_dict_into_datamodel(
            req.json(), datamodel.Observation)

    def list_observations(self):
        req = self.get('/observations/')
        return [datamodel.process_dict_into_datamodel(
            obs_dict, datamodel.Observation) for obs_dict in
                req.json()]

    def get_forecast(self, forecast_id):
        req = self.get(f'/forecasts/single/{forecast_id}')
        return datamodel.process_dict_into_datamodel(
            req.json(), datamodel.Forecast)

    def list_forecasts(self):
        req = self.get('/forecasts/single/')
        return [datamodel.process_dict_into_datamodel(
            fx_dict, datamodel.Forecast) for fx_dict in
                req.json()]

    def get_observation_values(self, observation_id, start, end):
        # make sure response id and requested id match
        req = self.get(f'/observations/{observation_id}/values',
                       params={'start_time': start, 'end_time': end})
        return json_payload_to_observation_df(req.json())

    def write_values(self, object_id, values):
        pass
