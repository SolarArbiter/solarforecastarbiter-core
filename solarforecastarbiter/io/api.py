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
    Subclass of requests.Session to handle requets to the SolarForecastArbiter
    API. The Session provides connection pooling, automatic retries for certain
    types of requets, default timeouts, and a default base url. Responses are
    converted into the appropriate class from datamodel.py or a pandas object.

    Parameters
    ----------
    access_token : string
        The base64 encoded Bearer token to authenticate with the API
    default_timeout : float or tuple, optional
        A default timeout to add to all requests. If a tuple, the first element
        is the connection timeout and the second is the read timeout.
        Default is 10 seconds for connection and 60 seconds to read from the
        server.
    base_url : string
        URL to use as the base for endpoints to APISession
    """

    def __init__(self, access_token, default_timeout=(10, 60),
                 base_url='https://api.solarforecastarbiter.org'):
        """
        """
        super().__init__()
        self.headers = {'Authorization': f'Bearer {access_token}',
                        'Accept': 'application/json',
                        'Accept-Encoding': 'gzip,deflate'}
        self.default_timeout = default_timeout
        self.base_url = base_url
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
        result.raise_for_status()
        return result

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
        datamodel.Site or SolarPowerPlant
           Dataclass with all the metadata for the site depending on if
           the Site is a power plant with modeling parameters or not.
        """
        req = self.get(f'/sites/{site_id}')
        return datamodel.process_site_dict(req.json())

    def list_sites(self):
        """
        List all the sites available to a user.

        Returns
        -------
        list of datamodel.Sites/SolarPowerPlants
        """
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
        return self._process_observation_dict(req.json())

    def list_observations(self):
        """
        List the observations a user has access to.

        Returns
        -------
        list of datamodel.Observation
        """
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
        return self._process_forecast_dict(req.json())

    def list_forecasts(self):
        """
        List all Forecasts a user has access to.

        Returns
        -------
        list of datamodel.Forecast
        """
        req = self.get('/forecasts/single/')
        return [self._process_forecast_dict(fx_dict)
                for fx_dict in req.json()]

    def get_observation_values(self, observation_id, start, end):
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

        Returns
        -------
        pandas.DataFrame
            With a datetime index and (value, quality_flag) columns
        """
        req = self.get(f'/observations/{observation_id}/values',
                       params={'start': start, 'end': end})
        return json_payload_to_observation_df(req.json())

    def get_forecast_values(self, forecast_id, start, end):
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

        Returns
        -------
        pandas.Series
           With the forecast values and a datetime index
        """
        req = self.get(f'/forecasts/single/{forecast_id}/values',
                       params={'start': start, 'end': end})
        return json_payload_to_forecast_series(req.json())

    def post_observation_values(self, observation_id, observation_df,
                                value_column='value',
                                quality_flag_column='quality_flag'):
        """
        Upload the given observation values to the appropriate observation_id
        of the API.

        Parameters
        ----------
        observation_id : string
            UUID of the observation to add values for
        observation_df : pandas.DataFrame
            Dataframe with a datetime index and the values and quality_flag
            to upload to the API
        value_column : string
            Column of the dataframe with the observation values
        quality_flag_column : string
            Column of the dataframe with the observation quality flags
        """
        json_vals = observation_df_to_json_payload(
            observation_df, value_column, quality_flag_column)
        self.post(f'/observations/{observation_id}/values',
                  data=json_vals,
                  headers={'Content-Type': 'application/json'})

    def post_forecast_values(self, forecast_id, forecast_obj,
                             value_column=None):
        """
        Upload the given forecast values to the appropriate forecast_id of the
        API

        Parameters
        ----------
        forecast_id : string
            UUID of the forecast to upload values to
        forecast_obj : pandas.DataFrame or Series
            Pandas object with a datetime index that contains the values to
            upload to the API
        value_column : string
            If forecast_obj is a pandas.DataFrame, upload data from this column
        """
        json_vals = forecast_object_to_json(forecast_obj, value_column)
        self.post(f'/forecasts/single/{forecast_id}/values',
                  data=json_vals,
                  headers={'Content-Type': 'application/json'})
