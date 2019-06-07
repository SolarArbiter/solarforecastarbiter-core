import os
from functools import partial
import json
from pathlib import Path

import pandas as pd

from solarforecastarbiter.reference_forecasts import main, models
from solarforecastarbiter.io.fetch import nwp as fetch_nwp


from solarforecastarbiter.io import nwp, api
# find the files
base_path = Path('/data')
# define file loading function that knows where to find the files
load_forecast = partial(nwp.load_forecast, base_path=base_path)

forecast_ids = [
    'da2bc386-8712-11e9-a1c7-0a580a8200ae',
    'da3a9be8-8712-11e9-ae90-0a580a8200ae',
    'da41e692-8712-11e9-8302-0a580a8200ae',
    '68a1c22c-87b5-11e9-bf88-0a580a8200ae'
]


def groupby_forecasts(forecasts):
    """
    Parameters
    ----------
    forecasts : list of datamodel.Forecast

    Returns
    -------
    grouped : dict
        Keys are the Forecast objects to pass to
        :py:func:`~solarforecastarbiter.reference_forecasts.main.run`.
        Values are the Forecast objects for which to use processed data.
    """
    data = []
    for fx in forecasts:
        nfx = fx.to_dict()
        nfx['fx'] = fx
        nfx['forecast_id'] = fx.forecast_id
        nfx['site'] = fx.site
        data.append(nfx)
    grouped = pd.DataFrame(data).groupby([
        'site', 'issue_time_of_day',
        'lead_time_to_start', 'interval_length',
        'run_length', 'interval_label',
        'interval_value_type', 'extra_parameters'
    ])
    return grouped


def get_init_time(run_time, fetch_metadata):
    """Determine the most recent init time for which all forecast data is
    available."""
    run_finish = (pd.Timedelta(fetch_metadata['delay_to_first_forecast']) +
                  pd.Timedelta(fetch_metadata['avg_max_run_length']))
    freq = fetch_metadata['update_freq']
    init_time = (run_time - run_finish).floor(freq=freq)
    return init_time


def run_reference_forecast(forecast, run_time, issue_time):
    extra_params = json.loads(forecast.extra_parameters)
    fetch_metadata = getattr(
        fetch_nwp, extra_params['fetch_metadata'])
    init_time = get_init_time(run_time, fetch_metadata)
    forecast_start, forecast_end = main.get_forecast_start_end(forecast,
                                                               issue_time)
    forecast_end -= pd.Timedelta('1s')
    model = getattr(models, extra_params['model'])
    # for testing
    model = partial(model, load_forecast=load_forecast)
    fx = main.run(forecast.site, model, init_time,
                  forecast_start, forecast_end)
    # fx is tuple of ghi, dni, dhi, air_temperature, wind_speed, ac_power
    # probably return for another function to post data to api
    return fx


token = api.request_cli_access_token(
    os.environ['SFA_API_USER'], os.environ['SFA_API_PASSWORD'])
session = api.APISession(token)
forecasts = [session.get_forecast(fxid) for fxid in forecast_ids]
grouped = groupby_forecasts(forecasts)
for name, group in grouped:
    variables = ('ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed',
                 'ac_power')
    ids = [group[group.variable == var].get('forecast_id').values
           for var in variables]
    key_fx = group[group.variable == 'ghi'].iloc[0].fx
    for day in pd.date_range(start='20190401T0700Z', end='20190604T0700Z',
                             freq='1D'):
        print(f'running for {day}')
        fxs = run_reference_forecast(key_fx, day, day)
        for fxids, fx_vals in zip(ids, fxs):
            if len(fxids) > 0 and fx_vals is not None:
                for fxid in fxids:
                    session.post_forecast_values(fxid, fx_vals)

