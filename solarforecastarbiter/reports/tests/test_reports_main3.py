import pandas as pd

from solarforecastarbiter import datamodel
from solarforecastarbiter.io.api import APISession, request_cli_access_token
from solarforecastarbiter.reports import template, main


tz = 'America/Phoenix'
start = pd.Timestamp('20190401 0000', tz=tz)
end = pd.Timestamp('20190403 2359', tz=tz)

# don't store your real passwords or tokens in plain text like this!
# only for demonstration purposes!
token = request_cli_access_token('testing@solarforecastarbiter.org',
                                 'Thepassword123!')
session = APISession(token)

# GHI
observation = session.get_observation('9f657636-7e49-11e9-b77f-0a580a8003e9')

# current day (0) and day ahead (1) GHI forecasts derived from GFS
forecast_0 = session.get_forecast('da2bc386-8712-11e9-a1c7-0a580a8200ae')
forecast_1 = session.get_forecast('68a1c22c-87b5-11e9-bf88-0a580a8200ae')

fxobs0 = datamodel.ForecastObservation(forecast_0, observation)
fxobs1 = datamodel.ForecastObservation(forecast_1, observation)

quality_flag_filter = datamodel.QualityFlagFilter([
    'USER FLAGGED', 'NIGHTTIME', 'LIMITS EXCEEDED', 'STALE VALUES',
    'INTERPOLATED VALUES', 'INCONSISTENT IRRADIANCE COMPONENTS'])

report = datamodel.Report(
    name='NREL MIDC OASIS GHI Forecast Analysis',
    start=start,
    end=end,
    forecast_observations=(fxobs0, fxobs1),
    metrics=('mae', 'rmse', 'mbe'),
    filters=(quality_flag_filter, )
)

metadata, prereport = main.create_prereport_from_metadata(token, report)

with open('bokeh_prereport.md', 'w') as f:
    f.write(prereport)

fx_obs_cds = main.get_data_for_report_embed(session, report)

prereport_html = template.prereport_to_html(prereport)

with open('bokeh_prereport.html', 'w') as f:
    f.write(prereport_html)

body = template.add_figures_to_prereport(
    fx_obs_cds, report, metadata, prereport_html)

full_report = template.full_html(body)

with open('bokeh_report.html', 'w') as f:
    f.write(full_report)
