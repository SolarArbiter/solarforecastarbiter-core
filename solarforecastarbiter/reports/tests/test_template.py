import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
import jinja2
from bokeh import __version__ as bokeh_version
from plotly import __version__ as plotly_version

from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import template


expected_metrics_json = """[{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"mae","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"mae","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"rmse","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"rmse","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"mbe","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"mbe","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"s","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"s","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"cost","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"cost","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"cost","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"mae","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"mae","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"rmse","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"rmse","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"mbe","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"mbe","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"s","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"s","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"cost","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"cost","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"cost","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"mae","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"mae","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"mae","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"rmse","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"rmse","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"rmse","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"mbe","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"mbe","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"mbe","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"s","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"s","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"s","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"cost","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"cost","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"cost","value":2,"index":1}]"""  # NOQA


expected_metadata_json = '[{"name": "0 Day GFS GHI", "interval_value_type": "interval_mean", "interval_length": 60.0, "interval_label": "beginning", "normalization_factor": 1.0, "uncertainty": 1.0, "cost": {"name": "example cost", "type": "constant", "parameters": {"cost": 1.0, "aggregation": "sum", "net": true}}, "forecast": {"name": "0 Day GFS GHI", "issue_time_of_day": "07:00", "lead_time_to_start": 0.0, "interval_length": 60.0, "run_length": 1440.0, "interval_label": "beginning", "interval_value_type": "interval_mean", "variable": "ghi", "forecast_id": "da2bc386-8712-11e9-a1c7-0a580a8200ae", "provider": "", "extra_parameters": "{\\"model\\": \\"gfs_quarter_deg_to_hourly_mean\\"}"}, "reference_forecast": null, "observation": {"name": "University of Arizona OASIS ghi", "variable": "ghi", "interval_value_type": "interval_mean", "interval_length": 1.0, "interval_label": "ending", "uncertainty": 1.0, "observation_id": "9f657636-7e49-11e9-b77f-0a580a8003e9", "provider": "", "extra_parameters": "{\\"network\\": \\"NREL MIDC\\", \\"network_api_id\\": \\"UAT\\", \\"network_api_abbreviation\\": \\"UA OASIS\\", \\"observation_interval_length\\": 1, \\"network_data_label\\": \\"Global Horiz (platform) [W/m^2]\\"}"}, "aggregate": null}, {"name": "Day Ahead GFS GHI", "interval_value_type": "interval_mean", "interval_length": 60.0, "interval_label": "beginning", "normalization_factor": 1000.0, "uncertainty": 15.0, "cost": {"name": "example cost", "type": "constant", "parameters": {"cost": 1.0, "aggregation": "sum", "net": true}}, "forecast": {"name": "Day Ahead GFS GHI", "issue_time_of_day": "07:00", "lead_time_to_start": 1440.0, "interval_length": 60.0, "run_length": 1440.0, "interval_label": "beginning", "interval_value_type": "interval_mean", "variable": "ghi", "forecast_id": "68a1c22c-87b5-11e9-bf88-0a580a8200ae", "provider": "", "extra_parameters": "{\\"model\\": \\"gfs_quarter_deg_to_hourly_mean\\"}"}, "reference_forecast": {"name": "0 Day GFS GHI", "issue_time_of_day": "07:00", "lead_time_to_start": 0.0, "interval_length": 60.0, "run_length": 1440.0, "interval_label": "beginning", "interval_value_type": "interval_mean", "variable": "ghi", "forecast_id": "refbc386-8712-11e9-a1c7-0a580a8200ae", "provider": "", "extra_parameters": "{\\"model\\": \\"gfs_quarter_deg_to_hourly_mean\\"}"}, "observation": {"name": "University of Arizona OASIS ghi", "variable": "ghi", "interval_value_type": "interval_mean", "interval_length": 1.0, "interval_label": "ending", "uncertainty": 1.0, "observation_id": "9f657636-7e49-11e9-b77f-0a580a8003e9", "provider": "", "extra_parameters": "{\\"network\\": \\"NREL MIDC\\", \\"network_api_id\\": \\"UAT\\", \\"network_api_abbreviation\\": \\"UA OASIS\\", \\"observation_interval_length\\": 1, \\"network_data_label\\": \\"Global Horiz (platform) [W/m^2]\\"}"}, "aggregate": null}, {"name": "GHI Aggregate FX 60", "interval_value_type": "interval_mean", "interval_length": 60.0, "interval_label": "beginning", "normalization_factor": 1.0, "uncertainty": 5.0, "cost": {"name": "example cost", "type": "constant", "parameters": {"cost": 1.0, "aggregation": "sum", "net": true}}, "forecast": {"name": "GHI Aggregate FX 60", "issue_time_of_day": "00:00", "lead_time_to_start": 0.0, "interval_length": 60.0, "run_length": 1440.0, "interval_label": "beginning", "interval_value_type": "interval_mean", "variable": "ghi", "forecast_id": "49220780-76ae-4b11-bef1-7a75bdc784e3", "provider": "", "extra_parameters": ""}, "reference_forecast": null, "observation": null, "aggregate": {"name": "Test Aggregate ghi", "description": "ghi agg", "variable": "ghi", "aggregate_type": "mean", "interval_length": 60.0, "interval_label": "ending", "timezone": "America/Denver", "observations": [{"effective_from": "2019-01-01T00:00:00+00:00", "effective_until": "2020-01-01T00:00:00+00:00", "observation_deleted_at": null, "observation_id": "123e4567-e89b-12d3-a456-426655440000"}, {"effective_from": "2019-01-01T00:00:00+00:00", "effective_until": null, "observation_deleted_at": null, "observation_id": "e0da0dea-9482-4073-84de-f1b12c304d23"}, {"effective_from": "2019-01-01T00:00:00+00:00", "effective_until": null, "observation_deleted_at": null, "observation_id": "b1dfe2cb-9c8e-43cd-afcf-c5a6feaf81e2"}], "aggregate_id": "458ffc27-df0b-11e9-b622-62adb5fd6af0", "provider": "Organization 1", "extra_parameters": "extra", "interval_value_type": "interval_mean"}}]'  # NOQA

expected_summary_stats_json = '[{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"forecast_mean","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"observation_mean","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"forecast_mean","value":1,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"observation_mean","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"forecast_mean","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"observation_mean","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"forecast_min","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"observation_min","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"forecast_min","value":1,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"observation_min","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"forecast_min","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"observation_min","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"forecast_max","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"observation_max","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"forecast_max","value":1,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"observation_max","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"forecast_max","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"observation_max","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"forecast_std","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"observation_std","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"forecast_std","value":1,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"observation_std","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"forecast_std","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"observation_std","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"forecast_median","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"total","metric":"observation_median","value":2,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"forecast_median","value":1,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"date","metric":"observation_median","value":2,"index":1546300800000},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"forecast_median","value":1,"index":1},{"name":"0 Day GFS GHI","abbrev":"0 Day GFS GHI","category":"hour","metric":"observation_median","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"forecast_mean","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"observation_mean","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"reference_forecast_mean","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"forecast_mean","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"observation_mean","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"reference_forecast_mean","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"forecast_mean","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"observation_mean","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"reference_forecast_mean","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"forecast_min","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"observation_min","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"reference_forecast_min","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"forecast_min","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"observation_min","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"reference_forecast_min","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"forecast_min","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"observation_min","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"reference_forecast_min","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"forecast_max","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"observation_max","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"reference_forecast_max","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"forecast_max","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"observation_max","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"reference_forecast_max","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"forecast_max","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"observation_max","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"reference_forecast_max","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"forecast_std","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"observation_std","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"reference_forecast_std","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"forecast_std","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"observation_std","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"reference_forecast_std","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"forecast_std","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"observation_std","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"reference_forecast_std","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"forecast_median","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"observation_median","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"total","metric":"reference_forecast_median","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"forecast_median","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"observation_median","value":2,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"date","metric":"reference_forecast_median","value":1,"index":1546300800000},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"forecast_median","value":1,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"observation_median","value":2,"index":1},{"name":"Day Ahead GFS GHI","abbrev":"Day Ahe. GFS GHI","category":"hour","metric":"reference_forecast_median","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"forecast_mean","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"observation_mean","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"forecast_mean","value":1,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"observation_mean","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"forecast_mean","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"observation_mean","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"forecast_min","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"observation_min","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"forecast_min","value":1,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"observation_min","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"forecast_min","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"observation_min","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"forecast_max","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"observation_max","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"forecast_max","value":1,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"observation_max","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"forecast_max","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"observation_max","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"forecast_std","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"observation_std","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"forecast_std","value":1,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"observation_std","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"forecast_std","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"observation_std","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"forecast_median","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"total","metric":"observation_median","value":2,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"forecast_median","value":1,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"date","metric":"observation_median","value":2,"index":1546300800000},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"forecast_median","value":1,"index":1},{"name":"GHI Aggregate FX 60","abbrev":"GHI Agg. FX 60","category":"hour","metric":"observation_median","value":2,"index":1}]'  # NOQA


@pytest.fixture
def mocked_timeseries_plots(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.timeseries_plots')
    mocked.return_value = ('{}', '{}', '{}', False)


@pytest.fixture
def mocked_timeseries_plots_exception(mocker):
    mocked = mocker.patch(
        'solarforecastarbiter.reports.figures.plotly_figures.timeseries_plots')
    mocked.side_effect = Exception


@pytest.fixture
def dash_url():
    return 'https://solarforecastarbiter.url'


@pytest.fixture(params=[True, False])
def with_series(request):
    return request.param


@pytest.fixture(params=[True, False])
def with_body(request):
    return request.param


@pytest.fixture
def expected_kwargs(dash_url):
    def fn(report, with_series, with_report=True):
        kwargs = {}
        kwargs['human_categories'] = datamodel.ALLOWED_CATEGORIES
        kwargs['human_metrics'] = datamodel.ALLOWED_METRICS
        kwargs['human_statistics'] = datamodel.ALLOWED_SUMMARY_STATISTICS
        kwargs['category_blurbs'] = datamodel.CATEGORY_BLURBS
        if with_report:
            kwargs['report'] = report
        kwargs['templating_messages'] = []
        if report.status == 'complete':
            kwargs['metrics_json'] = expected_metrics_json
            kwargs['metadata_json'] = expected_metadata_json
            kwargs['summary_stats'] = expected_summary_stats_json
        else:
            kwargs['metrics_json'] = '[]'
            kwargs['metadata_json'] = '[]'
            kwargs['summary_stats'] = '[]'
        if report.status == 'failed' and hasattr(report, 'raw_report'):
            kwargs['templating_messages'] = [
                'No data summary statistics were calculated '
                'with this report.']
        kwargs['dash_url'] = dash_url
        kwargs['bokeh_version'] = bokeh_version
        kwargs['plotly_version'] = plotly_version
        if with_series:
            kwargs['timeseries_spec'] = '{}'
            kwargs['scatter_spec'] = '{}'
            kwargs['timeseries_prob_spec'] = '{}'
            kwargs['includes_distribution'] = False
        return kwargs
    return fn


def test__get_render_kwargs_no_series(
        mocked_timeseries_plots, report_with_raw, dash_url, with_series,
        expected_kwargs):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        with_series
    )
    exp = expected_kwargs(report_with_raw, with_series)
    assert kwargs == exp


def test__get_render_kwargs_pending(
        mocked_timeseries_plots, pending_report, dash_url,
        expected_kwargs, mocker):
    kwargs = template._get_render_kwargs(
        pending_report,
        dash_url,
        False
    )
    exp = expected_kwargs(pending_report, False)
    assert kwargs == exp


def test__get_render_kwargs_with_series_exception(
        report_with_raw, dash_url, mocked_timeseries_plots_exception):
    kwargs = template._get_render_kwargs(
        report_with_raw,
        dash_url,
        True
    )
    assert 'timeseries_spec' not in kwargs
    assert 'scatter_spec' not in kwargs
    assert 'timeseries_prob_spec' not in kwargs


@pytest.fixture(params=[0, 1, 2])
def good_or_bad_report(request, report_with_raw, failed_report,
                       pending_report):
    if request.param == 0:
        out = report_with_raw
    elif request.param == 1:
        out = failed_report
    elif request.param == 2:
        out = pending_report
    return out


def test_get_template_and_kwargs(
        good_or_bad_report, dash_url, with_series, expected_kwargs,
        mocked_timeseries_plots, with_body):
    html_template, kwargs = template.get_template_and_kwargs(
        good_or_bad_report,
        dash_url,
        with_series,
        with_body
    )
    base = kwargs.pop('base_template')
    kwargs.pop('report')
    assert type(base) == jinja2.environment.Template
    assert type(html_template) == jinja2.environment.Template
    assert kwargs == expected_kwargs(good_or_bad_report,
                                     with_series, False)


def test_get_template_and_kwargs_no_stats(
        no_stats_report, dash_url, with_series, with_body,
        expected_kwargs, mocked_timeseries_plots):
    html_template, kwargs = template.get_template_and_kwargs(
        no_stats_report,
        dash_url,
        with_series,
        with_body
    )
    base = kwargs.pop('base_template')
    kwargs.pop('report')
    assert type(base) == jinja2.environment.Template
    assert type(html_template) == jinja2.environment.Template
    ek = expected_kwargs(no_stats_report,
                         with_series, False)
    ek['templating_messages'] = [
        'No data summary statistics were calculated with this report.']
    ek['summary_stats'] = '[]'
    assert kwargs == ek


def test_get_template_and_kwargs_bad_status(
        report_with_raw, dash_url, mocked_timeseries_plots):
    inp = report_with_raw.replace(status='notokay')
    with pytest.raises(ValueError):
        template.get_template_and_kwargs(
            inp, dash_url, False, True)


def test_render_html_body_only(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, True)
    assert rendered[:30] == '<style type="text/css" scoped>'


def test_render_html_full_html(report_with_raw, dash_url, with_series,
                               mocked_timeseries_plots):
    rendered = template.render_html(
        report_with_raw, dash_url, with_series, False)
    assert rendered[:46] == '<!doctype html>\n<html lang="en" class="h-100">'


def test_build_metrics_json(report_with_raw):
    assert template.build_metrics_json(
        report_with_raw) == expected_metrics_json


def test_build_metrics_json_empty(pending_report):
    assert template.build_metrics_json(pending_report) == '[]'


def test_build_summary_stats_json(report_with_raw):
    assert template.build_summary_stats_json(
        report_with_raw) == expected_summary_stats_json


def test_build_summary_stats_json_no_stats(no_stats_report):
    with pytest.raises(ValueError):
        template.build_summary_stats_json(no_stats_report)


def test_build_summary_stats_json_empty(pending_report):
    assert template.build_summary_stats_json(
        pending_report) == '[]'


def test_build_metadata_json(report_with_raw):
    out = template.build_metadata_json(report_with_raw)
    assert out == expected_metadata_json


def test_build_metadata_json_empty(pending_report):
    assert template.build_metadata_json(pending_report) == '[]'


def test_build_metadata_json_prob_report(pending_report, raw_report_xy):
    rep = pending_report.replace(status='complete',
                                 raw_report=raw_report_xy(False))
    out = template.build_metadata_json(rep)
    outd = json.loads(out)
    assert outd[0]['forecast']['constant_values'] == [25.0, 50.0, 75.0]


@pytest.mark.parametrize('val,expected', [
    ('<p> paragraph</p>', ' paragraph\n'),
    ('<em>italic</em>', '\\emph{italic}'),
    ('<code>nan</code>', '\\verb|nan|'),
    ('<b>bold</b>', '\\textbf{bold}'),
    ('<ol>\n<li>item one</li>\n</ol>',
     '\\begin{enumerate}\n\\item item one\n\n\\end{enumerate}'),
    ('<a href="tolink" class="what">stuff</a>', 'stuff'),
    (('<p>paragraph one <em>important</em> code here <code>null</code>'
      ' and more <b>bold</b><em> critical</em> <code>here</code></p>'
      ' <b>masbold</b>'),
     ('paragraph one \\emph{important} code here \\verb|null|'
      ' and more \\textbf{bold}\\emph{ critical} \\verb|here|\n'
     ' \\textbf{masbold}'))
])
def test_html_to_tex(val, expected):
    assert template._html_to_tex(val) == expected


def test_render_pdf(report_with_raw, dash_url):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    rendered = template.render_pdf(report_with_raw, dash_url)
    assert rendered.startswith(b'%PDF')


def test_render_pdf_special_chars(
        ac_power_observation_metadata, ac_power_forecast_metadata, dash_url,
        fail_pdf, preprocessing_result_types, report_metrics):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    quality_flag_filter = datamodel.QualityFlagFilter(
        (
            "USER FLAGGED",
        )
    )
    forecast = ac_power_forecast_metadata.replace(
        name="ac_power forecast (why,)  ()'-_,")
    observation = ac_power_observation_metadata.replace(
        name="ac_power observations  ()'-_,")
    fxobs = datamodel.ForecastObservation(forecast,
                                          observation)
    tz = 'America/Phoenix'
    start = pd.Timestamp('20190401 0000', tz=tz)
    end = pd.Timestamp('20190404 2359', tz=tz)
    report_params = datamodel.ReportParameters(
        name="NREL MIDC OASIS GHI Forecast Analysis  ()'-_,",
        start=start,
        end=end,
        object_pairs=(fxobs,),
        metrics=("mae", "rmse", "mbe", "s"),
        categories=("total", "date", "hour"),
        filters=(quality_flag_filter,)
    )
    report = datamodel.Report(
        report_id="56c67770-9832-11e9-a535-f4939feddd83",
        report_parameters=report_params
    )
    qflags = list(
        f.quality_flags for f in report.report_parameters.filters if
        isinstance(f, datamodel.QualityFlagFilter)
    )
    qflags = list(qflags[0])
    ser_index = pd.date_range(
        start, end,
        freq=to_offset(forecast.interval_length),
        name='timestamp')
    ser = pd.Series(
        np.repeat(100, len(ser_index)), name='value',
        index=ser_index)
    pfxobs = datamodel.ProcessedForecastObservation(
        forecast.name,
        fxobs,
        forecast.interval_value_type,
        forecast.interval_length,
        forecast.interval_label,
        valid_point_count=len(ser),
        validation_results=tuple(datamodel.ValidationResult(
            flag=f, count=0) for f in qflags),
        preprocessing_results=tuple(datamodel.PreprocessingResult(
            name=t, count=0) for t in preprocessing_result_types),
        forecast_values=ser,
        observation_values=ser
    )

    figs = datamodel.RawReportPlots(
        (
            datamodel.PlotlyReportFigure.from_dict(
                {
                    'name': 'mae tucson ac_power',
                    'spec': '{"data":[{"x":[1],"y":[1],"type":"bar"}]}',
                    'pdf': fail_pdf,
                    'figure_type': 'bar',
                    'category': 'total',
                    'metric': 'mae',
                    'figure_class': 'plotly',
                }
            ),), '4.5.3',
    )
    raw = datamodel.RawReport(
        generated_at=report.report_parameters.end,
        timezone=tz,
        plots=figs,
        metrics=report_metrics(report),
        processed_forecasts_observations=(pfxobs,),
        versions=(('test',  'test_with_underscore?'),),
        messages=(datamodel.ReportMessage(
            message="Failed to make metrics for ac_power forecast ()'-_,",
            step='', level='', function=''),))
    rr = report.replace(raw_report=raw)
    rendered = template.render_pdf(rr, dash_url)
    assert rendered.startswith(b'%PDF')


def test_render_pdf_not_settled(report_with_raw, dash_url):
    if shutil.which('pdflatex') is None:  # pragma: no cover
        pytest.skip('pdflatex must be on PATH to generate PDF reports')
    with pytest.raises(RuntimeError):
        template.render_pdf(report_with_raw, dash_url, 1)


def test_render_pdf_process_error(report_with_raw, dash_url, mocker):
    mocker.patch('solarforecastarbiter.reports.template.subprocess.run',
                 side_effect=subprocess.CalledProcessError(
                     cmd='', returncode=1))
    with pytest.raises(subprocess.CalledProcessError):
        template.render_pdf(report_with_raw, dash_url)


@pytest.mark.parametrize('text,exp', [
    ('stuf <a href="http://">link</a>', 'stuf \\href{http://}{link}'),
    ('a bunch\nstuf <a href="https://blah">\nlink\nmore\n</a>',
     'a bunch\nstuf \\href{https://blah}{\nlink\nmore\n}'),
    ('no link at all', 'no link at all')
])
def test_link_filter(text, exp):
    new = template._link_filter(text)
    assert new == exp


NOTWORD = re.compile('[^\\w-]')


def test_not_word():
    assert NOTWORD.match('-') is None
    assert NOTWORD.match('test') is None
    assert NOTWORD.match('+') is not None
    assert NOTWORD.match('*') is not None
    assert NOTWORD.match('^') is not None
    assert NOTWORD.match('%') is not None


@pytest.mark.parametrize('val', [
    'ac_power Prob(f <= 10.0%)',
    'ac_power Prob(f <= 10MW) = 10.99%',
    'ac_power Prob(f >= -all)',
    '*!@#$%?.}{[]}<>,."',
    'testit - now\\ W/m^2'
])
def test_figure_name_filter(val):
    new = template._figure_name_filter(val)
    match = NOTWORD.match(new)
    assert match is None


def test__get_render_kwargs_with_missing_fx_data(
        report_with_raw, dash_url):
    raw_report = report_with_raw.raw_report
    missing_data = report_with_raw.replace(
        raw_report=raw_report.replace(
            processed_forecasts_observations=tuple(
                pfxobs.replace(forecast_values=None)
                for pfxobs in raw_report.processed_forecasts_observations
            )
        )
    )

    kwargs = template._get_render_kwargs(
        missing_data,
        dash_url,
        True
    )
    assert 'timeseries_spec' not in kwargs
    assert 'scatter_spec' not in kwargs
    assert 'timeseries_prob_spec' not in kwargs
    assert not kwargs['includes_distribution']


def test__get_render_kwargs_with_missing_obs_data(
        report_with_raw, dash_url):
    raw_report = report_with_raw.raw_report
    missing_data = report_with_raw.replace(
        raw_report=raw_report.replace(
            processed_forecasts_observations=tuple(
                pfxobs.replace(observation_values=None)
                for pfxobs in raw_report.processed_forecasts_observations
            )
        )
    )

    kwargs = template._get_render_kwargs(
        missing_data,
        dash_url,
        True
    )
    assert 'timeseries_spec' in kwargs
    assert 'scatter_spec' not in kwargs
    assert 'timeseries_prob_spec' not in kwargs
    assert not kwargs['includes_distribution']
