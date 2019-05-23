"""
Make a report.

Steps:

  1. Consume metadata defined in :py:mod:`~solarforecastarbiter.datamodel`
  2. Run validation of metadata. Metadata creation might also include
     some validation so this may not be necessary.
  3. Get data using io.api.
  4. Align observation data to forecast data using metrics subpackage.
  5. Compute metrics specified in metadata using metrics subpackage
  6. Assemble metrics into output format. Maybe nested JSON with keys like:
        * metadata
            * checksum of data
            * date created
            * versions
        * metrics
            * total
            * filter A
                * metric 1
                * metric 2
            * filter B
                * metric 1
                * metric 2
            * filter C
                * metric 1
                * metric 2
       The JSON could be large and difficult for a person to read, but
       should be relatively easy to parse.
    7. Submit report to API.


Considerations:

* API uses queue system to initiate report generation
* Functions should not require an API session unless they really need it.
* The bokeh plots in the html version will be rendered at
  client load time. The metrics data will be immediately available, but
  the API will need to call the data query and realignment functions
  to be able to create time series, scatter, etc. plots.
"""

import json
import pkg_resources
from pkg_resources import DistributionNotFound
import platform

import pandas as pd

from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.reports import figures, template


def get_data_for_report(session, report):
    """
    Get data for report.

    1 API call is made for each unique forecast and observation object.

    Parameters
    ----------
    session : solarforecastarbiter.api.APISession
        API session for getting and posting data
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report

    Returns
    -------
    data : dict
        Keys are Forecast and Observation objects, values are
        the corresponding data.
    """
    data = {}
    for fxobs in report.forecast_observations:
        # forecasts and especially observations may be repeated.
        # only get the raw data once.
        if fxobs.forecast not in data:
            data[fxobs.forecast] = session.get_forecast_values(
                fxobs.forecast, report.start, report.end)
        if fxobs.observation not in data:
            data[fxobs.observation] = session.get_observation_values(
                fxobs.observation, report.start, report.end)
    return data


def data_dict_to_fxobs_data(data, report):
    """
    Sorts the data dict into a new dict where the keys are the report's
    ForecastObservation objects and the values are tuples of
    (forecast values, observation values).
    """
    return {fxobs: (data[fxobs.forecast], data[fxobs.observation])
            for fxobs in report.forecast_observations}


def create_metadata(report):
    """
    Create prereport metadata.

    Returns
    -------
    metadata: dict
    """
    metadata = dict(
        name=report.name, start=report.start, end=report.end,
        now=pd.Timestamp.utcnow())
    metadata['versions'] = get_versions()
    metadata['validation_issues'] = get_validation_issues()
    return metadata


def get_versions():
    packages = [
        'solarforecastarbiter',
        'pvlib',
        'pandas',
        'numpy',
        'bokeh',
        'netcdf4',
        'xarray',
        'tables',
        'numexpr',
        'bottleneck',
        'jinja2',
    ]
    versions = {}
    for p in packages:
        try:
            v = pkg_resources.get_distribution(p).version
        except DistributionNotFound:
            v = 'None'
        versions[p] = v
    versions['python'] = platform.python_version()
    versions['platform'] = platform.platform()
    return versions


def get_validation_issues():
    return {}


def get_data_for_report_embed(session, report):
    """
    Get time series data for report.

    1 API call is made for each unique forecast and observation object.

    Parameters
    ----------
    session : solarforecastarbiter.api.APISession
        API session for getting and posting data
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report

    Returns
    -------
    fx_obs_cds : list
        List of (forecast, observation, ColumnDataSource) tuples to
        pass to bokeh plotting objects.
    """
    data = get_data_for_report(session, report)
    fxobs_data = data_dict_to_fxobs_data(data, report)
    fx_obs_cds = [(k, figures.construct_fx_obs_cds(v[0], v[1]))
                  for k, v in fxobs_data.items()]
    return fx_obs_cds


def create_prereport_from_data(report, data):
    """
    Create a pre-report using data and report metadata.

    The prereport is a markdown file with all elements rendered except
    for bokeh plots.

    Parameters
    ----------
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    data : dict
        Keys are all Forecast and Observation objects in the report,
        values are the corresponding data.

    Returns
    -------
    metadata : str
        prereport metadata in JSON format.
    prereport : str
        prereport in markdown format.
    """
    # call function: metrics.align_observations_forecasts
    # call function: metrics.calculate_many
    # call function: reports.metrics_to_JSON
    # call function: add some metadata to JSON
    # call function: configure tables and figures, add to JSON
    # call function: pre-render report in md format
    # return json, prereport

    # debug
    from solarforecastarbiter.reports.tests.test_main import dummy_metrics
    metrics = dummy_metrics.copy()

    metadata = create_metadata(report)
    prereport = template.prereport(metadata, metrics)

    # put the metrics in the metadata because why not
    metadata['metrics'] = metrics
    metadata_json = json.dumps(metadata)
    return metadata_json, prereport


def create_prereport_from_metadata(access_token, report, base_url=None):
    """
    Create a pre-report using data from API and report metadata.

    Typically called as a task.

    Parameters
    ----------
    session : solarforecastarbiter.api.APISession
        API session for getting and posting data
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report

    Returns
    -------
    None
    """
    session = APISession(access_token, base_url=base_url)
    data = get_data_for_report(session, report)
    metadata, prereport = create_prereport_from_data(report, data)
    session.post_report(metadata, prereport)


def prereport_to_report(access_token, report, metadata, prereport,
                        base_url=None):
    """
    Convert pre-report to full report.

    Parameters
    ----------
    session : solarforecastarbiter.api.APISession
        API session for getting and posting data
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    metadata : str, json
        Describes the prereport
    prereport : str, md
        The templated pre-report.

    Returns
    -------
    report : str, markdown
        The full report.
    """
    session = APISession(access_token, base_url=base_url)
    fx_obs_cds = get_data_for_report_embed(session, report)
    report = template.add_figures_to_prereport(fx_obs_cds, report, metadata,
                                               prereport)
    return report


def prereport_to_pdf(access_token, metadata, prereport, base_url=None):
    """
    Maybe not necessary if we can go from fully rendered markdown to pdf.
    """
    report = prereport_to_report(access_token, metadata, prereport,
                                 base_url=base_url)
    report = report_to_pdf(report)
    return report


def report_to_pdf(report):
    # call pandoc
    raise NotImplementedError


def prereport_to_jupyter(report):
    raise NotImplementedError
