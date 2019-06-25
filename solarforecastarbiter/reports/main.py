"""
Make a report.

Steps:

  1. Consume metadata defined in :py:mod:`~solarforecastarbiter.datamodel`
  2. Run validation of metadata. Metadata creation might also include
     some validation so this may not be necessary.
  3. Get data using io.api.
  4. Align observation data to forecast data using metrics subpackage.
  5. Compute metrics specified in metadata using metrics subpackage
  6. Assemble metrics and aligned data into a raw report object which
     can then later be converted to a HTML or PDF report
  7. Prepare to post raw report, metrics, and aligned data to the API.
     The raw report sent to the API be JSON with keys like:
        * metadata
            * checksum of data
            * date created
            * versions
        * data: base64 encoded raw report

      The metrics will aloso be JSON with keys like:
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
       The aligned data will be encoded separately in a binary format.
  7. Submit the raw report to API.
  8. When the report is later requested from the API, get the raw report,
     metrics, and aligned data and convert to HTML or PDF


Considerations:

* API uses queue system to initiate report generation
* Functions should not require an API session unless they really need it.
* The bokeh plots in the html version will be rendered at
  client load time. The metrics data will be immediately available, but
  the API will need to call for the aligned data separately
  to be able to create time series, scatter, etc. plots.
"""
import pkg_resources
from pkg_resources import DistributionNotFound
import platform


import pandas as pd


from solarforecastarbiter.io.api import APISession
from solarforecastarbiter import datamodel
from solarforecastarbiter.reports import figures, template, metrics


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
        Keys are Forecast and Observation uuids, values are
        the corresponding data.
    """
    data = {}
    for fxobs in report.forecast_observations:
        # forecasts and especially observations may be repeated.
        # only get the raw data once.
        forecast_id = fxobs.forecast.forecast_id
        observation_id = fxobs.observation.observation_id
        if forecast_id not in data:
            data[fxobs.forecast] = session.get_forecast_values(
                forecast_id, report.start, report.end)
        if observation_id not in data:
            data[fxobs.observation] = session.get_observation_values(
                observation_id, report.start, report.end)
    return data


def create_metadata(report_request):
    """
    Create metadata for the raw report.

    Returns
    -------
    metadata: dict
    """
    versions = get_versions()
    validation_issues = get_validation_issues()
    metadata = datamodel.ReportMetadata(
        name=report_request.name, start=report_request.start,
        end=report_request.end, now=pd.Timestamp.utcnow(),
        versions=versions, validation_issues=validation_issues)
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
    test = {
        'USER FLAGGED': 0, 'NIGHTTIME': 39855,
        # 'CLOUDY': 0, 'SHADED': 0,
        # 'UNEVEN FREQUENCY': 4,
        'LIMITS EXCEEDED': 318,
        # 'CLEARSKY EXCEEDED': 9548,
        'STALE VALUES': 12104,
        'INTERPOLATED VALUES': 5598,
        # 'CLIPPED VALUES': 0,
        'INCONSISTENT IRRADIANCE COMPONENTS': 0,
        # 'NOT VALIDATED': 0
    }
    return test


def create_raw_report_from_data(report, data):
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

    metadata = create_metadata(report)

    processed_fxobs = metrics.validate_resample_align(report, data)

    # needs to be in json
    metrics_list = metrics.loop_forecasts_calculate_metrics(
        report, processed_fxobs)
    # can be ~50kb
    report_template = template.prereport(report, metadata, metrics_list)

    raw_report = datamodel.RawReport(
        metadata=metadata, template=report_template, metrics=metrics_list,
        processed_forecasts_observations=tuple(processed_fxobs))
    return raw_report


def create_raw_report_from_metadata(access_token, report, base_url=None):
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
    raw_report = create_raw_report_from_data(report, data)
    session.post_raw_report(report.report_id, raw_report)
    return raw_report


def render_raw_report(raw_report):
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
    fx_obs_cds = [
        (pfxobs.original, figures.construct_fx_obs_cds(
            pfxobs.forecast_values, pfxobs.observation_values))
        for pfxobs in raw_report.processed_forecasts_observations]
    report = template.add_figures_to_prereport(
        fx_obs_cds, raw_report.metadata, raw_report.template)
    return report


def report_to_pdf(report):
    # call pandoc
    raise NotImplementedError


def report_to_jupyter(report):
    raise NotImplementedError
