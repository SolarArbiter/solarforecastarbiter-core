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

      The metrics will also be JSON with keys like:
        * metrics
            * total
            * category A
                * metric 1
                * metric 2
            * category B
                * metric 1
                * metric 2
            * category C
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
import logging
import pkg_resources
import platform


import pandas as pd


from solarforecastarbiter.io.api import APISession
from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import preprocessing, calculator
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
        forecast_id = fxobs.forecast.forecast_id
        if fxobs.forecast not in data:
            data[fxobs.forecast] = session.get_forecast_values(
                forecast_id, report.start, report.end)
        if fxobs.data_object not in data:
            data[fxobs.data_object] = session.get_values(
                fxobs.data_object, report.start, report.end)
    return data


def create_metadata(report_request):
    """
    Create metadata for the raw report.

    Returns
    -------
    metadata: solarforecastarbiter.datamodel.ReportMetadata
    """
    versions = get_versions()
    timezone = infer_timezone(report_request)
    metadata = datamodel.ReportMetadata(
        name=report_request.name, start=report_request.start,
        end=report_request.end, now=pd.Timestamp.utcnow(),
        timezone=timezone, versions=versions)
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
        except pkg_resources.DistributionNotFound:
            v = 'None'
        versions[p] = v
    versions['python'] = platform.python_version()
    versions['platform'] = platform.platform()
    return versions


def infer_timezone(report_request):
    # maybe not ideal when comparing across sites. might need explicit
    # tz options ('infer' or spec IANA tz) in report interface.
    fxobs_0 = report_request.forecast_observations[0]
    if isinstance(fxobs_0, datamodel.ForecastObservation):
        timezone = fxobs_0.observation.site.timezone
    else:
        timezone = fxobs_0.aggregate.timezone
    return timezone


class ListHandler(logging.Handler):
    records = []

    def emit(self, record):
        self.records.append(record)


def create_raw_report_from_data(report, data):
    """
    Create a raw report using data and report metadata.

    Parameters
    ----------
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    data : dict
        Keys are all Forecast and Observation objects in the report,
        values are the corresponding data.

    Returns
    -------
    raw_report : datamodel.RawReport

    Todo
    ----
    * add reference forecast
    """
    message_logger = logging.getLogger('report_generation')
    message_logger.propagate = False
    handler = ListHandler()
    message_logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)
    message_logger.addHandler(handler)

    metadata = create_metadata(report)

    # Validate and resample
    processed_fxobs = preprocessing.process_forecast_observations(
        report.forecast_observations, report.filters, data, metadata.timezone,
        _logger=message_logger)

    # Calculate metrics
    metrics_list = calculator.calculate_metrics(processed_fxobs,
                                                list(report.categories),
                                                list(report.metrics))

    report_plots = figures.raw_report_plots(report, metrics_list)

    raw_report = datamodel.RawReport(
        metadata=metadata, plots=report_plots, metrics=tuple(metrics_list),
        processed_forecasts_observations=tuple(processed_fxobs))
    return raw_report


def compute_report(access_token, report_id, base_url=None):
    """
    Create a raw report using data from API.

    Typically called as a task.

    Parameters
    ----------
    session : solarforecastarbiter.api.APISession
        API session for getting and posting data
    report_id : str
        ID of the report to fetch from the API and generate the raw
        report for

    Returns
    -------
    raw_report : datamodel.RawReport
    """
    session = APISession(access_token, base_url=base_url)
    try:
        report = session.get_report(report_id)
        data = get_data_for_report(session, report)
        raw_report = create_raw_report_from_data(report, data)
        session.post_raw_report(report.report_id, raw_report)
    except Exception:
        session.update_report_status(report_id, 'failed')
        raise
    return raw_report


def report_to_html_body(
        report, dash_url='https://dashboard.solarforecastarbiter.org'):
    body = template.render_html(report, dash_url, with_timeseries=True,
                                body_only=True)
    return body


def report_to_pdf(report):
    # call pandoc
    raise NotImplementedError


def report_to_jupyter(report):
    raise NotImplementedError
