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
from functools import wraps
import pkg_resources
import platform


import pandas as pd


from solarforecastarbiter.io.api import APISession
from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics import preprocessing, calculator
from solarforecastarbiter.reports.figures import plotly_figures
from solarforecastarbiter.utils import hijack_loggers
from solarforecastarbiter.validation.tasks import apply_validation


def get_data_for_report(session, report):
    """
    Get data for report.

    1 API call is made for each unique forecast and observation object.

    Parameters
    ----------
    session : :py:class:`solarforecastarbiter.api.APISession`
        API session for getting and posting data
    report : :py:class:`solarforecastarbiter.datamodel.Report`
        Metadata describing report

    Returns
    -------
    data : dict
        Keys are Forecast and Observation objects, values are
        the corresponding data. Keys also include any reference
        forecasts that exist in the report.
    """
    data = {}
    start = report.report_parameters.start
    end = report.report_parameters.end
    for fxobs in report.report_parameters.object_pairs:
        # forecasts and especially observations may be repeated.
        # only get the raw data once.
        if fxobs.forecast not in data:
            # use get_values instead of get_forecast_values so that api module
            # can handle determ., prob constant value, or prob group values
            data[fxobs.forecast] = session.get_values(
                fxobs.forecast, start, end)
        if fxobs.data_object not in data:
            obs_data = session.get_values(
                fxobs.data_object, start, end)
            data[fxobs.data_object] = apply_validation(
                fxobs.data_object, obs_data)
        if fxobs.reference_forecast is not None:
            if fxobs.reference_forecast not in data:
                data[fxobs.reference_forecast] = session.get_values(
                    fxobs.reference_forecast, start, end)

    return data


def get_versions():
    packages = [
        'solarforecastarbiter',
        'pvlib',
        'pandas',
        'numpy',
        'scipy',
        'statsmodels',
        'plotly',
        'bokeh',
        'netcdf4',
        'xarray',
        'tables',
        'numexpr',
        'bottleneck',
        'jinja2',
    ]
    versions = []
    for p in packages:
        try:
            v = pkg_resources.get_distribution(p).version
        except pkg_resources.DistributionNotFound:
            v = 'None'
        versions.append((p, str(v)))
    versions.append(('python', str(platform.python_version())))
    versions.append(('platform', platform.platform()))
    return tuple(versions)


def infer_timezone(report_parameters):
    # maybe not ideal when comparing across sites. might need explicit
    # tz options ('infer' or spec IANA tz) in report interface.
    fxobs_0 = report_parameters.object_pairs[0]
    if isinstance(fxobs_0, datamodel.ForecastObservation):
        timezone = fxobs_0.observation.site.timezone
    else:
        timezone = fxobs_0.aggregate.timezone
    return timezone


def create_raw_report_from_data(report, data):
    """
    Create a raw report using data and report metadata.

    Parameters
    ----------
    report : :py:class:`solarforecastarbiter.datamodel.Report`
        Metadata describing report
    data : dict
        Keys are all Forecast and Observation (or Aggregate)
        objects in the report, values are the corresponding data.

    Returns
    -------
    raw_report : :py:class:`solarforecastarbiterdatamodel.RawReport`

    Todo
    ----
    * add reference forecast
    """
    generated_at = pd.Timestamp.now(tz='UTC')
    report_params = report.report_parameters
    timezone = infer_timezone(report_params)
    versions = get_versions()
    with hijack_loggers([
        'solarforecastarbiter.metrics',
        'solarforecastarbiter.reports.figures.plotly_figures'],
                        ) as handler:
        # Validate, fill forecast, and resample
        processed_fxobs = preprocessing.process_forecast_observations(
            report_params.object_pairs,
            report_params.filters,
            report_params.forecast_fill_method,
            report_params.start, report_params.end,
            data, timezone,
            costs=report_params.costs)

        # Calculate metrics
        metrics_list = calculator.calculate_metrics(
            processed_fxobs,
            list(report_params.categories),
            list(report_params.metrics))
        summary_stats = calculator.calculate_all_summary_statistics(
            processed_fxobs, list(report_params.categories))

        report_plots = plotly_figures.raw_report_plots(report, metrics_list)
        messages = handler.export_records()
    raw_report = datamodel.RawReport(
        generated_at=generated_at, timezone=timezone, versions=versions,
        plots=report_plots, metrics=tuple(metrics_list + summary_stats),
        processed_forecasts_observations=tuple(processed_fxobs),
        messages=messages)
    return raw_report


def capture_report_failure(report_id, session):
    """
    Decorator factory to handle errors in report generation by
    posting a message in an empty RawReport along with a failed
    status to the API.

    Parameters
    ----------
    report_id: str
        ID of the report to update with the message and failed status
    session: :py:class:`solarforecastarbiter.io.api.APISession`
        Session object to connect to the API.

    Returns
    -------
    decorator
        Decorator to handle any errors in the decorated function.
        The decorator has an optional `err_msg` keyword argument
        to specify the error message if the wrapped function fails.
    """
    def decorator(f, *,
                  err_msg='Critical failure computing report'):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                out = f(*args, **kwargs)
            except Exception:
                msg = datamodel.ReportMessage(
                    message=err_msg,
                    step='solarforecastarbiter.reports.main',
                    level='CRITICAL',
                    function=str(f)
                )
                raw = datamodel.RawReport(
                    pd.Timestamp.now(tz='UTC'), 'UTC', (), None,
                    (), (), (msg,))
                session.post_raw_report(report_id, raw, 'failed')
                raise
            else:
                return out
        return wrapper
    return decorator


def compute_report(access_token, report_id, base_url=None):
    """
    Create a raw report using data from API. Typically called as a task.
    Failures will attempt to post a message for the failure in an
    empty RawReport to the API.

    Parameters
    ----------
    session : :py:class:`solarforecastarbiter.api.APISession`
        API session for getting and posting data
    report_id : str
        ID of the report to fetch from the API and generate the raw
        report for

    Returns
    -------
    raw_report : :py:class:`solarforecastarbiter.datamodel.RawReport`
    """
    session = APISession(access_token, base_url=base_url)
    fail_wrapper = capture_report_failure(report_id, session)
    report = fail_wrapper(session.get_report, err_msg=(
        'Failed to retrieve report. Perhaps the report does not exist, '
        'the user does not have permission, or the connection failed.')
    )(report_id)
    data = fail_wrapper(get_data_for_report, err_msg=(
        'Failed to retrieve data for report which may indicate a lack '
        'of permissions or that an object does not exist.')
    )(session, report)
    raw_report = fail_wrapper(create_raw_report_from_data, err_msg=(
        'Unhandled exception when computing report.')
    )(report, data)
    fail_wrapper(session.post_raw_report, err_msg=(
        'Computation of report completed, but failed to upload result to '
        'the API.')
    )(report.report_id, raw_report)
    return raw_report
