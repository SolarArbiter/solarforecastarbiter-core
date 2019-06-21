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
from solarforecastarbiter import metrics
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


def data_dict_to_fxobs_data(data, forecast_observations):
    """
    Sorts the data dict into a new dict where the keys are the report's
    ForecastObservation objects and the values are tuples of
    (forecast values, observation values).
    """
    return {fxobs: (data[fxobs.forecast.forecast_id],
                    data[fxobs.observation.observation_id])
            for fxobs in forecast_observations}


def create_metadata(report):
    """
    Create metadata for the raw report.

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
    # this is a dictionary of just the raw timeseries
    data = get_data_for_report(session, report)
    # validates data, resamples, and aligns
    fxobs_resamp, data_resamp = metrics.validate_resample_align(report, data)
    # so instead, get report values.
    # from prereport, get mapping from fx-obs to report_values ids
    fx_obs_cds = [(
        fxobs,
        figures.construct_fx_obs_cds(data_resamp[fxobs.forecast],
                                     data_resamp[fxobs.observation]))
                  for fxobs in fxobs_resamp]
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

    metadata = create_metadata(report)

    fxobs_resampled, data_resampled = metrics.validate_resample_align(report,
                                                                      data)

    # needs to be in json
    metrics_list = metrics.loop_forecasts_calculate_metrics(fxobs_resampled,
                                                            data_resampled)

    # put the metrics in the metadata because why not
    metadata['metrics'] = metrics_list

    prereport = template.prereport(report, metadata, metrics_list)

    # fails because
    # "TypeError: Object of type Timestamp is not JSON serializable"
    # metadata_json = json.dumps(metadata)
    metadata_json = metadata
    # package metadata and prereport template together in one dict
    # return prereport and metrics and resampled data
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
    # session.post_report(metadata, prereport)
    return metadata, prereport


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
    # should instead read prereport to get data
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
