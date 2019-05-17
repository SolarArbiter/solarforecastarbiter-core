"""
Make a report.

Steps:

  1. Consume metadata defined in :py:mod:`~solarforecastarbiter.datamodel`
  2. Run validation of metadata. Metadata creation might also include
     some validation so this may not be necessary.
  3. Get data
  4. Align observation data to forecast data. Is this a metrics function?
  5. Compute metrics specified in metadata.
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


Questions:

* Does core create the html and pdf versions of the report?
* Does core submit pre-rendered html and pdf version of the report to
  the API?


Considerations:

* API uses queue system to initiate report generation
* Functions should not require an API session unless they really need it.
* The bokeh plots in the html version should probably be rendered at
  client load time. The metrics data will be immediately available, but
  the API will need to call the data query and realignment functions
  to be able to create time series, scatter, etc. plots.
"""

from solarforecastarbiter.io.api import APISession


def get_data_for_report(session, report):
    """
    Get data for report.

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


def create_report_from_data(report, data):
    """
    Create a report using data and report metadata.

    Parameters
    ----------
    report : solarforecastarbiter.datamodel.Report
        Metadata describing report
    data : dict
        Keys are all Forecast and Observation objects in the report,
        values are the corresponding data.

    Returns
    -------
    results : str
        Report results in JSON format.
    """
    raise NotImplementedError

    # call function: align obs and forecasts
    # call function: loop through metrics
    # call function: format metrics into JSON
    # call function: add some metadata to JSON
    # return formatted metrics


def create_report_from_metadata(access_token, report, base_url=None):
    """
    Create a report using data from API and report metadata.

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
    report = create_report_from_data(report, data)
    session.post_report(report)
