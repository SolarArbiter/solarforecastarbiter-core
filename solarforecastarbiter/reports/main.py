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


def create_report_from_metadata(session, metadata):
    """
    API access needed.
    """
    raise NotImplementedError

    # parse metadata
    # get data from API
    report = create_report_from_data()
    # post report to API


def create_report_from_data(metadata, data):
    """
    No API access needed
    """
    raise NotImplementedError

    # call function: align obs and forecasts
    # call function: loop through metrics
    # call function: format metrics into JSON
    # call function: add some metadata to JSON
    # return formatted metrics
