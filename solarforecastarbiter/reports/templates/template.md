{{ name }}
========================

{# fix this #}
{% set dash_url = 'https://dev-dashboard.solarforecastarbiter.org' %}

<div id='download'>
Download as [html]({{ dash_url|safe }}/reports/download/{{ html_link|safe }}) or [pdf]({{ dash_url|safe }}/reports/download/{{ pdf_link|safe }})
</div>

This report of solar forecast accuracy was automatically generated using the [Solar Forecast Arbiter](https://solarforecastarbiter.org).

This report was generated at {{ now }}.

Data
----

{# fix this so it is evaluated at report render time #}
{% if checksum_failure %}
<div class="warning">WARNING: One or more of the observation or forecast data has changed since this report was created. Consider creating a new report.</div>
{% endif %}

This report covers the period from {{ start }} to {{ end }}.

The table below shows the observation, forecast pairs analyzed in this report. The table includes the unprocessed observation and forecast *interval label* (beginning, ending, instanteous) and *interval length*. If these quantities do not match, the Solar Forecast Arbiter must align and/or resample the data before computing error statistics. The Solar Forecast Arbiter typically aligns the observation data to the forecast data. The aligned and resampled parameters are also shown below.

{# reformat into table with information described above. Left side should be block titled Observations with columns Name, Interval label, Interval length, Aligned interval label, Resampled interval length. Right side should be block titled Forecasts with same columns. Need some nice formatting to make it readable. Also fix forecast link's hard coded /single route #}
<ul>
{% for obs, obsid, fx, fxid in fx_obs %}
<li>Observation: <a href="{{ dash_url|safe }}/observations/{{ obsid|safe }}">{{ obs|safe }}</a>, Forecast: <a href="{{ dash_url|safe }}/forecasts/single/{{ fxid|safe }}">{{ fx|safe }}</a></li>
{% endfor %}
</ul>

The plot below shows the realigned and resampled time series of observation and forecast data.
{# if html   commented out because it doesn't work yet - something about the 2 step rendering #}
Controls to pan, zoom, and save the plot are shown on the right. Clicking on an item in the legend will hide/show it.
{# endif #}

{{ script_data | safe }}

{{ figures_timeseries | safe }}

The scatter plot below shows realigned and resampled forecast vs observed values.

{{ figures_scatter | safe }}

Data validation
~~~~~~~~~~~~~~~

The Solar Forecast Arbiter's data validation toolkit identified the following issues with the unprocessed observation data:

<ul>
{% for issue in validation_issues %}
<li>{{ issue.name }}: {{ issue.points }} intervals</li>
{% endfor %}
<li>Test Flag: 5 intervals</li>
</ul>

These intervals were removed from the raw time series before resampling and realignment. RULES FOR RESAMPLING WITH BAD DATA. For more details on the data validation results for each observation, please see the observation page linked to in the table above. Data providers may elect to reupload data to fix issues identified by the validation toolkit. The metrics computed in this report will remain unchanged, however, a user may generate a new report after the data provider submits new data. The online version of this report verifies that the data was not modified after the metrics were computed.

Metrics
-------

{{ script_metrics | safe }}

Metrics are displayed in tables and figures below for one or more time periods. **(TODO)** Metrics may be downloaded
in csv format.

Total analysis period
~~~~~~~~~~~~~~~~~~~~~

Metrics for the total analysis period are displayed in tables and figures below.

{{ tables | safe }}

<div class='figures_bar'>
{% for figure in figures_bar %}
    {{ figure | safe }}
{% endfor %}
</div>

Monthly
~~~~~~~
{# consider putting these in a jinja for loop. will also/alternatively need if statements in case the monthly/daily/hourly metrics were not part of the report specification #}

Metrics for each month of the analysis period are displayed in tables and figures below.

<div class='figures_bar'>
{% for figure in figures_bar_monthly %}
    {# each figure is for a different metric #}
    {# each figure is a stack of stack of short bar charts. one bar chart for each forecast #}
    {# consider putting the figures in collapsable divs with only the first one open #}
    {{ figure | safe }}
{% endfor %}
</div>


Daily
~~~~~

Metrics for each day of the analysis period are displayed in tables and figures below.


Hourly
~~~~~~

Metrics for each hour of the day during the analysis period are displayed in tables and figures below.


Versions
--------

This report was created using open source software packages. The relevant packages and their versions are listed below. Readers are encouraged to study the source code to understand exactly how the data was processed.

{% for package, version in versions.items() %}
    {{ package }}: {{ version }}
{% endfor %}

Hash
----

{# fix this #}
The report signature is: a46d9d6e1fbd85b1023a95835a09f5f42491cf5a

The signature can be verified using the Solar Forecast Arbiter [public key](solarforecastarbiter.org).
