# {{ name }}

{# this document is designed to be rendered in 3 steps #}
{# 1. jinja renders the "prereport" - a markdown file with bokeh html/js tables and metrics graphics #}
{# 2. jinja renders the "full report" - a markdown file with bokeh html/js with the above plus timeseries and scatter plots #}
{# 3. pandoc renders the html or pdf version of the full report #}

{# fix this #}
{% set dash_url = 'https://dashboard.solarforecastarbiter.org' %}

{# jinja requires that we escape the markdown div specification #}
{{ '::: {.metadata-table}' }}

## Report metadata

* Name: {{ name }}
* Start: {{ start }}
* End: {{ end }}
:::

{{ '::: {.download}' }}
{# Download as [html]({{ dash_url|safe }}/reports/download/{{ html_link|safe }}) or pdf]({{ dash_url|safe }}/reports/download/{{ pdf_link|safe }}) #}
Download as [html]() or [pdf]() **TODO**
:::

This report of solar forecast accuracy was automatically generated using the [Solar Forecast Arbiter](https://solarforecastarbiter.org).

This report was generated at {{ now }}.

Contents:

* [Report metadata](#report-metadata)
* [Data](#data)
  * [Observations and forecasts](#observations-and-forecasts)
  * [Data validation](#data-validation)
* [Metrics](#metrics)
  * [Total analysis period](#total-analysis-period)
  * [Monthly](#monthly)
  * [Daily](#daily)
  * [Hourly](#hourly)
* [Versions](#versions)
* [Hash](#hash)

## Data

{# replace with warning if not all forecast/obs data is accessible #}
{% if checksum_failure %}
{{ '::: warning' }}
WARNING: One or more of the observation or forecast data has changed since this report was created. Consider creating a new report.
:::
{% endif %}

This report covers the period from {{ start }} to {{ end }}.

### Observations and forecasts

The table below shows the observation, forecast pairs analyzed in this report. The table includes the unprocessed observation and forecast *interval label* (beginning, ending, instantaneous) and *interval length*. If these quantities do not match, the Solar Forecast Arbiter must align and/or resample the data before computing error statistics. The Solar Forecast Arbiter typically aligns the observation data to the forecast data. The aligned and resampled parameters are also shown below.

{# Need to get report's realignment and resampling info into here. Need some nice formatting to make this table more readable. Also fix forecast link's hard coded /single route. column widths controlled by |--|--| line #}

| Observations | | | | | Forecasts | | | | |
|:--------|---|---|---|---|:--------|---|---|---|---|
Name|Interval label|Interval length|Aligned interval label|Resampled interval length|Name|Interval label|Interval length|Aligned interval label|Resampled interval length
{% for fx_ob in fx_obs -%}
[{{ fx_ob.observation.name|safe }}]({{ dash_url|safe }}/observations/{{ fx_ob.observation.observation_id|safe }}) | {{ fx_ob.observation.interval_label | safe}} | {{ (fx_ob.observation.interval_length.total_seconds()/60)|int|safe }} min | {{ fx_ob.observation.interval_label|safe }} | {{ (fx_ob.observation.interval_length.total_seconds()/60)|int|safe }} min | [{{ fx_ob.forecast.name|safe }}]({{ dash_url|safe }}/forecasts/single/{{ fx_ob.forecast.forecast_id|safe }}) | {{ fx_ob.forecast.interval_label|safe }} | {{ (fx_ob.forecast.interval_length.total_seconds()/60)|int|safe }} min | {{ fx_ob.forecast.interval_label|safe }} | {{ (fx_ob.forecast.interval_length.total_seconds()/60)|int|safe }} min
{% endfor %}

The plots below show the realigned and resampled time series of observation and forecast data as well as a scatter plot of forecast vs observation data.
{# if html   commented out because it doesn't work yet - something about the 2 step rendering #}
Controls to pan, zoom, and save the plot are shown on the right. Clicking on an item in the legend will hide/show it.
{# endif #}

{{ script_data | safe }}

{{ figures_timeseries_scatter | safe }}

### Data validation

The Solar Forecast Arbiter's data validation toolkit identified the following **(EXAMPLE)** issues with the unprocessed observation data:

{% for issue, points in validation_issues.items() %}
* {{ issue }}: {{ points }} intervals
{% endfor %}

These intervals were removed from the raw time series before resampling and realignment. For more details on the data validation results for each observation, please see the observation page linked to in the table above. Data providers may elect to reupload data to fix issues identified by the validation toolkit. The metrics computed in this report will remain unchanged, however, a user may generate a new report after the data provider submits new data. The online version of this report verifies that the data was not modified after the metrics were computed.

## Metrics

{{ '{%' }} raw {{ '%}' }}
{{ script_metrics | safe }}
{{ '{%' }} endraw {{ '%}' }}

Metrics are displayed in tables and figures below for one or more time periods. **(TODO)** Metrics may be downloaded
in csv format.

### Total analysis period

Metrics for the total analysis period are displayed in tables and figures below.

{{ tables | safe }}

{{ '::: {.figures_bar}' }}
{% for figure in figures_bar %}
    {{ figure | safe }}
{% endfor %}
:::

### Monthly

{# consider putting these in a jinja for loop. will also/alternatively need if statements in case the monthly/daily/hourly metrics were not part of the report specification #}

Metrics for each month of the analysis period are displayed in tables and figures below.

{{ '::: {.figures_bar}' }}
{% for figure in figures_bar_month %}
    {# each figure is for a different metric #}
    {# each figure is a stack of stack of short bar charts. one bar chart for each forecast #}
    {# consider putting the figures in collapsable divs with only the first one open #}
    {{ figure | safe }}
{% endfor %}
:::

### Daily

Metrics for each day of the analysis period are displayed in tables and figures below.

{{ '::: {.figures_bar}' }}
{% for figure in figures_bar_day %}
    {# each figure is for a different metric #}
    {# each figure is a stack of stack of short bar charts. one bar chart for each forecast #}
    {# consider putting the figures in collapsable divs with only the first one open #}
    {{ figure | safe }}
{% endfor %}
:::

### Hourly

Metrics for each hour of the day during the analysis period are displayed in tables and figures below.

{{ '::: {.figures_bar}' }}
{% for figure in figures_bar_hour %}
    {# each figure is for a different metric #}
    {# each figure is a stack of stack of short bar charts. one bar chart for each forecast #}
    {# consider putting the figures in collapsable divs with only the first one open #}
    {{ figure | safe }}
{% endfor %}
:::

## Versions

This report was created using open source software packages. The relevant packages and their versions are listed below. Readers are encouraged to study the source code to understand exactly how the data was processed.

{% for package, version in versions.items() %}
    {{ package }}: {{ version }}
{% endfor %}

## Hash

{# fix this #}
The report signature is: a46d9d6e1fbd85b1023a95835a09f5f42491cf5a **(EXAMPLE ONLY)**

The signature can be verified using the Solar Forecast Arbiter [public key](solarforecastarbiter.org).
