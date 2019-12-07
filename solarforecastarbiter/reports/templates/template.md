# {{ name }}


{#- this document is designed to be rendered in 3 steps #}
{#- 1. jinja renders the "prereport" - a markdown file with bokeh html/js tables and metrics graphics #}
{#- 2. jinja renders the "full report" - a markdown file with bokeh html/js with the above plus timeseries and scatter plots #}
{#- 3. pandoc renders the html or pdf version of the full report #}

This report of solar forecast accuracy was automatically generated using the [Solar Forecast Arbiter](https://solarforecastarbiter.org).

{{ '::: {.download}' }}
{# Download as [html]({{ dash_url|safe }}/reports/download/{{ html_link|safe }}) or pdf]({{ dash_url|safe }}/reports/download/{{ pdf_link|safe }}) #}
Download as [html]() or [pdf]()
:::

Please see our GitHub repository for [known issues](https://github.com/SolarArbiter/solarforecastarbiter-core/issues?q=is%3Aissue+is%3Aopen+label%3Areports) with the reports or to create a new issue.

Contents:

* [Report metadata](#report-metadata)
* [Data](#data)
  * [Observations and forecasts](#observations-and-forecasts)
  * [Data validation](#data-validation)
* [Metrics](#metrics)
{%- for met_key, met_val in metrics_toc.items() %}
  {%- if met_key in figures.keys() %}
  * [{{met_val}} analysis](#{{met_key}}-analysis)
  {%- endif %}
{%- endfor %}
* [Versions](#versions)

## Report metadata

{#- jinja requires that we escape the markdown div specification #}
{{ '::: {.metadata-table}' }}

* Name: {{ name }}
* Start: {{ start }}
* End: {{ end }}
* Generated at: {{ now }}
:::

## Data

{#- replace with warning if not all forecast/obs data is accessible #}
{%- if checksum_failure %}
{{ '::: warning' }}
WARNING: One or more of the observation or forecast data has changed since this report was created. Consider creating a new report.
:::
{%- endif %}

This report includes forecast and observation data available from {{ start }} to {{ end }}.

### Observations and forecasts

The table below shows the observation, forecast pairs analyzed in this report. The table includes the unprocessed observation and forecast *interval label* (beginning, ending, instantaneous) and *interval length*. If these quantities do not match, the Solar Forecast Arbiter must align and/or resample the data before computing error statistics. The Solar Forecast Arbiter typically aligns the observation data to the forecast data. The aligned and resampled parameters are also shown below.

{# Need to get report's realignment and resampling info into here. Need some nice formatting to make this table more readable. Also fix forecast link's hard coded /single route. column widths controlled by |--|--| line #}

| Observations | | | | | Forecasts | | | | |
|:--------|---|---|---|---|:--------|---|---|---|---|
Name|Interval label|Interval length|Aligned interval label|Resampled interval length|Name|Interval label|Interval length|Aligned interval label|Resampled interval length
{% for fx_ob, route, id in proc_fx_obs -%}
[{{ fx_ob.original.data_object.name|safe }}](/{{ route|safe }}/{{ id|safe }}) | {{ fx_ob.original.data_object.interval_label | safe}} | {{ (fx_ob.original.data_object.interval_length.total_seconds()/60)|int|safe }} min | {{ fx_ob.interval_label|safe }} | {{ (fx_ob.interval_length.total_seconds()/60)|int|safe }} min | [{{ fx_ob.original.forecast.name|safe }}](/forecasts/single/{{ fx_ob.original.forecast.forecast_id|safe }}) | {{ fx_ob.original.forecast.interval_label|safe }} | {{ (fx_ob.original.forecast.interval_length.total_seconds()/60)|int|safe }} min | {{ fx_ob.interval_label|safe }} | {{ (fx_ob.interval_length.total_seconds()/60)|int|safe }} min
{% endfor %}

The plots below show the realigned and resampled time series of observation and forecast data as well as a scatter plot of forecast vs observation data.
{# if html   commented out because it doesn't work yet - something about the 2 step rendering #}
Controls to pan, zoom, and save the plot are shown on the right. Clicking on an item in the legend will hide/show it.
{# endif #}

{{ script_data | safe }}
{{ figures_timeseries_scatter | safe }}

### Data validation

The Solar Forecast Arbiter's data validation toolkit identified the following issues with the unprocessed observation data:

{% for issue, points in validation_issues.items() %}
* {{ issue }}: {{ points }} intervals
{% endfor %}

These intervals were removed from the raw time series before resampling and realignment. For more details on the data validation results for each observation, please see the observation page linked to in the table above. Data providers may elect to reupload data to fix issues identified by the validation toolkit. The metrics computed in this report will remain unchanged, however, a user may generate a new report after the data provider submits new data. The online version of this report verifies that the data was not modified after the metrics were computed.

## Metrics

{{- '{%' }} raw {{ '%}' }}
{{ script_metrics | safe }}
{{- '{%' }} endraw {{ '%}' }}

Metrics are displayed in tables and figures below for one or more time periods. Metrics may be downloaded in csv format.

{#- Loop through each metric as keys in figures_bar by calling the markdown #}

{% for met_key in metrics_toc.keys() -%}
{%- if met_key in figures.keys() %}
{% include 'metrics_' + met_key + '.md' %}
{%- endif %}
{% endfor %}

## Versions

This report was created using open source software packages. The relevant packages and their versions are listed below. Readers are encouraged to study the source code to understand exactly how the data was processed.

| Package | Version |
|:--------|:--------|
{% for package, version in versions.items() -%}
    | {{ package|e }} | {{ version|e }} |
{% endfor %}
