{{ name }}
========================

{# fix this and the /single hard code #}
{% set dash_url = 'https://dev-dashboard.solarforecastarbiter.org' %}

<div id='download'>
Download as [html]({{ dash_url|safe }}/reports/download/{{ html_link|safe }}) or [pdf]({{ dash_url|safe }}/reports/download/{{ pdf_link|safe }})
</div>

This report of solar forecast accuracy was automatically generated using the [Solar Forecast Arbiter](https://solarforecastarbiter.org).

This report analyzes the following observation, forecast pairs:

<ul>
{% for obs, obsid, fx, fxid in fx_obs %}
<li>Observation: <a href="{{ dash_url|safe }}/observations/{{ obsid|safe }}">{{ obs|safe }}</a>, Forecast: <a href="{{ dash_url|safe }}/forecasts/single/{{ fxid|safe }}">{{ fx|safe }}</a></li>
{% endfor %}
</ul>

No resampling was applied to observations or the forecasts.

This report covers the period from {{ start }} to {{ end }}. This report was generated at {{ now }}.

Data
----

{% if checksum_failure %}
WARNING: One or more of the observation or forecast data has changed since this report was created. Consider creating a new report.
{% endif %}

The plot below shows the time series of observation and forecast data.
{# if html   commented out because it doesn't work yet - something about the 2 step rendering #}
Controls to pan, zoom, and save the plot are shown on the right. Clicking on an item in the legend will hide/show it.
{# endif #}

{{ script_data | safe }}

{{ figures_timeseries | safe }}

The scatter plot below shows forecast values vs observed values.

{{ figures_scatter | safe }}

The data validation toolkit identified the following issues with the data:

<ul>
{% for issue in validation_issues %}
<li>{{ issue.name }}: {{ issue.points }} intervals</li>
{% endfor %}
<li>Test Flag: 5 intervals</li>
</ul>

Metrics
-------

{{ script_metrics | safe }}

Metrics are displayed in tables and figures below. Metrics may be downloaded
in csv format.

{{ tables | safe }}

<div class='figures_bar'>
{% for figure in figures_bar %}
    {{ figure | safe }}
{% endfor %}
</div>

Versions
--------

{% for package, version in versions.items() %}
    {{ package }}: {{ version }}
{% endfor %}

Hash
----

The report signature is: a46d9d6e1fbd85b1023a95835a09f5f42491cf5a

The signature can be verified using the Solar Forecast Arbiter [public key](solarforecastarbiter.org).
