### <a name="month-analysis"></a>Month of the year analysis

Metrics for each month of the analysis period are displayed in tables and figures below.

{% for figure in figures['month'] -%}
  {{ figure | safe }}
{%- endfor %}
