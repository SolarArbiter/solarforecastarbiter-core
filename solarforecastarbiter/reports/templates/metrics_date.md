
### <a name="date-analysis"></a>Date analysis

Metrics for each date of the analysis period are displayed in tables and figures below.

{% for figure in figures['date'] %}
  {{ figure | safe }}
{% endfor %}
