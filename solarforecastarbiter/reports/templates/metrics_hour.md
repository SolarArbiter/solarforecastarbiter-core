
### <a name="hour-analysis"></a>Hour of the day analysis

Metrics for each hour (0-23) of the analysis period are displayed in tables and figures below.

{% for figure in figures['hour'] %}
  {{ figure | safe }}
{% endfor %}
