
### Hourly analysis

Metrics for each hour (0-23) of the analysis period are displayed in tables and figures below.

{% for figure in figures['Hour of the day'] %}
  {{ figure | safe }}
{% endfor %}
