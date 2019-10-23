
### Monthly analysis

Metrics for each month of the analysis period are displayed in tables and figures below. Starting with January (1) up to December (12).

{% for figure in figures['month'] %}
  {{ figure | safe }}
{% endfor %}
