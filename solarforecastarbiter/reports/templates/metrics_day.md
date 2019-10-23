
### Daily analysis

Metrics for each day of the analysis period are displayed in tables and figures below. Each day is represented by an increasing integer.

{% for figure in figures['day'] %}
  {{ figure | safe }}
{% endfor %}
