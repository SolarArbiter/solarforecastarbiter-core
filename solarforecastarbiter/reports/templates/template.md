Report
======

This report was automatically generated using the Solar Forecast Arbiter.

Important metadata...

name = {{ name }}
start = {{ start }}
end = {{ end }}
report generation time = {{ now }}

warn if data checksum does not equal saved checksum

Data
----

{{ script_data | safe }}

{{ figures_timeseries | safe }}

This is a scatter plot

{{ figures_scatter | safe }}

Metrics
-------

{{ script_metrics | safe }}

Metrics are displayed in tables and figures below. Metrics may be downloaded
in csv format.

{{ tables | safe }}
{{ figures_bar_0 | safe }}
{{ figures_bar_1 | safe }}
{{ figures_bar_2 | safe }}

Versions
--------

{{ print_versions }}

Hash
----

Report hash
