"""
Metric module specific errors.
"""

class SfaMetricsError(Exception):
    """Base class for Solar Forecast Arbiter Metrics Errors."""
    pass

class SfaMetricsConfigError(SfaMetricsError):
    """Error thrown when the metrics context is configured incorrectly."""
    pass

class SfaMetricsInputEror(SfaMetricsError):
    """Error thrown when the input provided is incorrect."""
    pass