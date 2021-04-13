"""Summary statistics (observations and forecasts)."""

import numpy as np


def yes_fraction(data):
    non_nan_count = np.count_nonzero(~np.isnan(data))
    yes_count = np.nansum(data)
    return yes_count / non_nan_count


def no_fraction(data):
    return 1 - yes_fraction(data)


# Add new metrics to this map to map shorthand to function
_DETERMINISTIC_MAP = {
    'mean': (np.nanmean, 'Mean'),
    'min': (np.nanmin, 'Min'),
    'max': (np.nanmax, 'Max'),
    'median': (np.nanmedian, 'Median'),
    'std': (np.nanstd, 'Std.'),
}

_EVENT_MAP = {
    'yes_fraction': (yes_fraction, 'Yes Fraction'),
    'no_fraction': (no_fraction, 'No Fraction')
}
