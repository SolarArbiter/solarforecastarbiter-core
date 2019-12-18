"""Summary statistics (observations and forecasts)."""

import numpy as np


# Add new metrics to this map to map shorthand to function
_MAP = {
    'mean': (np.mean, 'Mean'),
    'min': (np.min, 'Min'),
    'max': (np.max, 'Max'),
    'std': (np.std, 'Std'),
    'median': (np.median, 'Median'),
    'var': (np.var, 'Variance'),
}

__all__ = [m[0].__name__ for m in _MAP.values()]
