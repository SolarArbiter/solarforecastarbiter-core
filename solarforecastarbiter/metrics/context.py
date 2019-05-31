"""
Provides a dictionary that defines the context to the Metrics' Evaluator,
which will control:
1. what metrics are calculated ('metrics' key)
2. any preprocessing of the input timeseries ('preprocessing' key)
3. any additional summary information of the evaluation ('results' key)

It also provides the default contexts for different Use Cases.
"""

import copy

from solarforecastarbiter.metrics import preprocessing, deterministic


_FILL_FUNCTIONS_MAP = {
    'exclude': preprocessing.exclude,
    'static': preprocessing.fill_static_value,
    'fill_forward': preprocessing.fill_forward,
    'linear_interpolate': preprocessing.fill_by_interpolation
}


_METRICS_MAP = {
    'mae': deterministic.mean_absolute,
    'mbe': deterministic.mean_bias,
    'rmse': deterministic.root_mean_square
}


_PREPROCESSING_CONTEXT = {

    'observations': {
        'fill_value': None,          # observation fill value (for 'static')
        'max_fill_intervals': None,  # number of intervals allowed to fill
        'fill_method': None          # one of the accepted methods for fillin
    },

    'forecasts': {
        'fill_value': None,          # forecast fill value if (for 'static')
        'max_fill_intervals': None,  # number of intervals allowed to fill
        'fill_method': None          # one of the accepted methods for fillin
    },

    'use_obs_interval_length': False   # use the observation instead of the
                                       # forecast interval run_length

}

_RESULTS_CONTEXT = {

    # result will always include the metrics specified overall timeframes

    # Timeseries to include
    'timeseries': {
        'observations': False,  # processed observation timeseries
        'forecasts': False,     # processed forecast(s) timeseries
    },

    # additional groupings to report metrics by
    'groupings': {
        # 'season': False,  # TODO: determine consistent way to calculate
        'month': False,
        'dow': False,     # day of the week
        'hod': False      # hour of the day
    }
}

DETERMINISTIC_METRICS_CONTEXT = {

    'is_pv_power': False,
    'include_night_hours': False,

    'metrics': {
        'mean': False,       # mean
        'std': False,        # standard deviation
        'mae': False,        # mean absolute error
        'mape': False,       # mean absolute percentage error
        'mbe': False,        # mean bias error
        'rmse': False,       # root mean squared error
        'nrmse': False,      # normalized root mean squared error
        'crmse': False,      # centered (unbiased) root mean squared error
        'pearson_r': False,  # pearson correlation coeefficient (r)
        'r2_score': False,   # coefficient of determination
        'ksi': False,        # Kolmogorov-Smirnov integral (KSI)
        'over_ksi': False,   # modified KSI over a critical limit
        'cpi': False         # combined performance index (CPI)
    },

    'preprocessing': copy.deepcopy(_PREPROCESSING_CONTEXT),

    'results': copy.deepcopy(_RESULTS_CONTEXT)

}

EVENT_METRICS_CONTEXT = {

    'is_pv_power': False,
    'include_night_hours': False,

    'metrics': {
        'pod': False,             # probability of detection
        'far': False,             # false alarm ratio
        'pofd': False,            # probability of false detection
        'csi': False,             # critical success index
        'event_bias': False,      # event bias
        'accuracy_score': False   # ratio that event was forecast correctly
    },

    'preprocessing': _PREPROCESSING_CONTEXT,

    'results': _RESULTS_CONTEXT

}


PROBABILISTIC_METRICS_CONTEXT = {

    'is_pv_power': False,
    'include_night_hours': False,

    'metrics': {
        'brier_score': False,      # Brier score
        'brier_skill': False,      # Brier skill score
        'reliability': False,      # reliability
        'resolution': False,      # resolution
        'uncertainty': False,      # variance of the event indicator
        'sharpness': False,      # sharpness
        'crps': False       # continuous ranked probability score
    },

    'preprocessing': copy.deepcopy(_PREPROCESSING_CONTEXT),

    'results': copy.deepcopy(_RESULTS_CONTEXT)

}


def get_default_deterministic_context(is_pv_power=False,
                                      include_night_hours=False):
    """Get a default deterministic metrics context."""

    # blank context
    context = copy.deepcopy(DETERMINISTIC_METRICS_CONTEXT)

    # night hours
    context['include_night_hours'] = include_night_hours

    # pv power forecast
    context['is_pv_power'] = is_pv_power

    # set default metrics
    context_met = context['metrics']
    # context_met['mean'] = True
    # context_met['std'] = True
    context_met['mae'] = True
    context_met['mbe'] = True
    if is_pv_power:
        context_met['nrmse'] = True
    else:
        context_met['rmse'] = True

    # set default fill
    context_obs = context['preprocessing']['observations']
    context_obs['fill_method'] = 'exclude'

    context_for = context['preprocessing']['forecasts']
    context_for['fill_method'] = 'exclude'

    context['use_obs_interval_length'] = False

    # set default results
    context_res_ser = context['results']['timeseries']
    context_res_ser['observations'] = True
    context_res_ser['forecasts'] = True

    context_res_grp = context['results']['groupings']
    context_res_grp['hod'] = True

    return context


def supported_groupings():
    """Returns a list of supported groupings."""
    return _RESULTS_CONTEXT['groupings'].keys()


def supported_fill_functions():
    """Returns a list of supported fill functions."""
    return _FILL_FUNCTIONS_MAP.keys()


def supported_metrics():
    """Returns a list of supported metrics."""
    return _METRICS_MAP.keys()


def get_default_event_context():
    """Get a default events metrics context."""
    raise NotImplementedError


def get_default_probabilistic_context():
    """Get a default probabilistic metrics context."""
    raise NotImplementedError


def print_context_string(context):
    """Pretty print context to string."""
    raise NotImplementedError


def show_context(context):
    """Print context to screen."""
    print(print_context_string(context))
