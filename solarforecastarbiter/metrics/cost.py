import numpy as np
import pandas as pd


def _np_agg_fnc(agg_str, net):
    fnc = AGG_OPTIONS[agg_str]
    if net:
        return lambda x: fnc(x)
    else:
        return lambda x: fnc(np.abs(x))


def constant_cost_wrapper(cost_params):
    """Wrapper to generate cost function appropriate for calling
    in the loop of metrics.calculator.calculate_deterministic_metrics
    """
    cost_const = cost_params.cost
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        return agg_fnc(error) * cost_const

    return cost_func


def _make_time_of_day_cost(times, costs, index, tz, fill):
    dates = np.unique(index.date)
    prod = [(pd.Timestamp.combine(x, y[0]), y[1])
            for x in dates for y in zip(times, costs)]
    base_ser = pd.DataFrame(
        prod, columns=['timestamp', 'cost']
    ).set_index('timestamp')['cost'].tz_localize(tz).sort_index()
    ser = base_ser.reindex(index, method=fill)
    return ser


def time_of_day_cost_wrapper(cost_params):
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)
    fill = FILL_OPTIONS[cost_params.fill]

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        tz = cost_params.timezone or error.tzinfo
        cost_ser = _make_time_of_day_cost(
            cost_params.times, cost_params.costs, error.index, tz, fill)
        error_cost = error * cost_ser
        return agg_fnc(error_cost)


def datetime_cost_wrapper(cost_params):
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)
    fill = FILL_OPTIONS[cost_params.fill]
    cost_ser = pd.Series(cost_params.costs,
                         index=cost_params.datetimes)
    if cost_params.timezone is not None and cost_ser.tzinfo is None:
        cost_ser = cost_ser.tz_localize(cost_params.timezone)

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        if cost_ser.tzinfo is None:
            cs = cost_ser.tz_localize(error.tzinfo)
        else:
            cs = cost_ser
        error_cost = error * cs.reindex(error.index, method=fill)
        return agg_fnc(error_cost)


def _band_masks(bands, error):
    """"""
    prev = np.zeros(error.shape, dtype=bool)
    out = []
    for band in bands:
        emin, emax = band.error_range
        new = (error >= emin) & (error <= emax)
        # only those new locations that not also in prev should be used
        both = prev & new
        new[both] = False
        out.append(~new)
        prev |= new
    return out


def error_band_cost_wrapper(cost_params):
    bands = cost_params.bands
    band_cost_functions = [
        COST_FUNCTION_MAP[band.cost_function](
            band.cost_function_parameters)
        for band in bands
    ]

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        out = 0
        masks = _band_masks(bands, error)
        for mask, fnc in zip(masks, band_cost_functions):
            if mask.all():
                continue
            mobs = np.ma.MaskedArray(obs, mask=mask)
            mfx = np.ma.MaskedArray(fx, mask=mask)
            out += fnc(mobs, mfx, error_fnc)
        return out
    return cost_func


def generate_cost_function(cost_params):
    return COST_FUNCTION_MAP[cost_params.type](cost_params.parameters)


COST_FUNCTION_MAP = {
    'constant': constant_cost_wrapper,
    'time_of_day': time_of_day_cost_wrapper,
    'datetime': datetime_cost_wrapper,
    'errorband': error_band_cost_wrapper,
}

FILL_OPTIONS = {
    'forward': 'ffill',
    'backward': 'bfill'
}

AGG_OPTIONS = {
    'sum': np.sum,
    'mean': np.mean
}
