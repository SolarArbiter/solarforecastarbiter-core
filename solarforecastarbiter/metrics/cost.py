import numpy as np
import pandas as pd


def _np_agg_fnc(agg_str, net):
    if agg_str == 'mean':
        if net:
            return lambda x: np.mean(x)
        else:
            return lambda x: np.mean(np.abs(x))
    elif agg_str == 'sum':
        if net:
            return lambda x: np.sum(x)
        else:
            return lambda x: np.sum(np.abs(x))
    else:
        raise ValueError()


def _fill_kwarg(fill_str):
    if fill_str == 'forward':
        return 'ffill'
    elif fill_str == 'backward':
        return 'bfill'


def constant_cost_wrapper(cost_params):
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
    fill = _fill_kwarg(cost_params.fill)

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        tz = cost_params.timezone or error.index.tzinfo
        cost_ser = _make_time_of_day_cost(
            cost_params.times, cost_params.costs, error.index, tz, fill)
        error_cost = error * cost_ser
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
    'errorband': error_band_cost_wrapper,
}
