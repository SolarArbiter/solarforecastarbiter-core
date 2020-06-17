from solarforecastarbiter.datamodel import CostParameters, ErrorBandCost, ConstantCost, CostBand
import numpy as np


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


def constant_cost_wrapper(cost_params):
    cost_const = cost_params.cost
    agg_fnc = _np_agg_fnc(cost_params.aggregation, cost_params.net)

    def cost_func(obs, fx, error_fnc):
        error = error_fnc(obs, fx)
        return agg_fnc(error) * cost_const

    return cost_func


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
