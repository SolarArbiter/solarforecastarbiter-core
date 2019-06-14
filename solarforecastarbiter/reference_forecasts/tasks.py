import json
import logging


import pandas as pd


from solarforecastarbiter.reference_forecasts import main, models, utils


logger = logging.getLogger(__name__)


def process_forecast_groups(session, run_time, forecasts=None):
    if forecasts is None:
        forecasts = session.list_forecasts()

    df_vals = []
    for fx in forecasts:
        try:
            extra_parameters = json.loads(fx.extra_parameters)
        except json.JSONDecodeError:
            logger.warning(
                'Failed to decode extra_parameters for %s: %s as JSON',
                fx.name, fx.forecast_id)
            continue

        try:
            model = extra_parameters['model']
        except KeyError:
            if 'piggyback_on' in extra_parameters:
                logger.error(
                    'Forecast, %s: %s, has piggyback_on in extra_parameters'
                    ' but no model. Cannot make forecast.',
                    fx.name, fx.forecast_id)
            else:
                logger.debug(
                    'Not model found for %s:%s, no forecast will be made',
                    fx.name, fx.forecast_id)
            continue

        piggyback_on = extra_parameters.get('piggyback_on', fx.forecast_id)
        df_vals.append((fx.forecast_id, fx, piggyback_on, model))

    forecast_df = pd.DataFrame(
        df_vals, columns=['forecast_id', 'forecast', 'piggyback_on', 'model']
        ).set_index('forecast_id')
    for run_for, group in forecast_df.groupby('piggyback_on'):
        if not len(group.model.unique()) == 1:
            logger.warning(
                'Not all forecasts in group with %s have the same model',
                run_for)
        key_fx = group.loc[run_for].forecast
        model = getattr(models, group.loc[run_for].model)
        issue_time = utils.get_next_issue_time(key_fx, run_time)

        nwp_result = main.run_nwp(key_fx, model, run_time, issue_time)
        for fx_id, fx in group['forecast'].iteritems():
            fx_vals = getattr(nwp_result, fx.variable)
            if fx_vals is None:
                continue
            session.post_forecast_values(fx_id, fx_vals)
