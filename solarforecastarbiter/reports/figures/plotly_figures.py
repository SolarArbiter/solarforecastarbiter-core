"""
Functions to make all of the metrics figures for Solar Forecast Arbiter reports
using Plotly.
"""
import base64
import calendar
from copy import deepcopy
import datetime as dt
from itertools import cycle
from pathlib import Path
import logging


import pandas as pd
from plotly import __version__ as plotly_version
import plotly.graph_objects as go
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


from solarforecastarbiter import datamodel
from solarforecastarbiter.metrics.event import _event2count
import solarforecastarbiter.plotting.utils as plot_utils


logger = logging.getLogger(__name__)
D3_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
              '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
              '#dbdb8d', '#9edae5']
PALETTE = (D3_PALETTE[::2] + D3_PALETTE[1::2])


def gen_grays(num_colors):
    """Generate a grayscale color list of length num_colors.
    """
    rgb_delta = int(255/num_colors + 1)
    color_list = ["#{h}{h}{h}".format(h=hex(i*rgb_delta)[2:])
                  for i in range(num_colors)]
    return color_list


_num_obs_colors = 3
# drop white
OBS_PALETTE = gen_grays(_num_obs_colors)
OBS_PALETTE.reverse()
OBS_PALETTE_TD_RANGE = pd.timedelta_range(
            freq='10min', end='60min', periods=_num_obs_colors)

# list of matplotlib's perceptually uniform sequential color pallettes
PROBABILISTIC_PALETTES = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

PLOT_BGCOLOR = '#FFF'
PLOT_MARGINS = {'l': 50, 'r': 50, 'b': 50, 't': 100, 'pad': 4}
PLOT_LAYOUT_DEFAULTS = {
    'autosize': True,
    'height': 250,
    'margin': PLOT_MARGINS,
    'plot_bgcolor': PLOT_BGCOLOR,
    'title_font_size': 16,
    'font': {'size': 14}
}

# Used to adjust plot height when many x axis labels or long labels  are
# present. The length of the longest label of the plot will be multiplies by
# this value and added o the height of PLOT_LAYOUT_DEFAULTS to determine the
# new height.
X_LABEL_HEIGHT_FACTOR = 11

# If for some reason, the fail.pdf (just a pdf with some text that
# pdf generation failed) is unavailable, use an empty pdf
try:
    with open(Path(__file__).parent / 'fail.pdf', 'rb') as f:
        fail_pdf = base64.a85encode(f.read()).decode()
except Exception:
    fail_pdf = ',u@!!/MSk8$73+IY58P_+>=pV@VQ644<Q:NASu.&BHT/T0Ha7#+<Vd[7VQ[\\ATAnH7VlLTAOL*>De*Dd5!B<pFE1r$D$kNX1K6%.6<uqiV.X\\GOIXKoa;)c"!&3^A=pehYA92j5ARTE_ASu$s@VQ6-+>=pV@VQ5m+<WEu$>"*cDdmGg1E\\@oDdmGg4?Ns74pkk=A8bpl$8N_X+E(_($9UEn03!49AKWX&@:s-o,p4oL+<Vd[:gnBUDKI!U+>=p9$6UH6026"gBjj>HGT^350H`%l0J5:A+>>E,2\'?03+<Vd[6Z6jaASuU2+>b2p+ArOh+<W=-Ec6)>+?Van+<VdL+<W=:H#R=;01U&$F`7[1+<VdL+>6Y902ut#DKBc*Eb0,uGmYZ:+<VdL01d:.Eckq#+<VdL+<W=);]m_]AThctAPu#b$6UH65!B;r+<W=8ATMd4Ear[%+>Y,o+ArP14pkk=A8bpl$8EYW+E(_($9UEn03!49AKWX&@:s.m$6UH602$"iF!+[01*A7n;BT6P+<Vd[6Z7*bF<E:F5!B<bDIdZpC\'ljA0Hb:CC\'m\'c+>6Q3De+!#ATAnA@ps(lD]gbe0fCX<+=LoFFDu:^0/$gDBl\\-)Ea`p#Bk)3:DfTJ>.1.1?+>6*&ART[pDf.sOFCcRC6om(W1,(C>0K1^?0ebFC/MK+20JFp_5!B<bDIdZpC\'lmB0Hb:CC\'m\'c+>6]>E+L.F6Xb(FCi<qn+<Vd[:gn!JF!*1[0Ha7#5!B<bDIdZpC\'o3+AS)9\'+?0]^0JG170JG170H`822)@*4AfqF70JG170JG:B0d&/(0JG1\'DBK9?0JG170JG4>0d&/(0JG1\'DBK9?0JG170JG4<0H`&\'0JG1\'DBK9?0JG170JG182\'=S,0JG1\'DBK9?0JG170JG493?U"00JG1\'DBK9?0JG170JG=?2BX\\-0JG1\'DBK9?0JG170JG@B1*A8)0JG1\'DBK:.Ea`ZuATA,?4<Q:UBmO>53!pcN+>6W2Dfd*\\+>=p9$6UH601g%nD]gq\\0Ha7#5!B<pFCB33G]IA-$8sUq$7-ue:IYZ'  # NOQA


def _value_frame_dict(idx, pfxobs, column=None):
    if column is None:
        forecast_values = pfxobs.forecast_values
    else:
        if pfxobs.forecast_values is not None:
            forecast_values = pfxobs.forecast_values[column]
        else:
            forecast_values = None
    value_frame_dict = {
        'pair_index': idx,
        'observation_values': pfxobs.observation_values,
        'forecast_values': forecast_values,
    }
    return value_frame_dict


def _meta_row_dict(idx, pfxobs, **kwargs):
    forecast_object = kwargs.pop('forecast_object', None)
    if forecast_object is None:
        forecast_object = pfxobs.original.forecast

    # Check for a case where we're adding metadata for a constant value, but
    # the pair contains a whole ProbabilisticForecast
    if (isinstance(forecast_object,
                   datamodel.ProbabilisticForecastConstantValue)
        and
        isinstance(pfxobs.original.forecast,
                   datamodel.ProbabilisticForecast)):
        distribution = str(hash((
            pfxobs.original.forecast,
            pfxobs.original.forecast.interval_length,
            pfxobs.original.forecast.interval_value_type,
            pfxobs.original.forecast.interval_label)))
    else:
        distribution = None
    try:
        axis = forecast_object.axis
    except AttributeError:
        axis = None
    try:
        constant_value = forecast_object.constant_value
    except AttributeError:
        constant_value = None
    meta = {
        'pair_index': idx,
        'observation_name': _obs_name(pfxobs.original),
        'forecast_name': _fx_name(
            forecast_object, pfxobs.original.data_object),
        'interval_label': pfxobs.interval_label,
        'interval_length': pfxobs.interval_length,
        'forecast_type': pfxobs.original.__class__.__name__,
        'axis': axis,
        'constant_value': constant_value,
        'observation_hash': str(hash((
            pfxobs.original.data_object,
            pfxobs.interval_length,
            pfxobs.interval_value_type,
            pfxobs.interval_label))),
        'forecast_hash': str(hash((
            forecast_object,
            pfxobs.interval_length,
            pfxobs.interval_value_type,
            pfxobs.interval_label))),
        'observation_color': _obs_color(
            pfxobs.interval_length),
        'distribution': distribution
    }
    meta.update(kwargs)
    return meta


def construct_timeseries_dataframe(report):
    """Construct two standardized Dataframes for the timeseries and scatter
    plot functions. One with timeseries data for all observations,
    aggregates, and forecasts in the report, and the other with
    associated metadata sharing a common `pair_index` key.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    data : pandas.DataFrame
        Keys are an integer `pair_index` for pairing values with the metadata
        in the metadata_cds, and two pandas.Series, `observation_values` and
        `forecast_values`.

    metadata : pandas.DataFrame
        This dataframe has the following columns:

        - `pair_index`: Integer for pairing metadata with the values in the data dataframe.
        - `observation_name`: Observation name.
        - `forecast_name`: Forecast name.
        - `interval_label`: Interval label of the processed forecast and observation data.
        - `observation_hash`: Hash of the original observation object and the `datamodel.ProcessedForecastObservations` metadata.
        - `forecast_hash`: Hash of the original forecast object and the `datamodel.ProcessedForecastObservations` metadata.

    """  # NOQA
    value_frames = []
    meta_rows = []
    # enumerate won't work because of the conditional for loop, so
    # manually keep track of the index
    idx = 0
    for pfxobs in report.raw_report.processed_forecasts_observations:
        if isinstance(pfxobs.original.forecast,
                      datamodel.ProbabilisticForecast):
            for cvfx in pfxobs.original.forecast.constant_values:
                value_frame_dict = _value_frame_dict(
                    idx, pfxobs, column=str(cvfx.constant_value))
                if value_frame_dict['forecast_values'] is None:
                    continue
                # specify fx type so we know the const value fx came from a
                # ProbabilisticForecast
                meta_row_dict = _meta_row_dict(
                    idx, pfxobs,
                    forecast_object=cvfx,
                    forecast_type='ProbabilisticForecast')
                value_frames.append(pd.DataFrame(value_frame_dict))
                meta_rows.append(meta_row_dict)
                idx += 1
        else:
            value_frame_dict = _value_frame_dict(idx, pfxobs)
            if value_frame_dict['forecast_values'] is None:
                continue
            meta_row_dict = _meta_row_dict(idx, pfxobs)
            value_frames.append(pd.DataFrame(value_frame_dict))
            meta_rows.append(meta_row_dict)
            idx += 1
    if value_frames:
        data = pd.concat(value_frames)
    else:
        data = pd.DataFrame()
    metadata = pd.DataFrame(meta_rows)
    # convert data to report timezone
    data = data.tz_convert(report.raw_report.timezone)
    data = data.rename_axis('timestamp')
    return data, metadata


def _fill_timeseries(df, interval_length):
    """Returns a dataframe with a datetimeindex with regular frequency of
    interval_length minutes. Previously missing values will be filled with
    nans. Useful for creating gaps in plotted timeseries data.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with timeseries data.
    interval_length: numpy.timedelta64
        Interval length of the processed forecast observation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with filled datetime index data.
    """
    if not df.index.empty:
        start = df.index[0]
        end = df.index[-1]
        freq_mins = int(interval_length / np.timedelta64(1, 'm'))
        filled_idx = pd.date_range(start, end, freq=f'{freq_mins}min')
        return df.reindex(filled_idx)
    else:
        return df


def _obs_name(fx_obs):
    # TODO: add code to ensure obs names are unique
    name = fx_obs.data_object.name
    if fx_obs.forecast.name == fx_obs.data_object.name:
        if isinstance(fx_obs.data_object, datamodel.Observation):
            name += ' Observation'
        else:
            name += ' Aggregate'
    return name


def _fx_name(forecast, data_object):
    # TODO: add code to ensure fx names are unique
    forecast_name = forecast.name
    if isinstance(forecast, datamodel.ProbabilisticForecastConstantValue):
        if forecast.axis == 'x':
            forecast_name += \
                f' Prob(x <= {forecast.constant_value} {forecast.units})'
        else:
            forecast_name += f' Prob(f <= x) = {forecast.constant_value}%'
    if forecast_name == data_object.name:
        forecast_name += ' Forecast'
    return forecast_name


def _obs_color(interval_length):
    idx = np.searchsorted(OBS_PALETTE_TD_RANGE, interval_length)
    obs_color = OBS_PALETTE[idx]
    return obs_color


def _boolean_filter_indices_by_pair(value_cds, pair_index):
    return value_cds.data['pair_index'] == pair_index


def _none_or_values0(metadata, key):
    value = metadata.get(key)
    if value is not None:
        value = value.values[0]
    return value


def _extract_metadata_from_df(metadata_df, hash_, hash_key):
    metadata = metadata_df[metadata_df[hash_key] == hash_]
    meta = {
        'pair_index': metadata['pair_index'].values[0],
        'observation_name': metadata['observation_name'].values[0],
        'forecast_name': metadata['forecast_name'].values[0],
        'interval_label': metadata['interval_label'].values[0],
        'interval_length': metadata['interval_length'].values[0],
        'observation_color': metadata['observation_color'].values[0],
    }
    meta['forecast_type'] = _none_or_values0(metadata, 'forecast_type')
    meta['axis'] = _none_or_values0(metadata, 'axis')
    meta['constant_value'] = _none_or_values0(metadata, 'constant_value')
    return meta


def _legend_text(name, max_length=20):
    """Inserts <br> tags in a name to mimic word-wrap behavior for long names
    in the legend of timeseries plots.

    Parameters
    ----------
    name: str
        The name/string to apply word-wrap effect to.
    max_length: int
        The maximum length of any line of text. Note that this will not break
        words across lines, but on the closest following space.

    Returns
    -------
    str
        The name after it is split appropriately.
    """
    if len(name) > max_length:
        temp = []
        new = []
        for part in name.split(' '):
            if len(' '.join(temp + [part])) > max_length:
                new.append(' '.join(temp))
                temp = [part]
            else:
                temp.append(part)
        if temp:
            new.append(' '.join(temp))
        return '<br>'.join(new)
    else:
        return name


def _plot_obs_timeseries(fig, timeseries_value_df, timeseries_meta_df):
    # construct graph objects in random hash order. collect them in a list
    # along with the pair index. Then add traces in order of pair index.
    gos = []
    # construct graph objects in random hash order
    for obs_hash in np.unique(timeseries_meta_df['observation_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, obs_hash, 'observation_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        plot_kwargs = plot_utils.line_or_step_plotly(
            metadata['interval_label'])
        data = _fill_timeseries(
            timeseries_value_df[pair_idcs],
            metadata['interval_length'],
        )
        if data['observation_values'].isnull().all():
            continue
        go_ = go.Scattergl(
            y=data['observation_values'],
            x=data.index,
            name=_legend_text(metadata['observation_name']),
            legendgroup=metadata['observation_name'],
            showlegend=True,
            marker=dict(color=metadata['observation_color']),
            connectgaps=False,
            **plot_kwargs)
        # collect in list
        gos.append((metadata['pair_index'], go_))
    # Add traces in order of pair index
    for idx, go_ in sorted(gos, key=lambda x: x[0]):
        fig.add_trace(go_)


def _plot_fx_timeseries(fig, timeseries_value_df, timeseries_meta_df, axis):
    palette = cycle(PALETTE)
    # pull metadata to plot in random hash order. collect them in a list
    # along with the pair index. Then add traces in order of pair index.
    metadatas = []

    # pull metadata to plot in random hash order
    for fx_hash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fx_hash, 'forecast_hash')
        if metadata['axis'] not in axis:
            # we're looking at a different kind of forecast than what we wanted
            # to plot
            continue
        # collect in list
        metadatas.append((metadata['pair_index'], metadata))

    for idx, metadata in sorted(metadatas, key=lambda x: x[0]):
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        # probably treat axis == None and axis == y separately in the future.
        # currently no need for a separate axis == x treatment either, so
        # removed an if statement on the axis.
        plot_kwargs = plot_utils.line_or_step_plotly(
            metadata['interval_label'])
        data = _fill_timeseries(
            timeseries_value_df[pair_idcs],
            metadata['interval_length'],
        )
        plot_kwargs['marker'] = dict(color=next(palette))
        go_ = go.Scattergl(
            y=data['forecast_values'],
            x=data.index,
            name=_legend_text(metadata['forecast_name']),
            legendgroup=metadata['forecast_name'],
            showlegend=True,
            connectgaps=False,
            **plot_kwargs)
        fig.add_trace(go_)


def _plot_fx_distribution_timeseries(
        fig, timeseries_value_df, timeseries_meta_df, axis):
    palette = cycle(PROBABILISTIC_PALETTES)
    gos = []

    for dist_hash in np.unique(timeseries_meta_df['distribution']):
        # indices to constant values in the metadata df
        cv_indices = timeseries_meta_df['distribution'] == dist_hash

        # sort constant values
        cv_metadata = timeseries_meta_df[cv_indices]
        cv_metadata = cv_metadata.sort_values('constant_value')
        cv_metadata = cv_metadata.reset_index()

        # Get a colormap for mapping fill colors
        color_map = cm.get_cmap(next(palette))
        color_scaler = cm.ScalarMappable(
            Normalize(vmin=0, vmax=1),
            color_map,
        )

        symmetric_percentiles = plot_utils.percentiles_are_symmetric(
            cv_metadata['constant_value'].tolist())
        # Plot confidence intervals
        for idx, cv in cv_metadata.iterrows():
            pair_idcs = timeseries_value_df['pair_index'] == cv['pair_index']
            data = _fill_timeseries(
                timeseries_value_df[pair_idcs],
                cv['interval_length'])

            # Fill missing data with 0 to avoid plotly bugs encountered with
            # go.Scatter fill and missing data.
            data = data.fillna(0)

            if idx == 0:
                # The first value will act as the lower bound for other values
                # to fill down to.
                fill = None
                showlegend = True
            else:
                fill = 'tonexty'
                showlegend = False

            # Split name of the distribution from the current constant value
            constant_label_index = cv['forecast_name'].find('Prob(') - 1
            fx_name = cv['forecast_name'][:constant_label_index]
            cv_label = cv['forecast_name'][constant_label_index:]

            if symmetric_percentiles:
                # Since plotly always fills below the line, for constants below
                # 50%, use the previous value to mimic fill upward behavior.
                # E.g. fill downward from 5% to 0% with the 100% interval.
                if cv['constant_value'] <= 50 and idx != 0:
                    fill_value = cv_metadata.iloc[idx - 1]['constant_value']
                else:
                    fill_value = cv['constant_value']

                # When constant values are symmetric, create intervals
                # centered around the 50th percentile
                fill_value = 2 * abs(fill_value - 50)
            else:
                # convert to complement percentile to invert shading, such that
                # bright colors appear at 0 and dark at 100 when plotted.
                fill_value = 100 - cv['constant_value']

            fill_color = plot_utils.distribution_fill_color(
                color_scaler, fill_value)

            plot_kwargs = plot_utils.line_or_step_plotly(cv['interval_label'])

            go_ = go.Scatter(
                x=data.index,
                y=data['forecast_values'],
                name=_legend_text(fx_name),
                hovertemplate=(
                    f'<b>{ cv_label }<br>'
                    '<b>Value<b>: %{y}<br>'
                    '<b>Time<b>: %{x}<br>'),
                connectgaps=False,
                mode='lines',
                fill=fill,
                showlegend=showlegend,
                legendgroup=cv['distribution'],
                fillcolor=fill_color,
                line=dict(
                    color=fill_color,
                ),
                **plot_kwargs,
            )

            # Add traces in order of pair index
            gos.append((cv['pair_index'], go_))
    for idx, go_ in sorted(gos, key=lambda x: x[0]):
        fig.add_trace(go_)


def timeseries(timeseries_value_df, timeseries_meta_df,
               start, end, units, axis, timezone='UTC'):
    """
    Timeseries plot of one or more forecasts and observations.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    start : pandas.Timestamp
        Report start time
    end : pandas.Timestamp
        Report end time
    axis : {(None,), ('x',), ('y',), (None, 'y')}
        Specifies the kinds of forecast to plot. None is appropriate for
        deterministic forecasts, 'x' for probabilistic forecasts with
        axis = 'x', and 'y' for probabilistic forecasts with
        axis = 'y'. Observations, deterministic forecasts, and
        probabilistic forecasts may all be plotted together if
        axis = (None, 'y'). Observations will not be plotted if
        axis = ('x',).
    timezone : str
        Timezone consistent with the data in the timeseries_metadata_df.

    Returns
    -------
    plotly.Figure
    """  # NOQA: E501
    # might want to make fig=None a kwarg and modify this line to
    # fig = fig if fig is not None else go.Figure()
    fig = go.Figure()

    if 'x' in axis:
        ylabel = 'Probability (%)'
    else:
        ylabel = f'Data ({units})'
        # adds observation traces to fig
        _plot_obs_timeseries(fig, timeseries_value_df, timeseries_meta_df)

    # add forecast traces that have correct axis to fig
    # get indices of probabilistic forecasts with axis y to create special
    # shaded distribution plots
    y_distribution_indices = (
        timeseries_meta_df['distribution'].notna()
        & (timeseries_meta_df['axis'] == 'y')
    )
    non_y_distribution_meta_df = timeseries_meta_df[~y_distribution_indices]
    distribution_meta_df = timeseries_meta_df[y_distribution_indices]

    _plot_fx_timeseries(
        fig, timeseries_value_df, non_y_distribution_meta_df, axis)
    _plot_fx_distribution_timeseries(
        fig, timeseries_value_df, distribution_meta_df, axis)

    fig.update_xaxes(title_text=f'Time ({timezone})', showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')
    fig.update_yaxes(title_text=ylabel, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     fixedrange=True)
    fig.update_layout(
        legend=dict(font=dict(size=10)),
    )
    return fig


def _get_scatter_limits(df):
    extremes = [np.nan]
    for kind in ('forecast_values', 'observation_values'):
        arr = np.asarray(df[kind]).astype(float)
        if len(arr) != 0:
            extremes.append(np.nanmin(arr))
            extremes.append(np.nanmax(arr))
    min_ = np.nanmin(extremes)
    if np.isnan(min_):
        min_ = -999
    max_ = np.nanmax(extremes)
    if np.isnan(max_):
        max_ = 999
    return min_, max_


def scatter(timeseries_value_df, timeseries_meta_df, units):
    """
    Adds Scatter plot traces of one or more forecasts and observations to
    the figure.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.

    Returns
    -------
    plotly.Figure
    """  # NOQA
    scatter_range = _get_scatter_limits(timeseries_value_df)

    palette = cycle(PALETTE)
    fig = go.Figure()
    # pull metadata to plot in random hash order. collect them in a list
    # along with the pair index. Then add traces in order of pair index.
    metadatas = []
    # accumulate labels and plot objects for manual legend
    for fxhash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fxhash, 'forecast_hash')
        if metadata['axis'] == 'x':
            # don't know how to represent probability forecasts on a
            # physical value vs. physical value plot.
            continue
        # collect in list
        metadatas.append((metadata['pair_index'], metadata))

    # plot in order of pair index
    for idx, metadata in sorted(metadatas, key=lambda x: x[0]):
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        data = timeseries_value_df[pair_idcs]

        if data['observation_values'].isnull().all():
            # observation values were not included, skip pair
            continue

        go_ = go.Scattergl(
            x=data['observation_values'],
            y=data['forecast_values'],
            name=_legend_text(metadata['forecast_name']),
            showlegend=True,
            legendgroup=metadata['forecast_name'],
            marker=dict(color=next(palette), opacity=0.25),
            mode='markers')
        fig.add_trace(go_)

    label = f'({units})'
    x_label = 'Observed ' + label
    y_label = 'Forecast ' + label
    nticks = 10
    fig.update_xaxes(title_text=x_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     range=scatter_range, nticks=nticks)
    fig.update_yaxes(title_text=y_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside',
                     range=scatter_range, nticks=nticks)
    return fig


def event_histogram(timeseries_value_df, timeseries_meta_df):
    """
    Adds histogram plot traces of the event outcomes of one or more event
    forecasts and observations to the figure.

    Parameters
    ----------
    timeseries_value_df: pandas.DataFrame
        DataFrame of timeseries data. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.
    timeseries_meta_df: pandas.DataFrame
        DataFrame of metadata for each Observation Forecast pair. See
        :py:func:`solarforecastarbiter.reports.figures.construct_timeseries_dataframe`
        for format.

    Returns
    -------
    plotly.Figure
    """  # NOQA

    fig = go.Figure()
    palette = cycle(PALETTE)
    # accumulate labels and plot objects for manual legend
    for fxhash in np.unique(timeseries_meta_df['forecast_hash']):
        metadata = _extract_metadata_from_df(
            timeseries_meta_df, fxhash, 'forecast_hash')
        pair_idcs = timeseries_value_df['pair_index'] == metadata['pair_index']
        data = timeseries_value_df[pair_idcs]

        if data['observation_values'].isnull().all():
            continue
        tp, fp, tn, fn = _event2count(data["observation_values"],
                                      data["forecast_values"])
        x = ["True Pos.", "False Pos.", "True Neg.", "False Neg."]
        y = [tp, fp, tn, fn]

        fig.add_trace(go.Bar(
            x=x,
            y=y,
            name=_legend_text(metadata['forecast_name']),
            showlegend=True,
            legendgroup=metadata['forecast_name'],
            marker_color=next(palette),
        ))

    # update axes
    x_label = "Outcome"
    y_label = "Count"
    fig.update_xaxes(title_text=x_label, showgrid=True,
                     gridwidth=0, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')
    fig.update_yaxes(title_text=y_label, showgrid=True,
                     gridwidth=1, gridcolor='#CCC', showline=True,
                     linewidth=1, linecolor='black', ticks='outside')

    return fig


def configure_axes(fig, x_axis_kwargs, y_axis_kwargs):
    """Applies plotly axes configuration to display zero line and grid, and the
    configuration passed in x_axis_kwargs and y_axis kwargs. Currently
    configured to supply base layout for metric plots.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure

    x_axis_kwargs: dict
        Dictionary to expand as arguments to fig.update_xaxes.
    y_axis_kwargs: dict
        Dictionary to expand as arguments to fig.update_x_axes.
    """
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                     ticks='outside')
    if x_axis_kwargs:
        fig.update_xaxes(**x_axis_kwargs)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#CCC')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                     ticks='outside')
    if y_axis_kwargs:
        fig.update_yaxes(**y_axis_kwargs)


def construct_metrics_dataframe(metrics, rename=None):
    """
    Possibly bad assumptions:
    * metrics contains keys: name, Total, etc.

    Parameters
    ----------
    metrics : list of datamodel.MetricResults
        Each metric dict is for a different forecast. Forecast name is
        specified by the name key.
    rename : function or None
        Function of one argument that is applied to each forecast name.

    Returns
    -------
    df: pandas.DataFrame
        Dataframe of computed metrics for the report.
    """

    if rename:
        f = rename
    else:
        def f(x): return x  # NOQA

    data = []
    for metric_result in metrics:
        for mvalue in metric_result.values:
            new = {
                'name': metric_result.name,
                'abbrev': f(metric_result.name),
                'category': mvalue.category,
                'metric': mvalue.metric,
                'value': mvalue.value
            }
            if new['category'] == 'date':
                new['index'] = dt.datetime.strptime(
                    mvalue.index, '%Y-%m-%d')
            else:
                new['index'] = mvalue.index
            data.append(new)
    df = pd.DataFrame(data, columns=[
        'name', 'abbrev', 'category', 'metric', 'value', 'index'
    ])
    return df


def abbreviate(x, limit=3):
    # might need to add logic to ensure uniqueness
    # and/or enforce max length using textwrap.shorten
    components = x.split(' ')
    out_components = []
    for c in components:
        if len(c) <= limit:
            out = c
        elif c.upper() == c:
            # probably an acronym
            out = c
        elif c == 'Prob(f' or c == 'Prob(x':
            # special case for probabilistic forecast labelling
            out = c
        else:
            out = f'{c[0:limit]}.'
        out_components.append(out)
    return ' '.join(out_components)


def bar(df, metric):
    """
    Create a bar graph comparing a single metric across forecasts.

    Parameters
    ----------
    df: pandas Dataframe
        Metric dataframe by :py:func:`solarforecastarbiter.reports.figures.construct_metrics_dataframe`
    metric: str
        The metric to plot. This value should be found in df['metric'].

    Returns
    -------
    plotly.Figure
        A bar chart representing the total category of the metric for each
        forecast.
    """  # NOQA
    data = df[(df['category'] == 'total') & (df['metric'] == metric)]
    y_range = None
    x_axis_kwargs = {}
    x_values = []

    # Ensure data aligns with the x labels by pre-sorting. x_labels are sorted
    # by the groupby process below.
    data = data.sort_values('abbrev')

    # to avoid stacking, add BOM characters to fx with
    # same abbreviated name. GH463
    for val, ser in data[['abbrev']].groupby('abbrev'):
        x_values += [val + ('\ufeff' * i) for i in range(len(ser))]
    x_values = pd.Series(x_values, name='abbrev')
    palette = cycle(PALETTE)
    palette = [next(palette) for _ in x_values]
    metric_name = datamodel.ALLOWED_METRICS[metric]

    # remove height limit when long abbreviations are used or there are more
    # than 5 pairs to problems with labels being cut off.
    plot_layout_args = deepcopy(PLOT_LAYOUT_DEFAULTS)
    # ok to cut off BOM characters at the end of the labels
    longest_x_label = x_values.map(lambda x: len(x.rstrip('\ufeff'))).max()
    if longest_x_label > 15 or x_values.size > 6:
        # Set explicit height and set automargin on x axis to allow for dynamic
        # sizing to accomodate long x axis labels. Height is set based on
        # length of longest x axis label, due to a failure that can occur when
        # plotly determines there is not enough space for automargins to work.
        plot_height = plot_layout_args['height'] + (
            longest_x_label * X_LABEL_HEIGHT_FACTOR)
        plot_layout_args['height'] = plot_height
        x_axis_kwargs = {'automargin': True}
        if longest_x_label > 60:
            x_axis_kwargs.update({'tickangle': 90})
        elif longest_x_label > 30:
            x_axis_kwargs.update({'tickangle': 45})

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_values, y=data['value'],
                         text=data['name'],
                         marker=go.bar.Marker(color=palette),
                         hovertemplate='(%{text}, %{y})<extra></extra>'))
    fig.update_layout(
        title=f'<b>{metric_name}</b>',
        xaxis_title=metric_name,
        **plot_layout_args)
    configure_axes(fig, x_axis_kwargs, y_range)
    return fig


def calc_y_start_end(y_min, y_max, pad_factor=1.03):
    """
    Determine y axis start, end.

    Parameters
    ----------
    y_min : float
    y_max : float
    pad_factor : float
        Number by which to multiply the start, end.

    Returns
    -------
    start, end : float, float
    """
    # limits cannot be nans or infs
    y_min = np.nan_to_num(y_min)
    y_max = np.nan_to_num(y_max)

    if y_max < 0:
        # all negative, so set range from y_min to 0
        start = y_min
        end = 0
    elif y_min > 0:
        # all positive, so set range from 0 to y_max
        start = 0
        end = y_max
    else:
        start = y_min
        end = y_max

    # if y_max or min was +/- inf then padding will result in overflow
    # that can be ignored
    with np.errstate(over='ignore'):
        start, end = pad_factor * start, pad_factor * end

    return start, end


def bar_subdivisions(df, category, metric):
    """
    Create bar graphs comparing a single metric across subdivisions of
    time for multiple forecasts. e.g.::

        Fx 1 MAE |
                 |_________________
        Fx 2 MAE |
                 |_________________
                   Year, Month of the year, etc.

    Parameters
    ----------
    df: pandas.DataFrame
        Fields must be kind and the names of the forecasts
    category : str
        One of the available metrics grouping categories (e.g., total)
    metric : str
        One of the available metrics (e.g. mae)

    Returns
    -------
    figs : dict of figures
    """
    palette = cycle(PALETTE)

    figs = {}

    human_category = datamodel.ALLOWED_CATEGORIES[category]
    metric_name = datamodel.ALLOWED_METRICS[metric]

    x_axis_label = human_category
    y_axis_label = metric_name

    data = df[(df['category'] == category) & (df['metric'] == metric)]

    x_offset = None

    # Special handling for x-axis with dates
    if category == 'weekday':
        x_ticks = calendar.day_abbr[0:]
        x_axis_kwargs = {'tickvals': x_ticks,
                         'range': (-.5, len(x_ticks))}
    elif category == 'hour':
        x_ticks = list(range(25))
        x_axis_kwargs = {'tickvals': x_ticks,
                         'range': (-.5, len(x_ticks))}
        # plotly's offset of 0, makes the bars left justified at the tick
        x_offset = 0
    elif category == 'year':
        x_axis_kwargs = {'dtick': 1}
    elif category == 'date':
        # Sets a '{month} {day}' tick label format when zoomed in to one week
        # of data. Plotly's default behavior at this zoom range is to display
        # date and time, which causes crowding. Ranges are defined in
        # miliseconds, with 604800000 being 7 days, and None being the absolute
        # minimum. When zoomed out beyond one week, Plotly's default behavior
        # takes over and intelligently displays day, month and year reducing to
        # month and year as the user zooms out further.
        x_axis_kwargs = {'tickformatstops': [
            dict(dtickrange=[None, 604800000], value='%b %e'),
            ]
        }
    else:
        x_axis_kwargs = {}

    y_data = np.asarray(data['value'])
    if len(y_data) == 0 or np.isnan(y_data).all():
        y_range = (None, None)
    else:
        y_min = np.nanmin(y_data)
        y_max = np.nanmax(y_data)
        y_range = calc_y_start_end(y_min, y_max)
    y_axis_kwargs = {'range': y_range}
    unique_names = np.unique(np.asarray(data['name']))
    palette = [next(palette) for _ in unique_names]
    for i, name in enumerate(unique_names):
        plot_data = data[data['name'] == name]
        if len(plot_data['index']):
            x_values = plot_data['index']
        else:
            x_values = []
        if category == 'weekday':
            # Fill with mon-fri values and pass to enforce displaying the full
            # week of data.
            y_values = [plot_data[plot_data['index'] == day]['value'].iloc[0]
                        if not plot_data[plot_data['index'] == day].empty
                        else np.nan for day in x_ticks]
            x_values = x_ticks
        else:
            y_values = plot_data['value']
        # Create figure
        title = name + ' ' + metric_name
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x_values, y=y_values, offset=x_offset,
                             marker=go.bar.Marker(color=palette[i])))

        fig.update_layout(
            title=f'<b>{title}</b>',
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            **PLOT_LAYOUT_DEFAULTS)
        configure_axes(fig, x_axis_kwargs, y_axis_kwargs)
        figs[name] = fig
    return figs


def nested_bar():
    raise NotImplementedError


def joint_distribution():
    raise NotImplementedError


def marginal_distribution():
    raise NotImplementedError


def taylor_diagram():
    raise NotImplementedError


def probabilistic_timeseries():
    raise NotImplementedError


def reliability_diagram():
    raise NotImplementedError


def rank_histogram():
    raise NotImplementedError


def output_svg(fig):
    """
    Generates an SVG from the Plotly figure. Errors in the process are logged
    and an SVG with error text is returned.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure

    Returns
    -------
    svg : str

    Notes
    -----
    Requires `Orca <https://plot.ly/python/orca-management/>`_ for generating
    svgs. If orca is not installed, an svg with an error message will be
    returned.
    """
    try:
        svg = fig.to_image(format='svg').decode('utf-8')
    except Exception:
        try:
            name = fig.layout.title['text'][3:-4]
        except Exception:
            name = 'unnamed'
        logger.error('Could not generate SVG for figure %s', name)
        svg = (
            '<svg width="100%" height="100%">'
            '<text x="50" y="50" class="alert alert-error">'
            'Unable to generate SVG plot.'
            '</text>'
            '</svg>')
    return svg


def output_pdf(fig):
    """
    Generates an PDF from the Plotly figure. Errors in the process are logged
    and an PDF with error text is returned.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure

    Returns
    -------
    pdf : str
       An ASCII-85 encoded PDF

    Notes
    -----
    Requires `Orca <https://plot.ly/python/orca-management/>`_ for generating
    pdfs. If orca is not installed, an pdf with an error message will be
    returned.
    """
    try:
        pdf = base64.a85encode(fig.to_image(format='pdf')).decode('utf-8')
    except Exception:
        try:
            name = fig.layout.title['text'][3:-4]
        except Exception:
            name = 'unnamed'
        logger.error('Could not generate PDF for figure %s', name)
        # should have same text as fail SVG
        pdf = fail_pdf
    return pdf


def raw_report_plots(report, metrics):
    """Create a RawReportPlots object from the metrics of a report.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`
    metrics: tuple of :py:class:`solarforecastarbiter.datamodel.MetricResult`

    Returns
    -------
    :py:class:`solarforecastarbiter.datamodel.RawReportPlots`
    """
    metrics_df = construct_metrics_dataframe(metrics, rename=abbreviate)
    # Create initial bar figures
    figure_dict = {}
    # Components for other metrics
    for category in report.report_parameters.categories:
        for metric in report.report_parameters.metrics:
            if category == 'total':
                fig = bar(metrics_df, metric)
                figure_dict[f'total::{metric}::all'] = fig
            else:
                figs = bar_subdivisions(metrics_df, category, metric)
                for name, fig in figs.items():
                    figure_dict[f'{category}::{metric}::{name}'] = fig
    mplots = []

    for k, v in figure_dict.items():
        cat, met, name = k.split('::', 2)
        figure_spec = v.to_json()
        pdf = output_pdf(v)
        mplots.append(datamodel.PlotlyReportFigure(
            name=name, category=cat, metric=met, spec=figure_spec,
            pdf=pdf, figure_type='bar'))

    out = datamodel.RawReportPlots(tuple(mplots), plotly_version)
    return out


def timeseries_plots(report):
    """Return the components for timeseries and scatter plots of the
    processed forecasts and observations.

    Parameters
    ----------
    report: :py:class:`solarforecastarbiter.datamodel.Report`

    Returns
    -------
    timeseries_spec: str
        String json specification of the timeseries plot. None if no
        forecast values are available.
    scatter_spec: None or str
        String json specification of the scatter plot. None if no observation
        values are available.
    timeseries_prob_spec: None or str
        If report contains a probabilistic forecast with axis='x',
        string json specification of the probability vs. time plot.
        Otherwise None.
    includes_distribution: bool
        True if the a plot was created for a pair containing a
        ProbabilisticForecast.
    """
    value_df, meta_df = construct_timeseries_dataframe(report)

    if value_df.empty:
        # No forecast data, don't plot anything
        return None, None, None, False

    pfxobs = report.raw_report.processed_forecasts_observations
    units = pfxobs[0].original.forecast.units
    units = units.replace('^2', '<sup>2</sup>')

    # data (units) vs time plot for the observation, deterministic fx,
    # and y-axis probabilistic fx
    ts_fig = timeseries(
        value_df, meta_df, report.report_parameters.start,
        report.report_parameters.end, units, (None, 'y'),
        report.raw_report.timezone)
    ts_fig.update_layout(
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(size=14),
        margin=PLOT_MARGINS,
    )
    if ts_fig.data:
        ts_fig_json = ts_fig.to_json()
    else:
        ts_fig_json = None

    # probability vs time plot for the x-axis probabilistic fx
    if any(
            (
                isinstance(pfxob.original.forecast, (
                    datamodel.ProbabilisticForecast,
                    datamodel.ProbabilisticForecastConstantValue)) and
                pfxob.original.forecast.axis == 'x')
            for pfxob in pfxobs
            ):
        ts_prob_fig = timeseries(
            value_df, meta_df, report.report_parameters.start,
            report.report_parameters.end, units, ('x',),
            report.raw_report.timezone)
        ts_prob_fig.update_layout(
            plot_bgcolor=PLOT_BGCOLOR,
            font=dict(size=14),
            margin=PLOT_MARGINS,
        )
        if ts_prob_fig.data:
            ts_prob_fig_json = ts_prob_fig.to_json()
        else:
            ts_prob_fig_json = None
    else:
        ts_prob_fig_json = None

    # switch secondary plot based on forecast type
    pfxobs = report.raw_report.processed_forecasts_observations
    fx = pfxobs[0].original.forecast
    if isinstance(fx, datamodel.EventForecast):
        scat_fig = event_histogram(value_df, meta_df)
        scat_fig.update_layout(
            plot_bgcolor=PLOT_BGCOLOR,
            font=dict(size=14),
            margin=PLOT_MARGINS,
        )
    else:
        margin = PLOT_MARGINS.copy()
        margin.pop('pad', None)
        scat_fig = scatter(value_df, meta_df, units)
        scat_fig.update_layout(
            plot_bgcolor=PLOT_BGCOLOR,
            font=dict(size=14),
            width=700,
            height=500,
            autosize=False,
            xaxis=dict(scaleanchor="y", scaleratio=1, constrain="domain"),
            yaxis=dict(constrain="domain"),
            margin=margin,
        )
    if scat_fig.data:
        scat_fig_json = scat_fig.to_json()
    else:
        scat_fig_json = None
    includes_distribution = ts_fig_json is not None and any(
        (
            isinstance(pfxob.original.forecast,
                       datamodel.ProbabilisticForecast) and
            pfxob.original.forecast.axis == 'y')
        for pfxob in pfxobs)
    return (ts_fig_json, scat_fig_json, ts_prob_fig_json,
            includes_distribution)
