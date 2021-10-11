"""
Manual QA processing for sites selected for Solar Forecasting 2 Topic
Area 2 and 3 evaluations.
"""

import json
import logging
from pathlib import Path

import click
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from solarforecastarbiter.cli import common_options, cli_access_token
from solarforecastarbiter.io.api import APISession
from solarforecastarbiter.validation.validator import (
    QCRAD_CONSISTENCY,
    check_irradiance_consistency_QCRad
)

SITES = {
    'NOAA SURFRAD Table Mountain Boulder CO': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]}
        },
        'overcast': '2018-02-03',
        'clear': '2018-04-20',
        'variable': '2018-03-03',
        'site_id': '9dfa7910-7e49-11e9-b4e8-0a580a8003e9',
        'timezone': 'Etc/GMT+7',
    },
    'DOE RTC Cocoa FL': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]}
        },
        'overcast': '2018-01-01',
        'clear': '2018-03-03',
        'variable': '2018-04-22',
        'site_id': 'a9d0d140-99fc-11e9-81fa-0a580a8200c9',
        'timezone': 'Etc/GMT+5',
    },
    'NOAA SURFRAD Goodwin Creek MS': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]},
            'ghi_ratio': {'ratio_bounds': [0.84, 1.16]},
        },
        'overcast': '2018-01-10',
        'clear': '2018-03-03',
        'variable': '2018-01-14',
        'site_id': '9e4e98ac-7e49-11e9-a7c4-0a580a8003e9',
        'timezone': 'Etc/GMT+6',
    },
    'NOAA SOLRAD Hanford California': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]},
            'ghi_ratio': {'ratio_bounds': [0.84, 1.16]},
        },
        'overcast': '2018-01-17',
        'clear': '2018-02-08',
        'variable': '2018-03-03',
        'site_id': 'c291964c-7e49-11e9-af46-0a580a8003e9',
        'timezone': 'Etc/GMT+8',
    },
    'NREL MIDC Humboldt State University': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]}
        },
        'overcast': '2018-01-14',
        'clear': '2018-07-01',
        'variable': '2018-03-03',
        'site_id': '9feac63a-7e49-11e9-9bde-0a580a8003e9',
        'timezone': 'Etc/GMT+8',
    },
    'DOE ARM Southern Great Plains SGP, Lamont, Oklahoma': {
        'consistency_limits': {},
        'overcast': '2018-02-10',
        'clear': '2018-01-23',
        'variable': '2018-05-04',
        'site_id': 'd52d47e6-88c4-11ea-b5f8-0a580a820092',
        'timezone': 'Etc/GMT+6',
    },
    'WRMC BSRN NASA Langley Research Center': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]}
        },
        'overcast': '2018-02-13',
        'clear': '2018-04-20',
        'variable': '2018-03-03',
        'site_id': '371a5e3a-1888-11eb-959e-0a580a820169',
        'timezone': 'Etc/GMT+5',
    },
    'NOAA SURFRAD Penn State Univ PA': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]},
            'ghi_ratio': {'ratio_bounds': [0.85, 1.15]},
        },
        'overcast': '2018-01-12',
        'clear': '2018-01-14',
        'variable': '2018-01-01',
        'site_id': '9e69b108-7e49-11e9-a3df-0a580a8003e9',
        'timezone': 'Etc/GMT+5',
    },
    'PNNL': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]},
            'ghi_ratio': {'ratio_bounds': [0.85, 1.15]},
        },
        'overcast': '2018-01-27',
        'clear': '2018-02-11',
        'variable': '2018-02-09',
        'site_id': '4a4e1f82-a2d1-11eb-90bf-0a580a820087',
        'timezone': 'Etc/GMT+8',
    },
    'NOAA SURFRAD Sioux Falls SD': {
        'consistency_limits': {
            'dhi_ratio': {'ratio_bounds': [0, 1.5]},
            'ghi_ratio': {'ratio_bounds': [0.85, 1.15]},
        },
        'overcast': '2018-01-14',
        'clear': '2018-01-01',
        'variable': '2018-03-03',
        'site_id': '9e888c48-7e49-11e9-9a66-0a580a8003e9',
        'timezone': 'Etc/GMT+6',
    },
}

# ideally would be set through an argument, but this is faster
OUTPUT_PATH = Path('ta23_site_data')

# config for command line interface
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
def qa_cli():
    """CLI for Solar Forecast Arbiter irradiance QA."""
    pass


def set_log_level(verbose):
    if verbose == 1:
        loglevel = 'INFO'
    elif verbose > 1:
        loglevel = 'DEBUG'
    else:
        loglevel = 'WARNING'
    logging.getLogger().setLevel(loglevel)


@qa_cli.command()
@common_options
def download(verbose, user, password, base_url):
    """Download metadata and time series data for all TA 2/3 sites.

    Data saved in new directory named ta23_site_data."""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    session = APISession(token, base_url=base_url)
    sites = session.list_sites()
    ta23_sites = tuple(filter(lambda x: x.name in SITES, sites))
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # observation names were not created consistently across the different
    # networks so need to keep a nested directory structure to keep things
    # clean. also mirrors the organization of data in the arbiter.
    for site in ta23_sites:
        p = OUTPUT_PATH / f'{site.name}'
        p.mkdir(exist_ok=True)
        p /= f'{site.name}.json'
        p.write_text(json.dumps(site.to_dict(), indent=4))
    obs = session.list_observations()
    ta23_obs = tuple(filter(
        lambda x: x.site in ta23_sites and x.variable in ['ghi', 'dni', 'dhi'],
        obs
    ))
    for o in ta23_obs:
        # WH: I'm terribly embarassed by the length of this for loop.
        logger.info('Fetching data for %s', o.name)
        # o.site.name will match paths created above
        p = OUTPUT_PATH / f'{o.site.name}' / f'{o.name}.json'
        p.write_text(json.dumps(o.to_dict(), indent=4))
        # pull data by quarters to work around API query length
        # limitation 2018 for TA2 analysis. eventually extend to 2021
        # for TA3. TA2/3 reports use Etc/GMT timezone, while some SFA
        # Site timezones are DST aware.
        tz = SITES[o.site.name]['timezone']
        quarters = pd.date_range('2018-01-01', freq='QS', periods=5, tz=tz)
        start_ends = pd.DataFrame(
            {'start': quarters[:4], 'end': quarters[1:]-pd.Timedelta('1s')}
        )
        values_segments = []
        for _, start_end in start_ends.iterrows():
            start = start_end['start']
            end = start_end['end']
            values_segment = session.get_observation_values(
                o.observation_id,
                start=start,
                end=end,
            )
            values_segments.append(values_segment)
        values = pd.concat(values_segments)
        # construct filename that follows same pattern as API but use our
        # requested times so we know for certain what the file will be named
        name = o.name.replace(' ', '_')
        first_start = start_ends.iloc[0]['start'].isoformat()
        last_end = start_ends.iloc[-1]['end'].isoformat()
        filename = f'{name}_{first_start}-{last_end}.csv'
        filename = filename.replace(':', '_')
        p_data = OUTPUT_PATH / f'{o.site.name}' / filename
        values.to_csv(p_data)


@qa_cli.command()
@click.option('-v', '--verbose', count=True,
              help='Increase logging verbosity')
@click.option('--site', type=str, help='Site to process', default='all')
def process(verbose, site):
    """Process time series data for all TA 2/3 sites.

    Reads data from directory ta23_site_data and writes QA results to
    this directory."""
    set_log_level(verbose)
    if site == 'all':
        sites_to_process = SITES
    else:
        sites_to_process = {site: SITES[site]}
    for name, parameters in sites_to_process.items():
        process_single_site(name, parameters)


def process_single_site(name, parameters):
    logger.info('Processing %s', name)
    loc = read_metadata(name)
    data = read_irradiance(name)
    # TA2/3 analysis uses fixed offsets, but reference site metadata
    # typically uses DST aware timezones
    data = data.tz_convert(parameters['timezone'])

    # replace negative DHI with 0, so that negative DNI doesn't amplify the
    # ratio of measured GHI to component sum GHI
    data['dni'] = np.maximum(data['dni'], 0.)

    sp = loc.get_solarposition(data.index)
    cs = loc.get_clearsky(data.index, solar_position=sp)
    # same as solarforecastarbiter.validation.validator.check_day_night
    daytime = sp['zenith'] < 87

    # check for component consistency

    limits = QCRAD_CONSISTENCY.copy()
    # reset lower bound on GHI throughout nested dictionary
    for irrad_ratio in limits:
        for m in ['low_zenith', 'high_zenith']:
            limits[irrad_ratio][m]['ghi_bounds'] = [0, np.Inf]
    # site-specific adjustments to ratio bounds
    new_limits = SITES[name]['consistency_limits']
    for irrad_ratio, new_bounds in new_limits.items():
        for m in ['low_zenith', 'high_zenith']:
            limits[irrad_ratio][m].update(new_bounds)

    consistent_comp, diffuse_ratio_limit = check_irradiance_consistency_QCRad(
        data['ghi'],
        sp['zenith'],
        data['dhi'],
        data['dni'],
        param=limits
    )

    # accept GHI and DHI when nearly equal, but not at very high zenith so that
    # we don't accept horizon shading
    overcast_ok = (sp['zenith'] < 75) & (np.abs(data['ghi'] - data['dhi']) < 50)

    good_overall = (consistent_comp | overcast_ok) & diffuse_ratio_limit

    component_sum = data['dni'] * pvlib.tools.cosd(sp['zenith']) + data['dhi']
    ghi_ratio = data['ghi'] / component_sum

    savefig_path = OUTPUT_PATH / name
    savefig_kwargs = dict(dpi=300)

    bad_comp = ~consistent_comp & daytime
    bad_comp = data['ghi'] * bad_comp
    bad_comp[bad_comp == 0] = np.nan
    fig_cons = plt.figure()
    plt.plot(data['ghi'])
    plt.plot(data['dni'])
    plt.plot(data['dhi'])
    plt.plot(bad_comp, 'r.')
    plt.legend(['GHI', 'DNI', 'DHI', "Bad"])
    plt.title(f'{name}\nConsistent components test')
    plt.savefig(
        savefig_path / f'{name} consistent components test.png',
        **savefig_kwargs,
    )

    bad_diff = ~diffuse_ratio_limit & daytime
    bad_diff = data['ghi'] * bad_diff
    bad_diff[bad_diff == 0] = np.nan
    fig_cons = plt.figure()
    plt.plot(data['ghi'])
    plt.plot(data['dni'])
    plt.plot(data['dhi'])
    plt.plot(bad_diff, 'r.')
    plt.legend(['GHI', 'DNI', 'DHI', "Bad"])
    plt.title(f'{name}\nDiffuse fraction test')
    plt.savefig(
        savefig_path / f'{name} diffuse fraction test.png',
        **savefig_kwargs,
    )

    # overall accept/reject plot
    fig_summary = plt.figure()
    plt.plot(data['ghi'])
    plt.plot(data['dni'])
    plt.plot(data['dhi'])
    good_mask = good_overall.copy()
    bad_mask = ~good_mask
    good_mask[good_mask == False] = np.nan
    bad_mask[bad_mask == False] = np.nan
    plt.plot(good_mask * data['ghi'], 'g.')
    plt.plot(bad_mask * data['ghi'], 'r.')
    plt.legend(['GHI', 'DNI', 'DHI', 'Good', "Bad"])
    plt.title(f'{name}\nOverall')
    plt.savefig(savefig_path / f'{name} overall.png', **savefig_kwargs)

    # report on count of data dropped by zenith bin
    bins = np.arange(np.min(sp['zenith']), np.max(sp['zenith'][daytime]), 1)
    count_tot = np.zeros(len(bins) - 1)
    count_good = count_tot.copy()
    count_cc = count_tot.copy()
    count_diff = count_tot.copy()
    for i in range(len(bins)-1):
        u = (sp['zenith'] >= bins[i]) & (sp['zenith'] < bins[i + 1])
        count_tot[i] = len(sp.loc[u, 'zenith'])
        count_cc[i] = (consistent_comp[u] | overcast_ok[u]).sum()
        count_diff[i] = diffuse_ratio_limit[u].sum()
        count_good[i] = good_overall[u].sum()
    fig_accept = plt.figure()
    plt.plot(bins[:-1], count_tot)
    plt.plot(bins[:-1], count_good)
    plt.plot(bins[:-1], count_cc)
    plt.plot(bins[:-1], count_diff)
    plt.xlabel('Zenith')
    plt.ylabel('Count')
    plt.legend(
        ['Total', 'Consistent OR Overcast', 'Diffuse', 'Passed all tests'])
    plt.title(f'{name}\nData dropped by zenith bin')
    plt.savefig(
        savefig_path / f'{name} data dropped by zenith bin.png',
        **savefig_kwargs,
    )

    # bar chart of data count within each hour
    hrs = range(4, 21)
    boxplot_data = []
    hr_count = good_overall.resample('H').sum()
    for idx, h in enumerate(hrs):
        boxplot_data.append(hr_count[hr_count.index.hour == h].values)
    fig_boxplot, ax_boxplot = plt.subplots()
    plt.boxplot(boxplot_data)
    ax_boxplot.set_xticklabels([str(h) for h in hrs])
    plt.xlabel('Hour of day')
    plt.ylabel('Count of data')
    plt.title(f'{name}\nData count within each hour')
    plt.savefig(
        savefig_path / f'{name} data count within each hour.png',
        **savefig_kwargs,
    )

    # plot one overcast, clear, and variable day for illustration
    for kind in ('overcast', 'clear', 'variable'):
        date = parameters[kind]
        dr = pd.date_range(
            start=f'{date} 06:00:00',
            end=f'{date} 18:00:00',
            freq='1T',
            tz=data.index.tz
        )
        fig_day = plt.figure()
        plt.plot(data.loc[dr, 'ghi'])
        plt.plot(data.loc[dr, 'dni'])
        plt.plot(data.loc[dr, 'dhi'])
        good_mask = good_overall.copy()
        bad_mask = ~good_mask
        good_mask[good_mask == False] = np.nan
        bad_mask[bad_mask == False] = np.nan
        plt.plot(good_mask * data.loc[dr, 'ghi'], 'g.')
        plt.plot(bad_mask * data.loc[dr, 'ghi'], 'r.')
        #plt.plot(sp.loc[dr, 'zenith'])
        plt.legend(['GHI', 'DNI', 'DHI', 'Good', 'Bad'])
        plt.title(f'{name}\nRepresentative {kind} day. {date}')
        plt.savefig(
            savefig_path / f'{name} representative {kind} day.png',
            **savefig_kwargs,
        )

    # determine deadband
    clear_times = pvlib.clearsky.detect_clearsky(
        data['ghi'], cs['ghi'], data.index, 10
    )
    ghi_rel_diff = (component_sum - data['ghi']) / data['ghi']
    u = daytime & clear_times & (ghi_ratio > 0) & (ghi_ratio < 2) & (data['ghi'] > 50)

    fig_deadband = plt.figure()
    plt.plot(ghi_rel_diff[u], 'r')
    plt.text(
        ghi_rel_diff.index[50000],
        -0.1,
        'Mean: ' + str(ghi_rel_diff[u].mean())
    )
    plt.text(
        ghi_rel_diff.index[50000],
        -0.15,
        '85%: ' + str(ghi_rel_diff[u].quantile(q=0.85))
    )
    plt.text(
        ghi_rel_diff.index[50000],
        -0.2,
        'Median: ' + str(ghi_rel_diff[u].quantile(q=0.5))
    )
    plt.text(
        ghi_rel_diff.index[50000],
        -0.25,
        '15%: ' + str(ghi_rel_diff[u].quantile(q=0.15))
    )
    plt.ylabel('(Comp. sum - GHI) / GHI')
    plt.savefig(
        savefig_path / f'{name} ghi ratio.png',
        **savefig_kwargs,
    )


@qa_cli.command()
@common_options
def post(verbose, user, password, base_url):
    """Post QA results for all TA 2/3 sites.

    Reads data from directory ta23_site_data.
    Posting requires access to reference data account."""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    session = APISession(token, base_url=base_url)


def read_metadata(name):
    metadata_file = OUTPUT_PATH / name / f'{name}.json'
    with open(metadata_file, 'r') as infile:
        meta = json.load(infile)
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'],
                                  meta['timezone'], meta['elevation'])
    return loc


def read_irradiance(name, column='value'):
    directory = OUTPUT_PATH / name
    variables = ['ghi', 'dni', 'dhi']
    data_all = {}
    for v in variables:
        # read in all csv files with e.g. ghi in the name
        data_variable = []
        for f in directory.glob(f'*{v}*.csv'):
            data_section = pd.read_csv(f, index_col=0, parse_dates=True)
            data_variable.append(data_section)
        data_variable = pd.concat(data_variable)
        data_all[v] = data_variable[column]
    data_all = pd.DataFrame(data_all)
    return data_all


if __name__ == "__main__":  # pragma: no cover
    qa_cli()
