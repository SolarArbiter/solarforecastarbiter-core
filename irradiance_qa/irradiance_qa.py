"""
Manual QA processing for sites selected for Solar Forecasting 2 Topic
Area 2 and 3 evaluations.
"""

import json
import logging
import os
from pathlib import Path

import click
import pandas as pd
import numpy as np
import pvlib
import matplotlib.pyplot as plt
from solarforecastarbiter.cli import common_options, cli_access_token
from solarforecastarbiter.io.api import APISession

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
        lambda x: x.site in ta23_sites and x.variable in ['ghi', 'dni'],
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
    loc = read_metadata()
    data = read_irradiance()
    # TA2/3 analysis uses fixed offsets, but reference site metadata
    # typically uses DST aware timezones
    data = data.tz_localize(parameters['timezone'])

@qa_cli.command()
@common_options
def post(verbose, user, password, base_url):
    """Post QA results for all TA 2/3 sites.

    Reads data from directory ta23_site_data.
    Posting requires access to reference data account."""
    set_log_level(verbose)
    token = cli_access_token(user, password)
    session = APISession(token, base_url=base_url)


def read_metadata(dirn, filen):
    with open (os.path.join(dirn, filen), 'r') as infile:
        meta = json.load(infile)
    loc = pvlib.location.Location(meta['latitude'], meta['longitude'],
                                  meta['timezone'], meta['elevation'])
    return loc


def read_irradiance(dirn, data_files):
    all_data = pd.DataFrame(columns=data_files.keys())
    for k in data_files.keys():
        with open(os.path.join(dirn, data_files[k]), 'r') as infile:
            data = pd.read_csv(infile, skiprows=2, index_col=0,
                               parse_dates=True)
        all_data[k] = data['value']
    return all_data


def go():
    dirn = 'D:\\SFA\\BoulderCO'
    meta_filen = 'NOAA_SURFRAD_Table_Mountain_Boulder_CO.json'

    data_files = {'ghi': 'Table_Mountain_Boulder_CO_ghi_2018-01-07T00_00_00+00_00-2019-01-01T06_59_00+00_00.csv',
                'dni': 'Table_Mountain_Boulder_CO_dni_2018-01-01T07_00_00+00_00-2019-01-01T06_59_00+00_00.csv',
                'dhi': 'Table_Mountain_Boulder_CO_dhi_2018-01-01T07_00_00+00_00-2019-01-01T06_59_00+00_00.csv'}

    loc = read_metadata(dirn, meta_filen)
    data = read_irradiance(dirn, data_files)
    data = data.tz_convert(loc.tz)

    # replace negative DHI with 0, so that negative DNI doesn't amplify the ratio
    # of measured GHI to component sum GHI
    data['dni'] = np.maximum(data['dni'], 0.)

    sp = loc.get_solarposition(data.index)
    cs = loc.get_clearsky(data.index, solar_position=sp)
    daytime = sp['zenith'] < 87

    # check for component consistency

    limits = pva.quality.irradiance.QCRAD_CONSISTENCY.copy()
    # reset lower bound on GHI
    for k in limits:
        for m in ['low_zenith', 'high_zenith']:
            limits[k][m]['ghi_bounds'] = [0, np.Inf]
    # raise limit on diffuse ratio
    for m in limits['dhi_ratio']:
        limits['dhi_ratio'][m]['ratio_bounds'] = [0, 1.5]

    consistent_comp, diffuse_ratio_limit = pva.quality.irradiance.check_irradiance_consistency_qcrad(
        data['ghi'], sp['zenith'], data['dhi'], data['dni'], param=limits)

    # accept GHI and DHI when nearly equal, but not at very high zenith so that
    # we don't accept horizon shading
    overcast_ok = (sp['zenith'] < 75) & (np.abs(data['ghi'] - data['dhi']) < 50)

    good_overall = (consistent_comp | overcast_ok) & diffuse_ratio_limit

    component_sum = data['dni'] * pvlib.tools.cosd(sp['zenith']) + data['dhi']
    ghi_ratio = data['ghi'] / component_sum

    bad_comp = ~consistent_comp & daytime
    bad_comp = data['ghi'] * bad_comp
    bad_comp[bad_comp == 0] = np.nan
    fig_cons = plt.figure()
    plt.plot(data['ghi'])
    plt.plot(data['dni'])
    plt.plot(data['dhi'])
    plt.plot(bad_comp, 'r.')
    plt.legend(['GHI', 'DNI', 'DHI', "Bad"])
    plt.title('Consistent components test')

    bad_diff = ~diffuse_ratio_limit & daytime
    bad_diff = data['ghi'] * bad_diff
    bad_diff[bad_diff == 0] = np.nan
    fig_cons = plt.figure()
    plt.plot(data['ghi'])
    plt.plot(data['dni'])
    plt.plot(data['dhi'])
    plt.plot(bad_diff, 'r.')
    plt.legend(['GHI', 'DNI', 'DHI', "Bad"])
    plt.title('Diffuse fraction test')

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
    plt.title('Overall')


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
    plt.legend(['Total', 'Consistent OR Overcast', 'Diffuse', 'Passed all tests'])
    plt.title('Boulder, CO')


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


    # plot one overcast day for illustration
    dr = pd.date_range(start='2018-02-04 06:00:00', end='2018-02-04 18:00:00',
                    freq='1T', tz=data.index.tz)
    fig_overcast_day = plt.figure()
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
    plt.title('Representative overcast day at Boulder, CO')

    # plot one clear day day for illustration
    dr = pd.date_range(start='2018-03-06 06:00:00', end='2018-03-06 18:00:00',
                    freq='1T', tz=data.index.tz)
    fig_clear_day = plt.figure()
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
    plt.title('Representative clear day at Boulder, CO')

    # plot one clear day day for illustration
    dr = pd.date_range(start='2018-03-22 06:00:00', end='2018-03-22 18:00:00',
                    freq='1T', tz=data.index.tz)
    fig_clear_day = plt.figure()
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
    plt.title('Representative day with variable conditions at Boulder, CO')


if __name__ == "__main__":  # pragma: no cover
    qa_cli()
