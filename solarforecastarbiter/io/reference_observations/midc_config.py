"""Contains a config dictionary mapping SFA variable names to MIDC fields.
The values in these fields are as they would appear returned from the
`pvlib.iotools.read_midc_raw_data_from_nrel` function.

All site's field list can be found at:
https://midcdmz.nrel.gov/apps/daily.pl?site=<SITE ID>&live=1
Where id is the key found in this dictionary
"""

midc_var_map = {
    'BMS': {
        'ghi': 'Global CMP22 (vent/cor) [W/m^2]',
        'dni': 'Direct NIP [W/m^2]',
        'dhi': 'Diffuse CM22-1 (vent/cor) [W/m^2]',
        'wind_speed': 'Avg Wind Speed @ 33ft [m/s]',
        'air_temperature': 'Tower Dry Bulb Temp [deg C]',
        'relative_humidity': 'Tower RH [%]',
    },
    'UOSMRL': {
        'ghi': 'Global CMP22 [W/m^2]',
        'dni': 'Direct NIP [W/m^2]',
        'dhi': 'Diffuse Schenk [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'relative_humidity': 'Relative Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
    },
    'HSU': {
        'ghi': 'Global Horiz [W/m^2]',
        'dni': 'Direct Normal (calc) [W/m^2]',
        'dhi': 'Diffuse Horiz (band corr) [W/m^2]',
    },
    'UTPASRL': {
        'ghi': 'Global Horizontal [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horizontal [W/m^2]',
        'air_temperature': 'CHP1 Temp [deg C]',
    },
    'UAT': {
        'ghi': 'Global Horiz (platform) [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horiz [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'relative_humidity': 'Rel Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
    },
    'STAC': {
        'ghi': 'Global Horizontal [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horizontal [W/m^2]',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
        'air_temperature': 'Air Temperature [deg C]',
        'relative_humidity': 'Rel Humidity [%]',
    },
    'UNLV': {
        'ghi': 'Global Horiz [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horiz (calc) [W/m^2]',
        'air_temperature': 'Dry Bulb Temp [deg C]',
        'wind_speed': 'Avg Wind Speed @ 30ft [m/s]',
    },
    'ORNL': {
        'ghi': 'Global Horizontal [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horizontal [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'relative_humidity': 'Rel Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 42ft [m/s]',
    },
    'NELHA': {
        'ghi': 'Global Horizontal [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
        'relative_humidity': 'Rel Humidity [%]',
    },
    'ULL': {
        'ghi': 'Global Horizontal [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horizontal [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'relative_humidity': 'Rel Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
    },
    'VTIF': {
        'ghi': 'Global Horizontal [W/m^2]',
        'dni': 'Direct Normal [W/m^2]',
        'dhi': 'Diffuse Horizontal [W/m^2]',
        'air_temperature': 'Air Temperature [deg C]',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
        'relative_humidity': 'Rel Humidity [%]'
    },
    'NWTC': {
        'ghi': 'Global PSP [W/m^2]',
        'air_temperature': 'Temperature @ 2m [deg C]',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
        'relative_humidity': 'Relative Humidity [%]',
    },
}
