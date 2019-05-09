"""Contains a config dictionary mapping SFA variable names to MIDC fields.
The values in these fields are as they would appear returned from the
`pvlib.iotools.read_midc_raw_data_from_nrel` function.

All site's field list can be found at:
    https://midcdmz.nrel.gov/apps/daily.pl?site=<SITE ID>&live=1
Where id is the key found in this dictionary
"""
midc_site_var_map = {
    'BMS': {
        'ghi': 'ghi_CMP22_(vent/cor)',
        'dni': 'dni_NIP',
        'dhi': 'dhi_CM22-1_(vent/cor)',
        'wind_speed': 'Avg Wind Speed @ 6ft [m/s]',
        'air_temperature': 'Tower Dry Bulb Temp [deg C]',
        'relative_humidity': 'Tower RH [%]',
    },
    'UOSMRL':{
        'ghi': 'ghi_CMP22',
        'dni': 'dni_NIP',
        'dhi': 'dhi_Schenk',
        'air_temperature': 'temp_air',
        'relative_humidity': 'relative_humidity',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
    },
    'HSU':{
        'ghi': 'ghi_Horiz',
        'dni': 'dni_Normal_(calc)',
        'dhi': 'dhi_Horiz_(band_corr)',
    },
    'UTPASRL':{
        'ghi': 'ghi_Horizontal',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horizontal',
        'air_temperature': 'CHP1_Temp',
    },
    'UAT':{
        'ghi': 'ghi_Horiz_(platform)',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horiz',
        'air_temperature': 'temp_air',
        'relative_humidity': 'Rel Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
    },
    'STAC':{
        'ghi': 'ghi_Horizontal',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horizontal',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
        'air_temperature': 'temp_air',
        'relative_humidity': 'Rel Humidity [%]',
    },
    'UNLV':{
        'ghi': 'ghi_Horiz',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horiz_(calc)',
        'air_temperature': 'Dry Bulb Temp [deg C]',
        'wind_speed': 'Avg Wind Speed @ 30ft [m/s]',
    },
    'ORNL':{
        'ghi': 'ghi_Horizontal',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horizontal',
        'air_temperature': 'temp_air',
        'relative_humidity': 'Rel_Humidity',
        'wind_speed': 'Avg Wind Speed @ 42ft [m/s]',
    },
    'NELHA':{
        'ghi': 'ghi_Horizontal',
        'air_temperature': 'temp_air',
        'wind_speed': 'Avg Wind Speed @ 10m [m/s]',
        'relative_humidity': 'Rel Humidity [%]',
    },
    'ULL':{
        'ghi': 'ghi_Horizontal',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horizontal',
        'air_temperature': 'temp_air',
        'relative_humidity': 'Rel Humidity [%]',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
    },
    'VTIF':{
        'ghi': 'ghi_Horizontal',
        'dni': 'dni_Normal',
        'dhi': 'dhi_Horizontal',
        'air_temperature': 'temp_air',
        'wind_speed': 'Avg Wind Speed @ 3m [m/s]',
        'relative_humidity': 'Rel Humidity [%]'
    },
    'NWTC':{
        'ghi': 'ghi_PSP',
        'air_temperature': 'temp_air_@_2m',
        'wind_speed': 'Avg Wind Speed @ 2m [m/s]',
        'relative_humidity': 'relative_humidity',
    },
}
