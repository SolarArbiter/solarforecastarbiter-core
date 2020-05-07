pvdaq_var_map = {
    4: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    10: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    33: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    34: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance',
            'wind_speed': 'wind_speed'
            },
        'freq': '15T'},
    39: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    50: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp'
            },
        'freq': '1T'},
    51: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    1199: {
        'columns': {
            'ac_power': 'inv1_ac_power',
            'dc_power': 'inv1_dc_power'
            },
        'freq': '5T'},  # jitter
    1200: {
        'columns': {
            'ac_power': 'ac_power', 'ac_power_metered'}, 'freq': '5T'},
    1201: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power'
            },
        'freq': '5T'},
    1202: {
        'columns': {
            'ac_power': 'ac_power_metered'
            },
        'freq': '5T'},
    1208: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power'
            },
        'freq': '15s'},  # convert to 1 min somewhere
    1232: {
        'columns': {
            'ac_power': 'ac_power'
            },
        'freq': '15T'},
    1234: {
        'columns': {
            'ac_power': 'ac_power_metered_A'  # ignore inverter B due to nans
            },
        'freq': '15T'},
    1276: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance',
            'wind_speed': 'wind_speed'
            },
        'freq': '15T'},
    1277: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance',
            'wind_speed': 'wind_speed'
            },
        'freq': '15T'},
    1278: {
        'columns': {
            'ac_power': 'inv1_ac_power',  # ignore inverter 2 due to nans
            'dc_power': 'inv1_dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance',
            'wind_speed': 'wind_speed'
            },
        'freq': '15T'},
    1283: {
        'columns': {
            'poa_global': 'poa_irradiance',
            'ac_power': 'ac_power',
            'air_temperature': 'ambient_temp',
            'dc_power': 'dc_power',
            'wind_speed': 'wind_speed'
            },
        'freq': '15s'},  # convert to 1 min somewhere
    1289: {
        'columns': {
            'ac_power': 'ac_power',
            'dc_power': 'dc_power',
            'air_temperature': 'ambient_temp',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'},
    1332: {
        'columns': {
            'ac_power': 'ac_power_metered'
            },
        'freq': '15s'},  # convert to 1 min somewhere
    # RTC sites that are already fetched from Sandia
    # 1403: {
    #     'columns': {
    #         'ac_power': 'ac_power',
    #         'air_temperature': 'ambient_temp',
    #         'dc_power': 'dc_power',
    #         'poa_global': 'poa_irradiance'
    #         },
    #     'freq': '1T'},
    # 1423: {
    #     'columns': {
    #         'ac_power': 'ac_power',
    #         'air_temperature': 'ambient_temp',
    #         'dc_power': 'dc_power',
    #         'poa_global': 'poa_irradiance'
    #         },
    #     'freq': '1T'},
    1426: {
        'columns': {
            'ac_power': 'AC_PowerA',
            'air_temperature': 'ambient_temperature',
            'dc_power': 'dc_power',
            'poa_global': 'poa_irradiance'
            },
        'freq': '1T'}
}
