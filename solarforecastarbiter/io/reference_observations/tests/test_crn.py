import pandas as pd

from solarforecastarbiter.io.reference_observations import crn
from solarforecastarbiter.datamodel import Site


test_site_dict = {
    "extra_parameters": '{"network": "NOAA USCRN", "network_api_id": "NY_Millbrook_3_W", "network_api_abbreviation": "NY_Millbrook_3_W", "observation_interval_length": 5}',  # noqa: E501
    "climate_zones": [
      "Reference Region 5"
    ],
    "created_at": "2019-05-24T17:31:44+00:00",
    "elevation": 126,
    "latitude": 41.78,
    "longitude": -73.74,
    "modified_at": "2020-09-15T18:22:05+00:00",
    "name": "NOAA USCRN Millbrook NY",
    "provider": "Reference",
    "site_id": "cc9f4180-7e49-11e9-b191-0a580a8003e9",
    "timezone": "America/New_York"
}

test_site_object = Site.from_dict(test_site_dict)


def test_fetch_fill_nan_99999(mocker):
    mocker.patch(
        'solarforecastarbiter.io.reference_observations.crn.iotools.read_crn',
        return_value=pd.DataFrame(
            data=[-99999],
            index=pd.DatetimeIndex(['20220101T0000Z']),
            columns=['ghi']
        )
    )
    result = crn.fetch(
        None,
        test_site_object,
        pd.Timestamp('20210101t0000z'),
        pd.Timestamp('20210101t0000z'),
    )
    assert pd.isna(result.iloc[0]['ghi'])
