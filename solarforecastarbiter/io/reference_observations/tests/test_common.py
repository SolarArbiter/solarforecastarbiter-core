import pytest

from solarforecastarbiter.datamodel import Site
from solarforecastarbiter.io.reference_observations import common


sites_list = [
    {
        'name': 'site',
        'latitude': 1,
        'longitude': 1,
        'elevation': 5,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': '{"network": "ARM"}',
    },
    {
        'name': 'site2',
        'latitude': 2,
        'longitude': 2,
        'elevation': 5,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': '{"network": "NOAA SURFRAD"}',
    },
    {
        'name': 'site3',
        'latitude': 3,
        'longitude': -3,
        'elevation': 6,
        'timezone': 'Etc/GMT+8',
        'extra_parameters': '{"network": "NOAA SOLRAD"}',
    }
]
site_objects = [Site.from_dict(site) for site in sites_list] 


@pytest.fixture()
def site_dicts():
    return sites_list

    
def test_decode_extra_parameters(site_dicts):
    metadata = Site.from_dict(site_dicts[0])
    assert common.decode_extra_parameters(metadata) == {'network': 'ARM'}

@pytest.mark.parametrize('networks,site,expected', [
    (['ARM'], sites_list[0], True),
    ('ARM', sites_list[0], True),
    (['ARM', 'NREL MIDC'], sites_list[1], False),
])
def test_check_network(networks, site, expected):
    metadata = Site.from_dict(site)
    assert common.check_network(networks, metadata) == expected

@pytest.mark.parametrize('networks,expected', [
    (['ARM'], site_objects[:1]),
    ('ARM',  site_objects[:1]),
    (['NOAA SURFRAD', 'NOAA SOLRAD'], site_objects[1:])
])
def test_filter_by_network(site_dicts, networks, expected):
     assert common.filter_by_networks(site_objects, networks) == expected
     

