from pathlib import Path


import numpy as np
import xarray as xr


from solarforecastarbiter.io import nwp


BASE_PATH = Path(nwp.__file__).resolve().parents[0] / 'tests/data'


def test_load_pnt_latlon():
    lats = np.array([32.0, 32.25, 32.5, 32.75, 33.0])
    lons = np.array([-111.0, -110.75, -110.5, -110.25, -110.0])
    ds = xr.Dataset(coords={'latitude': lats, 'longitude': lons})
    pnt = nwp._load_pnt(ds, 32.22, -110.9, 500)
    assert pnt.latitude == 32.25
    assert pnt.longitude == -111.0


def test_load_pnt_xy():
    y = xr.Variable(
        'y', np.array([951000., 954000., 957000., 960000., 963000.]),
        {'long_name': 'y coordinate of projection',
         'standard_name': 'projection_y_coordinate',
         'units': 'm',
         'grid_spacing': 3000.0})
    x = xr.Variable(
        'x', np.array([1470000., 1473000., 1476000., 1479000., 1482000.]),
        {'long_name': 'x coordinate of projection',
         'standard_name': 'projection_x_coordinate',
         'units': 'm',
         'grid_spacing': 3000.0})
    lat = xr.Variable(
        ('y', 'x'),
        np.array([[32.011720, 32.015625, 32.019530, 32.022460, 32.026367],
                  [32.038086, 32.041992, 32.045900, 32.049805, 32.052734],
                  [32.064453, 32.068360, 32.072266, 32.076170, 32.080080],
                  [32.090820, 32.094727, 32.098633, 32.102540, 32.106445],
                  [32.118164, 32.122070, 32.125000, 32.128906, 32.132812]],
                 dtype='float32'))
    lon = xr.Variable(
        ('y', 'x'),
        np.array([[249.51953, 249.55078, 249.58203, 249.61328, 249.64453],
                  [249.51465, 249.54590, 249.57812, 249.60938, 249.64062],
                  [249.51074, 249.54199, 249.57324, 249.60449, 249.63574],
                  [249.50586, 249.53711, 249.56836, 249.60059, 249.63184],
                  [249.50195, 249.53320, 249.56445, 249.59570, 249.62695]],
                 dtype='float32'))
    ds = xr.Dataset(coords={'latitude': lat, 'longitude': lon,
                            'x': x, 'y': y})
    pnt = nwp._load_pnt(ds, 32.05, -110.4, 500)
    assert (pnt.latitude - 32.049805) < 1e-6
    assert (pnt.longitude - 249.60938) < 1e-6
