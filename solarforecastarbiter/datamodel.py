# coding: utf-8
"""
Data classes and acceptable variables as defined by the SolarForecastArbiter
Data Model document. Python 3.7 is required.
"""

from dataclasses import dataclass, field
import datetime


import pandas as pd


ALLOWED_VARIABLES = {
    'surface_temperature': 'degC',
    'surface_wind_speed': 'm/s',
    'ghi': 'W/m^2',
    'dni': 'W/m^2',
    'dhi': 'W/m^2',
    'poa_global': 'W/m^2',
    'relative_humidity': '%',
    'ac_power': 'MW',
    'dc_power': 'MW',
}


@dataclass(frozen=True)
class GetItem(object):
    """Add a __getitem__ method for dict like access"""
    def __getitem__(self, key):
        return self.__dict__[key]


@dataclass(frozen=True)
class Site(GetItem):
    """
    Class for keeping track of Site metadata.

    Parameters
    ----------
    name : str
        Name of the Site, e.g. Desert Rock
    latitude : float
        Latitude of the Site in decimal degrees north of the equator,
        e.g. 36.62373
    longitude : float
        Longitude of the Site in decimal degrees east of the
        prime meridian, e.g. -116.01947
    elevation : float
        Elevation of the Site in meters above mean sea level, e.g. 1007
    timezone : str
        IANA timezone of the Site, e.g. Etc/GMT+8
    network : str, optional
        Measurement network name, e.g. SURFRAD
    well_known_text: str, optional
        Describes a geometric area for a Site which may be physically extended,
        e.g. a polygon over a city for a Site that describes many distributed
        generation PV systems.
    extra_parameters : dict, optional
        The extra parameters may be used by forecasters when
        implementing other PV models. The framework does not provide
        a standard set of extra parameters or require a particular
        format – these are up to the site owner.
    """
    name: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    network: str = ''
    well_known_text: str = ''
    extra_parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PVModelingParameters(GetItem):
    """
    Class for keeping track of generic PV modeling parameters

    Parameters
    ----------
    ac_power : float
        Nameplate AC power rating in megawatts
    dc_power : float
        Nameplate DC power rating in megawatts
    temperature_coefficient : float
        The temperature coefficient of DC power in units of 1/C.
        Typically -0.002 to -0.005 per degree C.

    See Also
    --------
    FixedTiltModelingParameters
    SingleAxisModelingParameters
    """
    ac_power: float
    dc_power: float
    temperature_coefficient: float


@dataclass(frozen=True)
class FixedTiltModelingParameters(PVModelingParameters):
    """
    A class based on PVModelingParameters that has additional parameters
    for fixed tilt PV systems.

    Parameters
    ----------
    surface_tilt : float
        Tilt from horizontal of a fixed tilt system, degrees
    surface_azimuth : float
        Azimuth angle of a fixed tilt system, degrees East of North


    See Also
    --------
    PVModelingParameters
    """
    surface_tilt: float
    surface_azimuth: float


@dataclass(frozen=True)
class SingleAxisModelingParameters(PVModelingParameters):
    """
    A class based on PVModelingParameters that has additional parameters
    for single axis tracking systems.

    Parameters
    ----------
    axis_tilt : float
        Tilt from horizontal of the tracker axis, degrees
    axis_azimuth : float
        Azimuth angle of the tracker axis, degrees East of North
    ground_coverage_ratio : float
        Ratio of total width of modules on a tracker to the distance between
        tracker axes. For example, for trackers each with two modules of 1m
        width each, and a spacing between tracker axes of 7m, the ground
        coverage ratio is 0.286(=2/7).
    backtrack : bool
        Indicator of if a tracking system uses backtracking
    maximum_rotation_angle : float
        maximum rotation from horizontal of a single axis tracker, degrees

    See Also
    --------
    PVModelingParameters
    """
    axis_tilt: float
    axis_azimuth: float
    ground_coverage_ratio: float
    backtrack: bool
    maximum_rotation_angle: float


@dataclass(frozen=True)
class SingleAxisPowerPlant(Site):
    """
    Class for keeping track of metadata associated with single axis tracking
    solar power plant Sites. Adds additional parameters to the Site dataclass.

    Parameters
    ----------
    modeling_parameters : SingleAxisModelingParameters
        Modeling parameters for a single axis system

    See Also
    --------
    Site
    SingleAxisModelingParameters
    """
    modeling_parameters: SingleAxisModelingParameters = field(
        default_factory=SingleAxisModelingParameters)


@dataclass(frozen=True)
class FixedTiltPowerPlant(Site):
    """
    Class for keeping track of metadata associated with fixed tilt tracking
    solar power plant Sites. Adds additional parameters to the Site dataclass.

    Parameters
    ----------
    modeling_parameters : FixedTiltModelingParameters
        Modeling parameters for a fixed tilt system

    See Also
    --------
    Site
    FixedTiltModelingParameters
    """
    modeling_parameters: FixedTiltModelingParameters = field(
        default_factory=FixedTiltModelingParameters)


@dataclass(frozen=True)
class UnitsSetter(GetItem):
    def __post_init__(self):
        if self.variable not in ALLOWED_VARIABLES:
            raise ValueError('variable is not allowed')
        object.__setattr__(self, 'units', ALLOWED_VARIABLES[self.variable])


@dataclass(frozen=True)
class Observation(UnitsSetter):
    """
    A class for keeping track of metadata associated with an observation.
    Units are set according to the variable type.

    Parameters
    ----------
    name : str
        Name of the Observation
    variable : str
        Variable name, e.g. power, GHI. Each allowed variable has an
        associated pre-defined unit.
    value_type : str
        The type of the data in the observation. Typically interval mean or
        instantaneous, but additional types may be defined for events.
    interval_label : str
        Indicates if a time labels the beginning or the ending of an interval
        average, or indicates an instantaneous value, e.g. beginning, ending,
        instant
    site : Site
        The site that this Observation was generated for.
    uncertainty : float
        A measure of the uncertainty of the observation values. The format
        will be determined later.
    description : str, optional
        A text description of the observation.
    extra_parameters : dict, optional
        Any extra parameters for the observation

    See Also
    --------
    Site
    """
    name: str
    variable: str
    value_type: str
    interval_label: str
    site: Site
    uncertainty: float
    description: str = ''
    extra_parameters: dict = field(default_factory=dict)
    units: str = field(init=False)


@dataclass(frozen=True)
class Forecast(UnitsSetter):
    """
    A class to hold metadata for Forecast objects.

    Parameters
    ----------
    name : str
        Name of the Forecast
    issue_time_of_day : datetime.time
        The time of day that a forecast run is issued, e.g. 00:30. For
        forecast runs issued multiple times within one day (e.g. hourly),
        this specifies the first issue time of day. Additional issue times
        are uniquely determined by the first issue time and the run length &
        issue frequency attribute.
    lead_time_to_start : pandas.Timedelta
        The difference between the issue time and the start of the first
        forecast interval, e.g. 1 hour.
    interval_length : pandas.Timedelta
        The length of time that each data point represents, e.g. 5 minutes,
        1 hour.
    run_length : pandas.Timedelta
        The total length of a single issued forecast run, e.g. 1 hour.
        To enforce a continuous, non-overlapping sequence, this is equal
        to the forecast run issue frequency.
    interval_label : str
        Indicates if a time labels the beginning or the ending of an interval
        average, or indicates an instantaneous value, e.g. beginning, ending,
        instant.
    value_type : str
        The type of the data in the forecast, e.g. mean, max, 95th percentile.
    variable : str
        The variable in the forecast, e.g. power, GHI, DNI. Each variable is
        associated with a standard unit.
    site : Site
        The predefined site that the forecast is for, e.g. Power Plant X
        or Aggregate Y.
    extra_parameters : dict
        Extra configuration parameters of forecast.

    See Also
    --------
    Site
    """
    name: str
    issue_time_of_day: datetime.time
    lead_time_to_start: pd.Timedelta
    interval_length: pd.Timedelta
    run_length: pd.Timedelta
    interval_label: str
    value_type: str
    variable: str
    site: Site
    extra_parameters: dict = field(default_factory=dict)
    units: str = field(init=False)
