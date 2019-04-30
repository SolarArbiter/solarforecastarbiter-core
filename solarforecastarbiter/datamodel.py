# coding: utf-8
"""
Data classes and acceptable variables as defined by the SolarForecastArbiter
Data Model document. Python 3.7 is required.
"""
from dataclasses import dataclass, field, fields, MISSING
import datetime


import pandas as pd


ALLOWED_VARIABLES = {
    'air_temperature': 'degC',
    'wind_speed': 'm/s',
    'ghi': 'W/m^2',
    'dni': 'W/m^2',
    'dhi': 'W/m^2',
    'poa_global': 'W/m^2',
    'relative_humidity': '%',
    'ac_power': 'MW',
    'dc_power': 'MW',
    'availability': '%',
    'curtailment': 'MW',
}


class BaseModel:
    @classmethod
    def from_dict(model, dict_, raise_on_extra=False):
        """
        Construct a dataclass from the given dict, matching keys with the class
        fields. A KeyError is raised for any missing values. If raise_on_extra
        is True, an errors is raised if keys of the dict are also not fields of
        the dataclass.

        Parameters
        ----------
        dict_ : dict
            The dict to process into dataclass fields
        raise_on_extra : boolean, default False
            If True, raise an exception on extra keys in dict_ that are not
            dataclass fields.

        Raises
        ------
        KeyError
            For missing required fields or if raise_on_extra is True and dict_
            contains extra keys.
        """
        model_fields = fields(model)
        kwargs = {}
        errors = []
        for model_field in model_fields:
            if model_field.name in dict_:
                if model_field.type == pd.Timedelta:
                    kwargs[model_field.name] = pd.Timedelta(
                        f'{dict_[model_field.name]}min')
                elif model_field.type == datetime.time:
                    kwargs[model_field.name] = datetime.datetime.strptime(
                        dict_[model_field.name], '%H:%M').time()
                elif model_field.name == 'modeling_parameters':
                    mp_dict = dict_.pop('modeling_parameters', {})
                    tracking_type = mp_dict.pop('tracking_type', '')
                    if tracking_type == 'fixed':
                        kwargs['modeling_parameters'] = (
                            FixedTiltModelingParameters.from_dict(
                                mp_dict))
                    elif tracking_type == 'single_axis':
                        kwargs['modeling_parameters'] = (
                            SingleAxisModelingParameters.from_dict(
                                mp_dict))
                else:
                    kwargs[model_field.name] = dict_[model_field.name]
            elif (
                    model_field.default is MISSING and
                    model_field.default_factory is MISSING and
                    model_field.init
            ):
                errors.append(model_field.name)
        if errors:
            raise KeyError(
                'Missing the following required arguments for the model '
                f'{str(model)}: {", ".join(errors)}')
        names = [f.name for f in model_fields]
        extra = [k for k in dict_.keys() if k not in names]
        if extra and raise_on_extra:
            raise KeyError(
                f'Extra keys for the model {str(model)}: {", ".join(extra)}')
        return model(**kwargs)


@dataclass(frozen=True)
class Site(BaseModel):
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
    site_id : str, optional
        UUID of the Site in the API
    provider : str, optional
        Provider of the Site information.
    well_known_text: str, optional
        Describes a geometric area for a Site which may be physically extended,
        e.g. a polygon over a city for a Site that describes many distributed
        generation PV systems.
    extra_parameters : str, optional
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
    site_id: str = ''
    provider: str = ''
    well_known_text: str = ''
    extra_parameters: str = ''


@dataclass(frozen=True)
class PVModelingParameters(BaseModel):
    """
    Class for keeping track of generic PV modeling parameters

    Parameters
    ----------
    ac_capacity : float
        Nameplate AC power rating in megawatts
    dc_capacity : float
        Nameplate DC power rating in megawatts
    temperature_coefficient : float
        The temperature coefficient of DC power in units of 1/C.
        Typically -0.002 to -0.005 per degree C.
    dc_loss_factor : float
        Applied to DC current in units of %. 0 = no loss.
    ac_loss_factor : float
        Appled to inverter power output in units of %. 0 = no loss.

    See Also
    --------
    FixedTiltModelingParameters
    SingleAxisModelingParameters
    """
    ac_capacity: float
    dc_capacity: float
    temperature_coefficient: float
    dc_loss_factor: float
    ac_loss_factor: float


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
    max_rotation_angle : float
        maximum rotation from horizontal of a single axis tracker, degrees

    See Also
    --------
    PVModelingParameters
    """
    axis_tilt: float
    axis_azimuth: float
    ground_coverage_ratio: float
    backtrack: bool
    max_rotation_angle: float


@dataclass(frozen=True)
class SolarPowerPlant(Site):
    """
    Class for keeping track of metadata associated with solar power plant
    Sites. Adds additional parameters to the Site dataclass.

    Parameters
    ----------
    modeling_parameters : PVModelingParameters
        Modeling parameters for a single axis system

    See Also
    --------
    Site
    SingleAxisModelingParameters
    FixedTiltModelingParameters
    """
    modeling_parameters: PVModelingParameters = field(
        default_factory=PVModelingParameters)


def __set_units__(cls):
    if cls.variable not in ALLOWED_VARIABLES:
        raise ValueError('variable %s is not allowed' % cls.variable)
    object.__setattr__(cls, 'units', ALLOWED_VARIABLES[cls.variable])


@dataclass(frozen=True)
class Observation(BaseModel):
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
    interval_value_type : str
        The type of the data in the observation. Typically interval mean or
        instantaneous, but additional types may be defined for events.
    interval_length : pandas.Timedelta
        The length of time between consecutive data points, e.g. 5 minutes,
        1 hour.
    interval_label : str
        Indicates if a time labels the beginning or the ending of an interval
        average, or indicates an instantaneous value, e.g. beginning, ending,
        instant
    site : Site
        The site that this Observation was generated for.
    uncertainty : float
        A measure of the uncertainty of the observation values. The format
        will be determined later.
    observation_id : str, optional
        UUID of the observation in the API
    description : str, optional
        A text description of the observation.
    extra_parameters : str, optional
        Any extra parameters for the observation

    See Also
    --------
    Site
    """
    name: str
    variable: str
    interval_value_type: str
    interval_length: pd.Timedelta
    interval_label: str
    site: Site
    uncertainty: float
    observation_id: str = ''
    description: str = ''
    extra_parameters: str = ''
    units: str = field(init=False)
    __post_init__ = __set_units__


@dataclass(frozen=True)
class Forecast(BaseModel):
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
        The length of time between consecutive data points, e.g. 5 minutes,
        1 hour.
    run_length : pandas.Timedelta
        The total length of a single issued forecast run, e.g. 1 hour.
        To enforce a continuous, non-overlapping sequence, this is equal
        to the forecast run issue frequency.
    interval_label : str
        Indicates if a time labels the beginning or the ending of an interval
        average, or indicates an instantaneous value, e.g. beginning, ending,
        instant.
    interval_value_type : str
        The type of the data in the forecast, e.g. mean, max, 95th percentile.
    variable : str
        The variable in the forecast, e.g. power, GHI, DNI. Each variable is
        associated with a standard unit.
    site : Site
        The predefined site that the forecast is for, e.g. Power Plant X
        or Aggregate Y.
    forecast_id : str, optional
        UUID of the forecast in the API
    extra_parameters : str, optional
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
    interval_value_type: str
    variable: str
    site: Site
    forecast_id: str = ''
    extra_parameters: str = ''
    units: str = field(init=False)
    __post_init__ = __set_units__
