"""
Data classes and acceptable variables as defined by the SolarForecastArbiter
Data Model document. Python 3.7 is required.
"""
from dataclasses import (dataclass, field, fields, MISSING, asdict,
                         replace, is_dataclass)
import datetime
import itertools
import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from typing import Tuple, Union, ClassVar


import numpy as np
import pandas as pd


from solarforecastarbiter.metrics.deterministic import \
    _MAP as deterministic_mapping
from solarforecastarbiter.metrics.deterministic import \
    _FILL_OPTIONS, _COST_FUNCTION_MAP, _AGG_OPTIONS
from solarforecastarbiter.metrics.event import _MAP as event_mapping
from solarforecastarbiter.metrics.probabilistic import \
    _MAP as probabilistic_mapping
from solarforecastarbiter.metrics.summary import \
    _MAP as summary_mapping
from solarforecastarbiter.validation.quality_mapping import \
    DESCRIPTION_MASK_MAPPING, DERIVED_MASKS


DASH_URL = 'https://dashboard.solarforecastarbiter.org'
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
    'event': 'boolean',
    'net_load': 'MW'
}


COMMON_NAMES = {
    'air_temperature': 'Air Temperature',
    'wind_speed': 'Wind Speed',
    'ghi': 'GHI',
    'dni': 'DNI',
    'dhi': 'DHI',
    'poa_global': 'Plane of Array Irradiance',
    'relative_humidity': 'Relative Humidty',
    'ac_power': 'AC Power',
    'dc_power': 'DC Power',
    'availability': 'Availability',
    'curtailment': 'Curtailment',
    'event': 'Event',
    'net_load': 'Net Load',
}


CLOSED_MAPPING = {
    'event': None,
    'instant': None,
    'beginning': 'left',
    'ending': 'right'
}
ALLOWED_INTERVAL_LABELS = tuple(CLOSED_MAPPING.keys())
ALLOWED_INTERVAL_VALUE_TYPES = (
    'interval_mean', 'interval_max', 'interval_min', 'interval_median',
    'instantaneous')
ALLOWED_AGGREGATE_TYPES = ('sum', 'mean', 'median', 'max', 'min', 'std')


# Keys are the categories passed to pandas groupby, values are the human
# readable versions for plotting and forms.
ALLOWED_CATEGORIES = {
    'total': 'Total',
    'year': 'Year',
    'season': 'Season',
    'month': 'Month of the year',
    'hour': 'Hour of the day',
    'date': 'Date',
    'weekday': 'Day of the week'
}


# sentences/paragraphs that will appear in the report
# under the heading in the key
CATEGORY_BLURBS = {
    'total': "Metric totals for the entire selected period.",
    'year': "Metrics per year.",
    'season': "Metrics per season.",
    'month': "Metrics per month.",
    'hour': "Metrics per hour of the day.",
    'date': "Metrics per individual date.",
    'weekday': "Metrics per day of the week."
}


ALLOWED_DETERMINISTIC_METRICS = {
    k: v[1] for k, v in deterministic_mapping.items()}

ALLOWED_EVENT_METRICS = {k: v[1] for k, v in event_mapping.items()}

ALLOWED_PROBABILISTIC_METRICS = {
    k: v[1] for k, v in probabilistic_mapping.items()}

ALLOWED_METRICS = ALLOWED_DETERMINISTIC_METRICS.copy()
ALLOWED_METRICS.update(ALLOWED_PROBABILISTIC_METRICS)
ALLOWED_METRICS.update(ALLOWED_EVENT_METRICS)
ALLOWED_SUMMARY_STATISTICS = {
    k: v[1] for k, v in summary_mapping.items()}


ALLOWED_COST_FUNCTIONS = tuple(_COST_FUNCTION_MAP.keys())
ALLOWED_COST_AGG_OPTIONS = tuple(_AGG_OPTIONS.keys())
ALLOWED_COST_FILL_OPTIONS = tuple(_FILL_OPTIONS.keys())

ALLOWED_QUALITY_FLAGS = tuple([
    k for k in DESCRIPTION_MASK_MAPPING.keys()
    if not k.startswith('VERSION IDENTIFIER') and
    not k.startswith("RESERVED")
]) + tuple(DERIVED_MASKS.keys())


def _time_conv(inp):
    if isinstance(inp, datetime.time):
        return inp.strftime('%H:%M')
    elif isinstance(inp, datetime.datetime):
        return inp.isoformat()
    elif isinstance(inp, pd.Timedelta):
        # convert to integer minutes
        return inp.total_seconds() // 60
    else:
        return inp


def _dict_factory(inp):
    dict_ = {}
    for k, v in dict(inp).items():
        if isinstance(v, tuple):
            dict_[k] = tuple(_time_conv(i) for i in v)
        elif isinstance(v, list):  # pragma: no cover
            dict_[k] = [_time_conv(i) for i in v]
        else:
            dict_[k] = _time_conv(v)
    if 'units' in dict_:
        del dict_['units']
    if 'constant_value_units' in dict_:
        del dict_['constant_value_units']
    if 'data_object' in dict_:
        del dict_['data_object']
    return dict_


def _single_field_processing(model, field, val, field_type=None):
    type_ = field_type or field.type
    if (
            # If the value is already the right type, return
            # typing type_s do not work with isinstance, so check __origin__
            not hasattr(type_, '__origin__') and
            isinstance(val, type_)
    ):
        return val
    elif type_ == pd.Timedelta:
        return pd.Timedelta(f'{val}min')
    elif type_ == pd.Timestamp:
        out = pd.Timestamp(val)
        if pd.isna(out):
            raise ValueError(f'{val} is not a time')
        return out
    elif type_ == datetime.time:
        return datetime.datetime.strptime(val, '%H:%M').time()
    elif (
            is_dataclass(type_) and
            isinstance(val, dict)
    ):
        return type_.from_dict(val)
    elif (
            hasattr(type_, '__origin__') and
            type_.__origin__ is Union
    ):
        # with a Union, we must return the right type
        for ntype in type_.__args__:
            try:
                processed_val = _single_field_processing(
                    model, field, val, ntype
                )
            except (TypeError, ValueError, KeyError):
                continue
            else:
                if not isinstance(processed_val, ntype):
                    continue
                else:
                    return processed_val
        raise TypeError(f'Unable to process {val} as one of {type_.__args__}')
    else:
        return model._special_field_processing(
            model, field, val)


class BaseModel:
    def _special_field_processing(self, model_field, val):
        return val

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        """
        Construct a dataclass from the given dict, matching keys with
        the class fields. A KeyError is raised for any missing values.
        If raise_on_extra is True, an errors is raised if keys of the
        dict are also not fields of the dataclass. For pandas.Timedelta
        model fields, it is assumed input_dict contains a number
        representing minutes. For datetime.time model fields, input_dict
        values are assumed to be strings in the %H:%M format. If a
        modeling_parameters field is present, the modeling_parameters
        key from input_dict is automatically parsed into the appropriate
        PVModelingParameters subclass based on tracking_type.

        Parameters
        ----------
        input_dict : dict
            The dict to process into dataclass fields
        raise_on_extra : boolean, default False
            If True, raise an exception on extra keys in input_dict that
            are not dataclass fields.

        Returns
        -------
        model : subclass of BaseModel
            Instance of the desired model.

        Raises
        ------
        KeyError
            For missing required fields or if raise_on_extra is True and
            input_dict contains extra keys.
        ValueError
            If a pandas.Timedelta, pandas.Timestamp, datetime.time, or
            modeling_parameters field cannot be parsed from the input_dict
        TypeError
            If the field has a Union type and the input parameter is not
            processed into one of the Union arguments
        """
        dict_ = input_dict.copy()
        model_fields = fields(model)
        kwargs = {}
        errors = []
        for model_field in model_fields:
            if model_field.name in dict_:
                field_val = dict_[model_field.name]
                if (
                        hasattr(model_field.type, '__origin__') and
                        model_field.type.__origin__ is tuple
                ):
                    out = []
                    default_type = model_field.type.__args__[0]
                    for i, arg in enumerate(field_val):
                        if (
                                i < len(model_field.type.__args__) and
                                model_field.type.__args__[i] is not Ellipsis
                        ):
                            this_type = model_field.type.__args__[i]
                        else:
                            this_type = default_type

                        out.append(
                            _single_field_processing(
                                model, model_field, arg, this_type))
                    kwargs[model_field.name] = tuple(out)
                else:
                    kwargs[model_field.name] = _single_field_processing(
                        model, model_field, field_val)
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

    def to_dict(self):
        """
        Convert the dataclass into a dictionary suitable for uploading to the
        API. This means some types (such as pandas.Timedelta and times) are
        converted to strings.
        """
        # using the dict_factory recurses through all objects for special
        # conversions
        dict_ = asdict(self, dict_factory=_dict_factory)
        return dict_

    def replace(self, **kwargs):
        """
        Convience wrapper for :py:func:`dataclasses.replace` to create a
        new dataclasses from the old with the given keys replaced.
        """
        return replace(self, **kwargs)


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
    extra_parameters : str, optional
        The extra parameters may be used by forecasters when
        implementing other PV models. The framework does not provide
        a standard set of extra parameters or require a particular
        format – these are up to the site owner.
    climate_zones : tuple of str, optional
        The climate zones that the site is within
    """
    name: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    site_id: str = ''
    provider: str = ''
    extra_parameters: str = ''
    climate_zones: Tuple[str, ...] = ()

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        if 'modeling_parameters' in dict_:
            mp_dict = dict_.get('modeling_parameters', {})
            if not isinstance(mp_dict, PVModelingParameters):
                tracking_type = mp_dict.get('tracking_type', None)
                if tracking_type == 'fixed':
                    dict_['modeling_parameters'] = (
                        FixedTiltModelingParameters.from_dict(
                            mp_dict))
                    return SolarPowerPlant.from_dict(dict_, raise_on_extra)
                elif tracking_type == 'single_axis':
                    dict_['modeling_parameters'] = (
                        SingleAxisModelingParameters.from_dict(
                            mp_dict))
                    return SolarPowerPlant.from_dict(dict_, raise_on_extra)
                elif tracking_type is not None:
                    raise ValueError(
                        'tracking_type must be None, fixed, or '
                        'single_axis')
        return super().from_dict(dict_, raise_on_extra)


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
        The temperature coefficient of DC power in units of %/C.
        Typically -0.2 to -0.5 % per degree C.
    dc_loss_factor : float
        Applied to DC current in units of %. 0 = no loss.
    ac_loss_factor : float
        Appled to inverter power output in units of %. 0 = no loss.

    See Also
    --------
    :py:class:`solarforecastarbiter.datamodel.FixedTiltModelingParameters`
    :py:class:`solarforecastarbiter.datamodel.SingleAxisModelingParameters`
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
    :py:class:`solarforecastarbiter.datamodel.PVModelingParameters`
    """
    surface_tilt: float
    surface_azimuth: float
    tracking_type: str = 'fixed'


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
    :py:class:`solarforecastarbiter.datamodel.PVModelingParameters`
    """
    axis_tilt: float
    axis_azimuth: float
    ground_coverage_ratio: float
    backtrack: bool
    max_rotation_angle: float
    tracking_type: str = 'single_axis'


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
    :py:class:`solarforecastarbiter.datamodel.Site`
    :py:class:`solarforecastarbiter.datamodel.SingleAxisModelingParameters`
    :py:class:`solarforecastarbiter.datamodel.FixedTiltModelingParameters`
    """
    modeling_parameters: PVModelingParameters = field(
        default_factory=PVModelingParameters)


def __set_units__(cls):
    if cls.variable not in ALLOWED_VARIABLES:
        raise ValueError('variable %s is not allowed' % cls.variable)
    object.__setattr__(cls, 'units', ALLOWED_VARIABLES[cls.variable])


def __generic_oneof__(cls, field, allowed):
    if getattr(cls, field) not in allowed:
        raise ValueError(f'{field} must be one of {allowed}')


def __check_interval_params__(cls):
    __generic_oneof__(
        cls, 'interval_label', ALLOWED_INTERVAL_LABELS)
    __generic_oneof__(
        cls, 'interval_value_type', ALLOWED_INTERVAL_VALUE_TYPES)


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
        The type of the data in the observation. Typically interval_mean or
        instantaneous, but additional types may be defined for events.
    interval_length : pandas.Timedelta
        The length of time between consecutive data points, e.g. 5 minutes,
        1 hour.
    interval_label : {'beginning', 'ending', 'instant', 'event'}
        Indicates if a time labels the beginning or the ending of an interval
        average, indicates an instantaneous value, or indicates an event.
    site : Site
        The site that this Observation was generated for.
    uncertainty : float or None
        A measure of the uncertainty of the observation values. The format
        will be determined later. None indicates that the uncertainty is
        unknown for this Observation.
    observation_id : str, optional
        UUID of the observation in the API
    provider : str, optional
        Provider of the Observation information.
    extra_parameters : str, optional
        Any extra parameters for the observation

    See Also
    --------
    :py:class:`solarforecastarbiter.datamodel.Site`
    """
    __blurb__: ClassVar[str] = 'Observation'
    name: str
    variable: str
    interval_value_type: str
    interval_length: pd.Timedelta
    interval_label: str
    site: Site
    uncertainty: Union[float, None]
    observation_id: str = ''
    provider: str = ''
    extra_parameters: str = ''
    units: str = field(init=False)

    def __post_init__(self):
        __set_units__(self)
        __check_interval_params__(self)


@dataclass(frozen=True)
class AggregateObservation(BaseModel):
    """
    Class for keeping track of an Observation and when it is added and
    (optionally) removed from an Aggregate. This metadata allows the
    Arbiter to calculate the correct quantities while the Aggregate grows
    or shrinks over time.

    Parameters
    ----------
    observation : Observation
        The Observation object that is part of the Aggregate
    effective_from : pandas.Timestamp
        The datetime of when the Observation should be
        included in the Aggregate
    effective_until : pandas.Timestamp
        The datetime of when the Observation should be
        excluded from the Aggregate
    observation_deleted_at : pandas.Timestamp
        The datetime that the Observation was deleted from the
        Arbiter. This indicates that the Observation should be
        removed from the Aggregate, and without the data
        from this Observation, the Aggregate is invalid before
        this time.

    See Also
    --------
    :py:class:`solarforecastarbiter.datamodel.Observation`
    :py:class:`solarforecastarbiter.datamodel.Aggregate`
    """
    __blurb__: ClassVar[str] = 'Aggregate Observation'
    observation: Observation
    effective_from: pd.Timestamp
    effective_until: Union[pd.Timestamp, None] = None
    observation_deleted_at: Union[pd.Timestamp, None] = None


def __check_variable__(variable, *args):
    if not all(arg.variable == variable for arg in args):
        raise ValueError('All variables must be identical.')


def __check_aggregate_interval_compatibility__(interval, *args):
    if any(arg.interval_length > interval for arg in args):
        raise ValueError('observation.interval_length cannot be greater than '
                         'aggregate.interval_length.')
    if any(arg.interval_value_type not in ('interval_mean', 'instantaneous')
           for arg in args):
        raise ValueError('Only observations with interval_value_type of '
                         'interval_mean or instantaneous are acceptable')


@dataclass(frozen=True)
class Aggregate(BaseModel):
    """
    Class for keeping track of Aggregate metadata. Aggregates always
    have interval_value_type of 'interval_mean'.

    Parameters
    ----------
    name : str
        Name of the Aggregate, e.g. Utility X Solar PV
    description : str
        A description of what the aggregate is.
    variable : str
        Variable name, e.g. power, GHI. Each allowed variable has an
        associated pre-defined unit. All observations that make up the
        Aggregate must also have this variable.
    aggregate_type : str
        The aggregation function that will be applied to observations.
        Generally, this will be 'sum' although one might be interested,
        for example, in the 'mean' irradiance of some observations.
        May be an aggregate function string supported by Pandas. Common
        options include ('sum', 'mean', 'min', 'max', 'median', 'std').
    interval_length : pandas.Timedelta
        The length of time between consecutive data points, e.g. 5 minutes,
        1 hour. This must be >= the interval lengths of any Observations that
        will make up the Aggregate.
    interval_label : str
        Indicates if a time labels the beginning or the ending of an interval
        average.
    timezone : str
        IANA timezone of the Aggregate, e.g. Etc/GMT+8
    aggregate_id : str, optional
        UUID of the Aggregate in the API
    provider : str, optional
        Provider of the Aggregate information.
    extra_parameters : str, optional
        Any extra parameters for the Aggregate.
    observations : tuple of AggregateObservation
        The Observations that contribute to the Aggregate

    See Also
    --------
    :py:class:`solarforecastarbiter.datamodel.Observation`
    """
    __blurb__: ClassVar[str] = 'Aggregate'
    name: str
    description: str
    variable: str
    aggregate_type: str
    interval_length: pd.Timedelta
    interval_label: str
    timezone: str
    observations: Tuple[AggregateObservation, ...]
    aggregate_id: str = ''
    provider: str = ''
    extra_parameters: str = ''
    units: str = field(init=False)
    interval_value_type: str = field(default='interval_mean')

    def __post_init__(self):
        __set_units__(self)
        observations = [
            ao.observation for ao in self.observations
            if ao.observation is not None]
        __check_variable__(
            self.variable,
            *observations)
        __check_aggregate_interval_compatibility__(
            self.interval_length,
            *observations)
        __generic_oneof__(self, 'aggregate_type', ALLOWED_AGGREGATE_TYPES)
        __generic_oneof__(self, 'interval_label', ('beginning', 'ending'))
        object.__setattr__(self, 'interval_value_type', 'interval_mean')


@dataclass(frozen=True)
class _ForecastBase:
    name: str
    issue_time_of_day: datetime.time
    lead_time_to_start: pd.Timedelta
    interval_length: pd.Timedelta
    run_length: pd.Timedelta
    interval_label: str
    interval_value_type: str
    variable: str


@dataclass(frozen=True)
class _ForecastDefaultsBase:
    site: Union[Site, None] = None
    aggregate: Union[Aggregate, None] = None
    forecast_id: str = ''
    provider: str = ''
    extra_parameters: str = ''
    units: str = field(init=False)


def __site_or_agg__(cls):
    if cls.site is not None and cls.aggregate is not None:
        raise KeyError('Only provide one of "site" or "aggregate" to Forecast')
    elif cls.site is None and cls.aggregate is None:
        raise KeyError('Must provide one of "site" or "aggregate" to Forecast')


# Follow MRO pattern in https://stackoverflow.com/a/53085935/2802993
# to avoid problems with inheritance in ProbabilisticForecasts
@dataclass(frozen=True)
class Forecast(BaseModel, _ForecastDefaultsBase, _ForecastBase):
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
        issue frequency attribute. This is assumed to be a UTC time.
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
    interval_label : {"beginning", "ending", "instant"}
        Indicates if a time labels the beginning or the ending of an interval
        average, or indicates an instantaneous value, e.g. beginning, ending,
        instant.
    interval_value_type : str
        The type of the data in the forecast, e.g. mean, max, 95th percentile.
    variable : str
        The variable in the forecast, e.g. power, GHI, DNI, event. Each
        variable is associated with a standard unit.
    site : Site or None
        The predefined site that the forecast is for, e.g. Power Plant X.
    aggregate : Aggregate or None
        The predefined aggregate that the forecast is for, e.g. Aggregate Y.
    forecast_id : str, optional
        UUID of the forecast in the API
    provider : str, optional
        Provider of the Forecast information.
    extra_parameters : str, optional
        Extra configuration parameters of forecast.

    See Also
    --------
    :py:class:`solarforecastarbiter.datamodel.Site`
    :py:class:`solarforecastarbiter.datamodel.Aggregate`
    """
    __blurb__: ClassVar[str] = 'Forecast'

    def __post_init__(self):
        __set_units__(self)
        __site_or_agg__(self)
        __check_interval_params__(self)

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        if model != Forecast:
            return super().from_dict(dict_, raise_on_extra)
        if dict_.get('constant_value', None) is not None:
            return ProbabilisticForecastConstantValue.from_dict(
                dict_, raise_on_extra)
        elif dict_.get('constant_values', None) is not None:
            return ProbabilisticForecast.from_dict(dict_, raise_on_extra)
        elif dict_.get('variable') == 'event':
            return EventForecast.from_dict(dict_, raise_on_extra)
        else:
            return super().from_dict(dict_, raise_on_extra)


@dataclass(frozen=True)
class EventForecast(Forecast):
    """
    Extends Forecast dataclass to include event forecast attributes.

    Parameters
    ----------
    name : str
        Name of the Forecast
    issue_time_of_day : datetime.time
        The time of day that a forecast run is issued, e.g. 00:30. For
        forecast runs issued multiple times within one day (e.g. hourly),
        this specifies the first issue time of day. Additional issue times
        are uniquely determined by the first issue time and the run length &
        issue frequency attribute. This is assumed to be a UTC time.
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
    interval_label : {'event'}
    interval_value_type : str
        The type of the data in the forecast, e.g. mean, max, 95th percentile.
    variable : {'event'}
    site : Site or None
        The predefined site that the forecast is for, e.g. Power Plant X.
    aggregate : Aggregate or None
        The predefined aggregate that the forecast is for, e.g. Aggregate Y.
    forecast_id : str, optional
        UUID of the forecast in the API
    provider : str, optional
        Provider of the Forecast information.
    extra_parameters : str, optional
        Extra configuration parameters of forecast.

    See also
    --------
    :py:class:`solarforecastarbiter.datamodel.Forecast`
    """
    __blurb__: ClassVar[str] = 'Event Forecast'

    def __post_init__(self):
        if self.interval_label != "event":
            raise ValueError("Interval label must be 'event'")
        elif self.variable != "event":
            raise ValueError("Variable must be 'event'")
        super().__post_init__()


def __set_constant_value_units__(cls):
    if cls.axis == 'x':
        # e.g. Prob(o < 10 MW). Forecast is in %, constant value is 10 MW
        object.__setattr__(cls, 'constant_value_units', cls.units)
        object.__setattr__(cls, 'units', '%')
    else:
        # e.g. Prob(o < f) = 90%. Forecast in units of obs, constant value is %
        object.__setattr__(cls, 'constant_value_units', '%')


@dataclass(frozen=True)
class _ProbabilisticForecastConstantValueBase:
    axis: str
    constant_value: float
    constant_value_units: str = field(init=False)


@dataclass(frozen=True)
class ProbabilisticForecastConstantValue(
        Forecast, _ProbabilisticForecastConstantValueBase):
    """
    Extends Forecast dataclass to include probabilistic forecast
    attributes.

    Parameters
    ----------
    name : str
        Name of the Forecast
    issue_time_of_day : datetime.time
        The time of day that a forecast run is issued, e.g. 00:30. For
        forecast runs issued multiple times within one day (e.g. hourly),
        this specifies the first issue time of day. Additional issue times
        are uniquely determined by the first issue time and the run length &
        issue frequency attribute. This is assumed to be a UTC time.
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
    site : Site or None
        The predefined site that the forecast is for, e.g. Power Plant X.
    aggregate : Aggregate or None
        The predefined aggregate that the forecast is for, e.g. Aggregate Y.
    axis : str
        The axis on which the constant values of the CDF is specified.
        The axis can be either *x* (constant variable values) or *y*
        (constant percentiles).
    constant_value : float
        The variable value or percentile.
    forecast_id : str, optional
        UUID of the forecast in the API
    provider : str, optional
        Provider of the ProbabilisticForecastConstantValue information.
    extra_parameters : str, optional
        Extra configuration parameters of forecast.

    See also
    --------
    :py:class:`solarforecastarbiter.datamodel.ProbabilisticForecast`
    """
    __blurb__: ClassVar[str] = 'Probabilistic Forecast Constant Value'

    def __post_init__(self):
        super().__post_init__()
        __check_axis__(self.axis)
        __set_constant_value_units__(self)


@dataclass(frozen=True)
class _ProbabilisticForecastBase:
    axis: str
    constant_values: Tuple[Union[ProbabilisticForecastConstantValue, float, int], ...]  # NOQA
    constant_value_units: str = field(init=False)


@dataclass(frozen=True)
class ProbabilisticForecast(
        Forecast, _ProbabilisticForecastBase):
    """
    Tracks a group of ProbabilisticForecastConstantValue objects that
    together describe 1 or more points of the same probability
    distribution.

    Parameters
    ----------
    name : str
        Name of the Forecast
    issue_time_of_day : datetime.time
        The time of day that a forecast run is issued, e.g. 00:30. For
        forecast runs issued multiple times within one day (e.g. hourly),
        this specifies the first issue time of day. Additional issue times
        are uniquely determined by the first issue time and the run length &
        issue frequency attribute. This is assumed to be a UTC time.
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
    site : Site or None
        The predefined site that the forecast is for, e.g. Power Plant X.
    aggregate : Aggregate or None
        The predefined aggregate that the forecast is for, e.g. Aggregate Y.
    axis : str
        The axis on which the constant values of the CDF is specified.
        The axis can be either *x* (constant variable values) or *y*
        (constant percentiles).
    constant_values : tuple of ProbabilisticForecastConstantValue or float
        The variable values or percentiles. Floats will automatically
        be converted to ProbabilisticForecastConstantValue objects.
    forecast_id : str, optional
        UUID of the forecast in the API
    provider : str, optional
        Provider of the ProbabilisticForecast information.
    extra_parameters : str, optional
        Extra configuration parameters of forecast.

    See also
    --------
    ProbabilisticForecastConstantValue
    Forecast
    """
    __blurb__: ClassVar[str] = 'Probabilistic Forecast'

    def __post_init__(self):
        super().__post_init__()
        __check_axis__(self.axis)
        __set_constant_value_units__(self)
        __set_constant_values__(self)
        __check_axis_consistency__(self.axis, self.constant_values)


def __validate_cost__(index_var):
    def val(obj):
        if hasattr(obj, 'fill'):
            fillkeys = ALLOWED_COST_FILL_OPTIONS
            if obj.fill not in fillkeys:
                raise ValueError(
                    f"Cost 'fill' must be one of {str(fillkeys)}")
        if hasattr(obj, 'aggregation'):
            aggkeys = ALLOWED_COST_AGG_OPTIONS
            if obj.aggregation not in aggkeys:
                raise ValueError(
                    f"Cost 'aggregation' must be one of {str(aggkeys)}")
        if index_var is not None:
            if len(obj.cost) != len(getattr(obj, index_var)):
                raise ValueError(
                    f"'cost' and '{index_var}' must have the same length")
    return val


@dataclass(frozen=True)
class TimeOfDayCost(BaseModel):
    """Cost values based on the time of day.

    Parameters
    ----------
    times : tuple of datetime.time
        The times to associate with each cost value
    cost : tuple of float
        The cost per unit error of the forecasted variable for each time.
        Must have the same length as `times`.
    aggregation : str
        Aggregation method to use after calculating cost for the error series.
        Currently only 'sum' or 'mean' are available.
    net : bool
        If True, compute the 'net' aggregate error instead of first calcuating
        the absolute error before performing the aggregation.
    fill : str
        Fill method to apply for times between those specified in `times`.
        Options are 'forward' or 'backward'.
    timezone : str, default None
        IANA timezone string to use when constructing datetimes. If None,
        the timezone of the observations is used, which is the report
        timezone when calculated in a report.
    """
    times: Tuple[datetime.time, ...]
    cost: Tuple[float, ...]
    aggregation: str
    net: bool
    fill: str
    timezone: str = None
    __post_init__ = __validate_cost__('times')


@dataclass(frozen=True)
class DatetimeCost(BaseModel):
    """Cost values based on datetimes.

    Parameters
    ----------
    datetimes : tuple/iterable of datetime-like objects
       The datetimes to associate with each cost value
    cost : tuple of float
       The cost per unit error of the forecasted variable for each datetime.
       Must have the same length as `datetimes`.
    aggregation : str
        Aggregation method to use after calculating cost for the error series.
        Currently only 'sum' or 'mean' are available.
    net : bool
        If True, compute the 'net' aggregate error instead of first calcuating
        the absolute error before performing the aggregation.
    fill : str
       Fill method to apply for datetimes between those specified in
       `datetimes`. Options are 'forward' or 'backward'.
    timezone : str, default None
        IANA timezone string to use when constructing datetimes. If None,
        the timezone of the observations is used, which is the report
        timezone when calculated in a report.
        """
    datetimes: Tuple[pd.Timestamp, ...]
    cost: Tuple[float, ...]
    aggregation: str
    net: bool
    fill: str
    timezone: str = None
    __post_init__ = __validate_cost__('datetimes')


@dataclass(frozen=True)
class ConstantCost(BaseModel):
    """A constant cost per unit error of the forecasted variable

    Parameters
    ----------
    cost : float
    aggregation : str
        Aggregation method to use after calculating cost for the error series.
        Currently only 'sum' or 'mean' are available.
    net : bool
        If True, compute the 'net' aggregate error instead of first calcuating
        the absolute error before performing the aggregation.
    """
    cost: float
    aggregation: str
    net: bool
    __post_init__ = __validate_cost__(None)


@dataclass(frozen=True)
class CostBand(BaseModel):
    """Cost specification for one error band

    Parameters
    ----------
    error_range : tuple(float, float)
        Bounds of the error to apply the specified cost function to.
        Inf and -Inf are valid range points, and the error may be positive or
        negative. Inclusion/exclusion of endpoints is determined by ordering
        in :py:class:`solarforecastarbiter.datamodel.ErrorBandCost`.
    cost_function : str
        One of 'timeofday', 'datetime', or 'constant'. Specifies which
        cost model should be used to calculate the cost in this band.
    cost_function_parameters : :py:class:`solarforecastarbiter.datamodel.ConstantCost` or :py:class:`solarforecastarbiter.TimeOfDayCost` or :py:class:`solarforecastarbiter.DatetimeCost`
        Parameters for the selected cost function.
    """  # NOQA: E501
    error_range: Tuple[float, float]
    cost_function: str
    cost_function_parameters: Union[TimeOfDayCost, DatetimeCost, ConstantCost]

    def _special_field_processing(self, model_field, val):
        # support passing "inf", "-inf" as strings via json/dict
        if model_field.name == 'error_range':
            return float(val)
        else:  # pragma: no cover
            return val

    def __post_init__(self):
        if self.cost_function == 'timeofday':
            if not isinstance(self.cost_function_parameters, TimeOfDayCost):
                raise TypeError(
                    "'cost_function_parameters' must be of type TimeOfDayCost "
                    "for 'timeofday' cost function.")
        elif self.cost_function == 'datetime':
            if not isinstance(self.cost_function_parameters, DatetimeCost):
                raise TypeError(
                    "'cost_function_parameters' must be of type DatetimeCost "
                    "for 'datetime' cost function.")
        elif self.cost_function == 'constant':
            if not isinstance(self.cost_function_parameters, ConstantCost):
                raise TypeError(
                    "'cost_function_parameters' must be of type ConstantCost "
                    "for 'constant' cost function.")
        else:
            raise ValueError(
                "'cost_function' must be one of 'timeofday', 'datetime', or"
                " 'constant'")


@dataclass(frozen=True)
class ErrorBandCost(BaseModel):
    """Cost that varies based on the error value. For each error band,
    one of the other cost functions is applied to the errors within the band.
    If an error value does not fall within any band ranges, no cost is
    calculated for that error.

    Parameters
    ----------
    bands : tuple of :py:class:`solarforecastarbiter.datamodel.CostBand`
       Specification of the error bands and associated cost functions.

    Notes
    -----
    Each error is restricted to a single band/cost function, so the
    order in bands determines which band is applied in ascending
    priority. For example, if ``bands[0].error_range = (0, 2)``
    and ``bands[1].error_range == (1, 3)``, the cost function of
    bands[0] is applied for all errors from [0, 2] and bands[1]
    is applied for errors from (2, 3].
    """
    bands: Tuple[CostBand, ...]


@dataclass(frozen=True)
class Cost(BaseModel):
    """Specify how cost metrics should be calculated.

    Parameters
    ----------
    name : str
        Identifier for these cost parameters
    type : str
       The type of cost parameters that are included in `parameters`. One of
       'timeofday', 'datetime', 'constant', or 'errorband'.
    parameters : :py:class:`solarforecastarbiter.datamodel.ConstantCost` or :py:class:`solarforecastarbiter.TimeOfDayCost` or :py:class:`solarforecastarbiter.DatetimeCost` or :py:class:`solarforecastarbiter.ErrorBandCost`
        Parameters for the specific cost function type.
    """  # NOQA: E501
    name: str
    type: str
    parameters: Union[TimeOfDayCost, DatetimeCost, ConstantCost, ErrorBandCost]

    def __post_init__(self):
        if self.type not in ALLOWED_COST_FUNCTIONS:
            raise ValueError(
                f"'type' must be one of {ALLOWED_COST_FUNCTIONS}")

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        type_ = dict_['type']
        param_dict = dict_.get('parameters', {})
        if type_ == 'timeofday':
            dict_['parameters'] = TimeOfDayCost.from_dict(param_dict)
        elif type_ == 'datetime':
            dict_['parameters'] = DatetimeCost.from_dict(param_dict)
        elif type_ == 'constant':
            dict_['parameters'] = ConstantCost.from_dict(param_dict)
        elif type_ == 'errorband':
            dict_['parameters'] = ErrorBandCost.from_dict(param_dict)
        else:
            raise ValueError(
                f"'type' must be one of {ALLOWED_COST_FUNCTIONS}")
        return super().from_dict(dict_, raise_on_extra)


def __set_constant_values__(self):
    out = []
    for cv in self.constant_values:
        if isinstance(cv, ProbabilisticForecastConstantValue):
            out.append(cv)
        elif isinstance(cv, (float, int)):
            cv_dict = self.to_dict()
            cv_dict.pop('forecast_id', None)
            cv_dict['constant_value'] = cv
            out.append(
                ProbabilisticForecastConstantValue.from_dict(cv_dict))
        else:
            raise TypeError(
                f'Invalid type for a constant value {cv}. '
                'Must be int, float, or ProbablisticConstantValue')
    object.__setattr__(self, 'constant_values', tuple(out))


def __check_axis__(axis):
    if axis not in ('x', 'y'):
        raise ValueError('Axis must be x or y')


def __check_axis_consistency__(axis, constant_values):
    if not all(arg.axis == axis for arg in constant_values):
        raise ValueError('All axis attributes must be identical')


def __check_units__(*args):
    if len(args) == 0:
        return
    unique_units = set()
    for arg in args:
        if getattr(arg, 'axis', None) == 'x':
            unique_units.add(arg.constant_value_units)
        else:
            unique_units.add(arg.units)
    if len(unique_units) > 1:
        raise ValueError('All units must be identical.')


def __check_interval_compatibility__(forecast, observation):
    if observation.interval_length > forecast.interval_length:
        raise ValueError('observation.interval_length cannot be greater than '
                         'forecast.interval_length.')
    if ('instant' in forecast.interval_label and
            'instant' not in observation.interval_label):
        raise ValueError('Instantaneous forecasts cannot be evaluated against '
                         'interval average observations.')


@dataclass(frozen=True)
class ForecastObservation(BaseModel):
    """
    Class for pairing Forecast and Observation objects for evaluation.

    Parameters
    ----------
    forecast: :py:class:`solarforecastarbiter.datamodel.Forecast`
    observation: :py:class:`solarforecastarbiter.datamodel.Observation`
    reference_forecast: :py:class:`solarforecastarbiter.datamodel.Forecast` or None
    normalization: float or None
        If None, determined by __set_normalization__
    uncertainty: None, float, or str
        If None, uncertainty is not accounted for. Float specifies the
        uncertainty as a percentage from 0 to 100%. If str, may be
        'observation_uncertainty' to indicate that the value should be
        set to ``observation.uncertainty``, or may be coerceable to a
        float.
    cost: str or None
        Cost parameters to use from the costs associated with ReportParameters
    """  # NOQA
    forecast: Forecast
    observation: Observation
    reference_forecast: Union[Forecast, None] = None
    # some function applied to observation (e.g. mean per day)
    # possible in future. maybe add pd.Series like for
    # ProcessedForecastObservation
    normalization: Union[float, None] = None
    uncertainty: Union[None, float, str] = None
    cost: Union[str, None] = None
    data_object: Observation = field(init=False)

    def __post_init__(self):
        __set_normalization__(self)
        __set_uncertainty__(self)
        object.__setattr__(self, 'data_object', self.observation)
        __check_units__(self.forecast, self.data_object)
        __check_interval_compatibility__(self.forecast, self.data_object)


def __set_normalization__(self):
    if self.normalization is None:
        if self.observation.variable == 'ac_power':
            norm = self.observation.site.modeling_parameters.ac_capacity
        elif self.observation.variable == 'dc_power':
            norm = self.observation.site.modeling_parameters.dc_capacity
        elif self.observation.units == 'W/m^2':
            # normalizing by 1000 W/m^2 was considered and rejected
            # https://github.com/SolarArbiter/solarforecastarbiter-core/pull/379#discussion_r402434134
            # keep W/m^2 as separate item for likely future improvements
            norm = np.nan
        else:
            norm = np.nan
    else:
        # norm was supplied, but we're going to make sure it can coerced
        # to a float
        norm = self.normalization
    norm = float(norm)  # from_dict only checks for floats, chokes on ints
    object.__setattr__(self, 'normalization', norm)


def __set_aggregate_normalization__(self):
    # https://github.com/SolarArbiter/solarforecastarbiter-core/issues/381
    norm = np.nan
    object.__setattr__(self, 'normalization', norm)


def __set_uncertainty__(self):
    if isinstance(self.uncertainty, str):
        try:
            unc = float(self.uncertainty)
        except ValueError:
            if self.uncertainty == 'observation_uncertainty':
                object.__setattr__(
                    self, 'uncertainty', self.observation.uncertainty)
            else:
                # easy to mistype 'observation_uncertainty', so be helpful
                raise ValueError(
                    ('Invalid uncertainty %s. uncertainty must be set to None,'
                     ' a float, or "observation_uncertainty"') %
                    self.uncertainty)
        else:
            object.__setattr__(self, 'uncertainty', unc)


@dataclass(frozen=True)
class ForecastAggregate(BaseModel):
    """
    Class for pairing Forecast and Aggregate objects for evaluation.

    Parameters
    ----------
    forecast: :py:class:`solarforecastarbiter.datamodel.Forecast`
    aggregate: :py:class:`solarforecastarbiter.datamodel.Aggregate`
    reference_forecast: :py:class:`solarforecastarbiter.datamodel.Forecast` or None
    normalization: float or None
        If None, assigned 1.
    uncertainty: None, float, or str
        If None, uncertainty is not accounted for. Float specifies the
        uncertainty as a percentage from 0 to 100%. Strings must be
        coerceable to a float.
    cost: str or None
        Cost parameters to use from the costs associated with ReportParameters
    """  # NOQA
    forecast: Forecast
    aggregate: Aggregate
    reference_forecast: Union[Forecast, None] = None
    normalization: Union[float, None] = None
    uncertainty: Union[float, None] = None
    cost: Union[str, None] = None
    data_object: Aggregate = field(init=False)

    def __post_init__(self):
        if self.normalization is None:
            __set_aggregate_normalization__(self)
        if self.uncertainty is not None:
            object.__setattr__(self, 'uncertainty', float(self.uncertainty))
        object.__setattr__(self, 'data_object', self.aggregate)
        __check_units__(self.forecast, self.data_object)
        __check_interval_compatibility__(self.forecast, self.data_object)


@dataclass(frozen=True)
class BaseFilter(BaseModel):
    """
    Base class for filters to be applied in a report.
    """
    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        if model != BaseFilter:
            return super().from_dict(dict_, raise_on_extra)

        if 'quality_flags' in dict_:
            return QualityFlagFilter.from_dict(dict_, raise_on_extra)
        elif 'time_of_day_range' in dict_:
            return TimeOfDayFilter.from_dict(dict_, raise_on_extra)
        elif 'value_range' in dict_:
            return ValueFilter.from_dict(dict_, raise_on_extra)
        else:
            raise NotImplementedError(
                'Do not know how to process dict into a Filter.')


@dataclass(frozen=True)
class QualityFlagFilter(BaseFilter):
    """Quality flag filters to be applied in a report.

    Parameters
    ----------
    quality_flags : Tuple of str
        Strings corresponding to ``BITMASK_DESCRIPTION_DICT`` or
        ``DERIVED_MASKS`` keys.
        These periods will be excluded from the analysis.
    discard_before_resample : bool, default True
        Determines if points should be discarded before resampling or
        only during resampling (when ``resample_threshold_percentage``
        is exceeded).
    resample_threshold_percentage : float, default 10.
        The percentage of points in a resampled interval that must be
        flagged for the resampled interval to be flagged.

    Notes
    -----
    If ``discard_before_resample`` is ``True``, the ``quality_flags``
    are applied to the data before it is resampled. During resampling,
    intervals are discarded if ``resample_threshold_percentage`` is
    exceeded.

    If ``discard_before_resample`` is ``False``, the ``quality_flags``
    are only considered during the resampling operation. The
    ``quality_flags`` of the raw observations are combined with ``OR``,
    the total number of flagged points within a resample period is
    computed, and intervals are discarded where
    ``resample_threshold_percentage`` is exceeded.
    """
    quality_flags: Tuple[str, ...] = (
        'UNEVEN FREQUENCY', 'LIMITS EXCEEDED', 'CLEARSKY EXCEEDED',
        'DAYTIME STALE VALUES', 'INCONSISTENT IRRADIANCE COMPONENTS'
    )
    discard_before_resample: bool = True
    resample_threshold_percentage: float = 10.

    def __post_init__(self):
        if not all(flag in ALLOWED_QUALITY_FLAGS
                   for flag in self.quality_flags):
            raise ValueError('Quality flags must be in '
                             'BITMASK_DESCRIPTION_DICT or DERIVED_MASKS')


@dataclass(frozen=True)
class TimeOfDayFilter(BaseFilter):
    """
    Class representing a time of day filter to be applied in a report.

    Parameters
    ----------
    time_of_day_range : (datetime.time, datetime.time) tuple
        Time of day range to calculate errors. Range is inclusive of
        both endpoints. Do not use this to exclude nighttime; instead
        set the corresponding quality_flag.
    """
    time_of_day_range: Tuple[datetime.time, datetime.time]
    # add these?
    # discard_before_resample: bool = True
    # resample_threshold_percentage: float = 10.


@dataclass(frozen=True)
class ValueFilter(BaseFilter):
    """
    Class representing an observation or forecast value filter to be
    applied in a report.

    Parameters
    ----------
    metadata : :py:class:`solarforecastarbiter.datamodel.Forecast` or :py:class:`solarforecastarbiter.datamodel.Observation`
        Object to get values for.
    value_range : (float, float) tuple
        Value range to calculate errors. Range is inclusive
        of both endpoints. Filters are applied before resampling.
    """  # NOQA
    # TODO: implement. Also add Aggregate
    metadata: Union[Observation, Forecast]
    value_range: Tuple[float, float]
    # add these?
    # discard_before_resample: bool = True
    # resample_threshold_percentage: float = 10.


def __check_metrics__(fx, metrics):
    """Validate metrics selection.

    Check that the selected metrics are valid for the given scenario (e.g.
    if deterministic forecasts, then deterministic metrics).

    Parameters
    ----------
    fx : Forecast, ProbabilisticForecast, ProbabilisticForecastConstantValue
        Forecast to be evaluated by metrics.
    metrics : Tuple of str
        Metrics to be computed in the report.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the selected metrics are not valid for the given forecast type.

    """

    if isinstance(fx, (ProbabilisticForecast,
                       ProbabilisticForecastConstantValue)):
        if not set(metrics) <= ALLOWED_PROBABILISTIC_METRICS.keys():
            raise ValueError("Metrics must be in "
                             "ALLOWED_PROBABILISTIC_METRICS.")
    elif isinstance(fx, EventForecast):
        if not set(metrics) <= ALLOWED_EVENT_METRICS.keys():
            raise ValueError("Metrics must be in "
                             "ALLOWED_EVENT_METRICS.")
    elif isinstance(fx, Forecast):
        if not set(metrics) <= ALLOWED_DETERMINISTIC_METRICS.keys():
            raise ValueError("Metrics must be in "
                             "ALLOWED_DETERMINISTIC_METRICS.")


def __check_categories__(categories):
    if not set(categories) <= ALLOWED_CATEGORIES.keys():
        raise ValueError('Categories must be in ALLOWED_CATEGORIES')


@dataclass(frozen=True)
class ValidationResult(BaseModel):
    """Store the validation result for a flag or combination of flags.

    Parameters
    ----------
    flag: str
        The quality flag(s) being recorded. See
        :py:mod:`solarforecastarbiter.validation.quality_mapping`.
    count: int
        The number of timestamps that were flagged.
    before_resample: bool
        If the flag was applied before resampling.
    """
    flag: str
    count: int
    before_resample: bool = True


@dataclass(frozen=True)
class PreprocessingResult(BaseModel):
    """Stores summary information to record preprocessing results that
    detail how data has been handled.

    Parameters
    ----------
    name: str
        The human readable name noting the process and data applied.
    count: int
        The number of timestamps that were managed in the process.
    """
    name: str
    count: int


# need apply filtering + resampling to each forecast obs pair
@dataclass(frozen=True)
class ProcessedForecastObservation(BaseModel):
    """
    Hold the processed forecast and observation data with the resampling
    parameters.

    Parameters
    ----------
    name: str
    original: :py:class:`solarforecastarbiter.datamodel.ForecastObservation` or :py:class:`solarforecastarbiter.ForecastAggregate`
    interval_value_type: str
    interval_length: pd.Timedelta
    interval_label: str
    valid_point_count: int
        The number of valid points in the processed forecast.
    forecast_values: pandas.Series or str or None
        The values of the forecast, the forecast id or None.
    observation_values: pandas.Series or str or None
        The values of the observation, the observation or aggregated id, or
        None.
    reference_forecast_values: pandas.Series or str or None
        The values of the reference forecast, the reference forecast id or
        None.
    validation_results: tuple of :py:class:`solarforecastarbiter.datamodel.ValidationResult`
    preprocessing_results: tuple of :py:class:`solarforecastarbiter.datamodel.PreprocessingResult`
    normalization_factor: pandas.Series or Float
    uncertainty: None or float
        If None, uncertainty is not accounted for. Float specifies the
        uncertainty as a percentage from 0 to 100%.
    cost: :py:class:`solarforecastarbiter.datamodel.Cost` or None
        The parameters to use when calculating cost metrics.
    """  # NOQA
    name: str
    # do this instead of subclass to compare objects later
    original: Union[ForecastObservation, ForecastAggregate]
    interval_value_type: str
    interval_length: pd.Timedelta
    interval_label: str
    valid_point_count: int
    forecast_values: Union[pd.Series, str, None]
    observation_values: Union[pd.Series, str, None]
    reference_forecast_values: Union[pd.Series, str, None] = None
    validation_results: Tuple[ValidationResult, ...] = ()
    preprocessing_results: Tuple[PreprocessingResult, ...] = ()
    # This may need to be a series, e.g. normalize by the average
    # observed value per day. Hence, repeat here instead of
    # only in original
    normalization_factor: Union[pd.Series, float] = 1.0
    uncertainty: Union[None, float] = None
    cost: Union[Cost, None] = None


@dataclass(frozen=True)
class MetricValue(BaseModel):
    """Class for storing the result of a single metric calculation.

    Parameters
    ----------
    category: str
        The category of the metric value, e.g. total, monthly, hourly.
    metric: str
        The metric that was calculated.
    index: str
        The index of the metric value, e.g. '1-12' for monthly metrics or
        0-23 for hourly.
    value: float
        The value calculated for the metric.
    """
    category: str
    metric: str
    index: str
    value: float


@dataclass(frozen=True)
class MetricResult(BaseModel):
    """Class for storing the results of many metric calculations for a single
    observation and forecast pair.

    Parameters
    ----------
    name: str
        A descriptive name for the MetricResult.
    forecast_id: str
        UUID of the forecast being analyzed.
    values: tuple of :py:class: `solarforecastarbiter.datamodel.MetricValue`
    observation_id: str or None
        UUID of the observation being analyzed.
    aggregate_id: str or None
        UUID of the aggregate being analyzed.
    is_summary: bool
        If this metric result represents summary statistics of the
        observation and forecasts timeseries.

    Notes
    -----
    Only one of `aggregate_id` or `observation_id` may be set.

    Raises
    ------
    ValueError
        When both `aggregate_id` and `observation_id` are not None, or when
        both are None.
    """
    name: str
    forecast_id: str
    values: Tuple[MetricValue, ...]
    observation_id: Union[str, None] = None
    aggregate_id: Union[str, None] = None
    is_summary: bool = False

    def __post_init__(self):
        if (
                (self.observation_id is None and self.aggregate_id is None)
                or (
                    self.observation_id is not None and
                    self.aggregate_id is not None
                )
        ):
            raise ValueError(
                'One of observation_id OR aggregate_id must be set')


def __check_plot_spec__(plot_spec):
    """Ensure that the provided plot specification is a valid JSON object"""
    try:
        spec_dict = json.loads(plot_spec)
        validate(instance=spec_dict, schema={'type': 'object'})
    except (json.JSONDecodeError, ValidationError):
        raise ValueError('Figure spec must be a valid json object.')


@dataclass(frozen=True)
class ReportFigure(BaseModel):
    """Parent class for different types of Report Figures"""
    def __post_init__(self):
        if type(self) == ReportFigure:
            raise ValueError("Invalid Report Figure. Figures must be of class "
                             "PlotlyReportFigure or BokehReportFigure.")

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        if model != ReportFigure:
            return super().from_dict(dict_, raise_on_extra)
        figure_class = dict_.get('figure_class')
        if figure_class == 'plotly':
            return PlotlyReportFigure.from_dict(dict_, raise_on_extra)
        elif figure_class == 'bokeh':
            return BokehReportFigure.from_dict(dict_, raise_on_extra)
        else:
            raise NotImplementedError(
                'Do not know how to process dict into a ReportFigure.')


@dataclass(frozen=True)
class PlotlyReportFigure(ReportFigure):
    """A class for storing metric plots for a report with associated metadata.

    Parameters
    ----------
    name: str
        A descriptive name for the figure.
    spec: str
        JSON string representation of the plotly plot.
    figure_type: str
        The type of plot, e.g. bar or scatter.
    pdf: str
        A static PDF copy of the plot, for including in PDF reports.
    svg: str
        DEPRECATED for pdf. A static svg copy of the plot.
    category: str
        The metric category. One of ALLOWED_CATEGORIES keys.
    metric: str
        The metric being plotted.
    """
    name: str
    spec: str
    figure_type: str
    pdf: str = ''
    svg: str = ''
    category: str = ''
    metric: str = ''
    figure_class: str = 'plotly'

    def __post_init__(self):
        __check_plot_spec__(self.spec)


@dataclass(frozen=True)
class BokehReportFigure(ReportFigure):
    """A class for storing metric plots for a report with associated metadata.
    Parameters
    ----------
    name: str
        A descriptive name for the figure.
    div: str
        An html div element to be target of Bokeh javascript.
    svg: str
        A static svg copy of the plot, for including in the pdf version.
    figure_type: str
        The type of plot, e.g. bar or scatter.
    category: str
        The metric category. One of ALLOWED_CATEGORIES keys.
    metric: str
        The metric being plotted.
    """
    name: str
    div: str
    svg: str
    figure_type: str
    category: str = ''
    metric: str = ''
    figure_class: str = 'bokeh'


def __bokeh_or_plotly__(cls):
    if cls.bokeh_version is not None and cls.plotly_version is not None:
        raise KeyError('Only provide one of "bokeh_version" or '
                       '"plotly_version" to RawReportPlots')
    elif cls.bokeh_version is None and cls.plotly_version is None:
        raise KeyError('Must provide one of "bokeh_version" or '
                       '"plotly_version" to RawReportPlots')


@dataclass(frozen=True)
class RawReportPlots(BaseModel):
    """Class for storing collection of all metric plots on a raw report.

    Parameters
    ----------
    figures: tuple of :py:class:`solarforecastarbiter.datamodel.ReportFigure`
    plotly_version: str
        The plotly version used when generating metrics plots.
    """
    figures: Tuple[ReportFigure, ...]
    plotly_version: Union[str, None] = None
    bokeh_version: Union[str, None] = None
    script: Union[str, None] = None

    def __post_init__(self):
        __bokeh_or_plotly__(self)
        if self.bokeh_version is not None:
            if self.script is None:
                raise KeyError('Must provide script for Bokeh plots to '
                               'RawReportPlots')


@dataclass(frozen=True)
class ReportMessage(BaseModel):
    """Class for intercepting errors and warnings associated with report
    processing.

    Parameters
    ----------
    messages: str
    step: str
    level: str
    function: str
        The function where the error originated.
    """
    message: str
    step: str
    level: str
    function: str


@dataclass(frozen=True)
class RawReport(BaseModel):
    """Class for holding the result of processing a report request
    including some metadata, the calculated metrics, plots, the
    processed forecast/observation data, and messages from report
    generation. This is called a "raw" report because this object,
    along with the report parameters, can be processed into a HTML or
    PDF report.

    Parameters
    ----------
    generated_at: pandas.Timestamp
        The time at report computation.
    timezone: str
        The IANA timezone of the report.
    versions: dict
        Dictionary of version information to ensure the correct version of
        the core library is used when rendering or recomputing the report.
    plots: :py:class:`solarforecastarbiter.datamodel.RawReportPlots`
    metrics: tuple of :py:class:`solarforecastarbiter.datamodel.MetricResult`
    processed_forecasts_observations: tuple of :py:class:`solarforecastarbiter.datamodel.ProcessedForecastObservation`
    messages: tuple of :py:class:`solarforecastarbiter.datamodel.ReportMessage`
    data_checksum: str or None
        SHA-256 checksum of the raw data used in the report.

    """  # NOQA
    generated_at: pd.Timestamp
    timezone: str
    versions: Tuple[Tuple[str, str], ...]
    plots: RawReportPlots
    metrics: Tuple[MetricResult, ...]
    processed_forecasts_observations: Tuple[ProcessedForecastObservation, ...]
    messages: Tuple[ReportMessage, ...] = ()
    data_checksum: Union[str, None] = None


def __check_cost_consistency__(object_pairs, available_costs):
    cost_names = [ac.name for ac in available_costs]
    for op in object_pairs:
        if op.cost is not None and op.cost not in cost_names:
            raise ValueError(
                f'Object pair cost, {op.cost}, not present in cost '
                'parameters specified here')


@dataclass(frozen=True)
class ReportParameters(BaseModel):
    """Parameters required to define and generate a Report.

    Parameters
    ----------
    name : str
        Name of the report.
    start : pandas.Timestamp
        Start time of the reporting period.
    end : pandas.Timestamp
        End time of the reporting period.
    forecast_fill_method : {'drop', 'forward', float}
        Indicates what process to use for handling missing forecasts.
    object_pairs: Tuple of ForecastObservation or ForecastAggregate
        Paired Forecasts and Observations or Aggregates to be analyzed
        in the report.
    metrics : Tuple of str
        Metrics to be computed in the report.
    categories : Tuple of str
        Categories to compute and organize metrics over in the report.
    filters : Tuple of Filters
        Filters to be applied to the data in the report.
    costs : Tuple of Costs
        Cost parameters that can be referenced in `object_pairs`
        to compute cost metrics for that pair. Each object pair must have
        the 'cost' parameter set to None (no cost calculation will be
        performed) or one of the names of these costs.
    """
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    object_pairs: Tuple[Union[ForecastObservation, ForecastAggregate], ...]
    metrics: Tuple[str, ...] = ('mae', 'mbe', 'rmse')
    categories: Tuple[str, ...] = ('total', 'date', 'hour')
    forecast_fill_method: str = 'drop'
    filters: Tuple[BaseFilter, ...] = field(
        default_factory=lambda: (QualityFlagFilter(), ))
    costs: Tuple[Cost, ...] = tuple()

    def __post_init__(self):
        # ensure that all forecast and observation units are the same
        __check_units__(*itertools.chain.from_iterable(
            (k.forecast, k.data_object) for k in self.object_pairs))
        # ensure the metrics can be applied to the forecasts and observations
        for k in self.object_pairs:
            __check_metrics__(k.forecast, self.metrics)
        # ensure that categories are valid
        __check_categories__(self.categories)
        __check_cost_consistency__(self.object_pairs, self.costs)


@dataclass(frozen=True)
class Report(BaseModel):
    """Class for keeping track of report metadata and the raw report that
    can later be rendered to HTML or PDF. Functions in
    :py:mod:`~solarforecastarbiter.reports.main` take a Report object
    with `raw_report` set to None, generate the report, and return
    another Report object with `raw_report` set to a RawReport object
    that can be rendered.

    Parameters
    ----------
    report_parameters : ReportParameters
        Metadata required to specify and generate the report.
    raw_report : RawReport or None
        Once computed, the raw report should be stored here
    status : str
        Status of the report
    report_id : str
        ID of the report in the API
    provider : str, optional
        Provider of the Report information.
    __version__ : str
        Should be used to version reports to ensure even older
        reports can be properly rendered

    """
    report_parameters: ReportParameters
    raw_report: Union[None, RawReport] = None
    status: str = 'pending'
    report_id: str = ''
    provider: str = ''
    __version__: int = 0  # should add version to api


FORECAST_TYPE_MAPPING = {
    'forecast': Forecast,
    'event_forecast': EventForecast,
    'probabilistic_forecast': ProbabilisticForecast,
    'probabilistic_forecast_constant_value': ProbabilisticForecastConstantValue
}
