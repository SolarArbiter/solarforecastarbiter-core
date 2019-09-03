# coding: utf-8
"""
Data classes and acceptable variables as defined by the SolarForecastArbiter
Data Model document. Python 3.7 is required.
"""
from dataclasses import (dataclass, field, fields, MISSING, asdict,
                         replace, is_dataclass)
import datetime
import itertools
from typing import Tuple, Union


import pandas as pd


from solarforecastarbiter.validation.quality_mapping import \
    DESCRIPTION_MASK_MAPPING


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
    'curtailment': 'Curtailment'
}


CLOSED_MAPPING = {
    'instant': None,
    'beginning': 'left',
    'ending': 'right'
}


def _dict_factory(inp):
    dict_ = dict(inp)
    for k, v in dict_.items():
        if isinstance(v, datetime.time):
            dict_[k] = v.strftime('%H:%M')
        elif isinstance(v, datetime.datetime):
            dict_[k] = v.isoformat()
        elif isinstance(v, pd.Timedelta):
            # convert to integer minutes
            dict_[k] = v.total_seconds() // 60

    if 'units' in dict_:
        del dict_['units']
    return dict_


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
        """
        dict_ = input_dict.copy()
        model_fields = fields(model)
        kwargs = {}
        errors = []
        for model_field in model_fields:
            if model_field.name in dict_:
                if model_field.type == pd.Timedelta:
                    kwargs[model_field.name] = pd.Timedelta(
                        f'{dict_[model_field.name]}min')
                elif model_field.type == pd.Timestamp:
                    kwargs[model_field.name] = pd.Timestamp(
                        dict_[model_field.name])
                elif model_field.type == datetime.time:
                    kwargs[model_field.name] = datetime.datetime.strptime(
                        dict_[model_field.name], '%H:%M').time()
                elif (
                        is_dataclass(model_field.type) and
                        isinstance(dict_[model_field.name], dict)
                ):
                    kwargs[model_field.name] = model_field.type.from_dict(
                        dict_[model_field.name])
                else:
                    kwargs[model_field.name] = model._special_field_processing(
                        model, model_field, dict_[model_field.name])
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
    """
    name: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    site_id: str = ''
    provider: str = ''
    extra_parameters: str = ''

    @classmethod
    def from_dict(model, input_dict, raise_on_extra=False):
        dict_ = input_dict.copy()
        if 'modeling_parameters' in dict_:
            mp_dict = dict_.get('modeling_parameters', {})
            if not isinstance(mp_dict, PVModelingParameters):
                tracking_type = mp_dict.pop('tracking_type', None)
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
    PVModelingParameters
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


def __check_units__(*args):
    ref_unit = args[0].units
    if not all(arg.units == ref_unit for arg in args):
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

    Maybe not needed, but makes Report type spec easier and allows for
    __post_init__ checking.
    """
    forecast: Forecast
    observation: Observation

    def __post_init__(self):
        __check_units__(self.forecast, self.observation)
        __check_interval_compatibility__(self.forecast, self.observation)


@dataclass(frozen=True)
class BaseFilter(BaseModel):
    """
    Base class for filters to be applied in a report.
    """
    pass


@dataclass(frozen=True)
class QualityFlagFilter(BaseFilter):
    """
    Class representing quality flag filters to be applied in a report.

    Parameters
    ----------
    quality_flags : Tuple of str
        Strings corresponding to ``BITMASK_DESCRIPTION_DICT`` keys.
        These periods will be excluded from the analysis.
    """
    quality_flags: Tuple[str] = (
        'UNEVEN FREQUENCY', 'LIMITS EXCEEDED', 'CLEARSKY EXCEEDED',
        'STALE VALUES', 'INCONSISTENT IRRADIANCE COMPONENTS'
    )

    def __post_init__(self):
        if not all(flag in DESCRIPTION_MASK_MAPPING
                   for flag in self.quality_flags):
            raise ValueError('Quality flags must be in '
                             'BITMASK_DESCRIPTION_DICT')


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


@dataclass(frozen=True)
class ValueFilter(BaseFilter):
    """
    Class representing an observation or forecast value filter to be
    applied in a report.

    Parameters
    ----------
    metadata : Observation or Forecast
        Object to get values for.
    value_range : (float, float) tuple
        Value range to calculate errors. Range is inclusive
        of both endpoints. Filters are applied before resampling.
    """
    metadata: Union[Observation, Forecast]
    value_range: Tuple[float, float]


def __check_metrics__():
    # maybe belongs in the metrics package
    # deterministic forecasts --> deterministic metrics
    # probabilistic forecasts --> probabilistic metrics
    # event forecasts --> event metrics
    pass


@dataclass(frozen=True)
class ReportMetadata(BaseModel):
    """
    Hold additional metadata about the report
    """
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    now: pd.Timestamp
    timezone: str
    versions: dict
    validation_issues: dict


# need apply filtering + resampling to each forecast obs pair
@dataclass(frozen=True)
class ProcessedForecastObservation(BaseModel):
    """
    Hold the processed forecast and observation data with the resampling
    parameters
    """
    # do this instead of subclass to compare objects later
    original: ForecastObservation
    interval_value_type: str
    interval_length: pd.Timedelta
    interval_label: str
    forecast_values: Union[pd.Series, str, None]
    observation_values: Union[pd.Series, str, None]


@dataclass(frozen=True)
class RawReport(BaseModel):
    """
    Class for holding the result of processing a report request including
    the calculated metrics, some metadata, the markdown template, and
    the processed forecast/observation data.
    """
    metadata: ReportMetadata
    template: str
    metrics: dict  # later MetricsResult
    processed_forecasts_observations: Tuple[ProcessedForecastObservation]

    def _special_field_processing(self, model_field, val):
        if model_field.name == 'processed_forecasts_observations':
            out = []
            for v in val:
                if isinstance(v, dict):
                    out.append(ProcessedForecastObservation.from_dict(v))
                else:
                    out.append(v)
            return tuple(out)
        else:
            return val


@dataclass(frozen=True)
class Report(BaseModel):
    """
    Class for keeping track of report metadata and the raw report that
    can later be rendered to HTML or PDF. Functions in
    :py:mod:`~solarforecastarbiter.reports.main` take a Report object
    with `raw_report` set to None, generate the report, and return
    another Report object with `raw_report` set to a RawReport object
    that can be rendered.

    Parameters
    ----------
    name : str
        Name of the report.
    start : pandas.Timestamp
        Start time of the reporting period.
    end : pandas.Timestamp
        End time of the reporting period.
    forecast_observations : Tuple of ForecastObservation
        Paired Forecasts and Observations to be analyzed in the report.
    metrics : Tuple of str
        Metrics to be computed in the report.
    filters : Tuple of Filters
        Filters to be applied to the data in the report.
    status : str
        Status of the report
    report_id : str
        ID of the report in the API
    raw_report : RawReport or None
        Once computed, the raw report should be stored here
    __version__ : str
        Should be used to version reports to ensure even older
        reports can be properly rendered
    """
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    forecast_observations: Tuple[ForecastObservation]
    metrics: Tuple[str] = ('mae', 'mbe', 'rmse')
    filters: Tuple[BaseFilter] = field(default_factory=QualityFlagFilter)
    status: str = 'pending'
    report_id: str = ''
    raw_report: Union[None, RawReport] = None
    __version__: int = 0  # should add version to api

    def __post_init__(self):
        # ensure that all forecast and observation units are the same
        __check_units__(*itertools.chain.from_iterable(
            ((k.forecast, k.observation) for k in self.forecast_observations)))
        # ensure the metrics can be applied to the forecasts and observations
        __check_metrics__()
