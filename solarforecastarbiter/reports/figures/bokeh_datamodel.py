"""File containing datamodel objects for creating bokeh metric
plots.
"""
from typing import Tuple

from solarforecastarbiter.datamodel import (dataclass, ReportFigure,
                                            RawReportPlots)


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
    div: str = ''
    svg: str = ''
    figure_type: str = ''
    category: str = ''
    metric: str = ''
    spec: str = ''  # To be ignored for bokeh plots

    def __post_init__(self):
        pass


@dataclass(frozen=True)
class BokehRawReportPlots(RawReportPlots):
    """Class for storing collection of all metric plots on a raw report.
    Parameters
    ----------
    bokeh_version: str
        The bokeh version used when generating the plots.
    script: str
        The html script tag containing all of the bokeh javascript for the
        plots.
    figures: tuple of :py:class:`solarforecastarbiter.datamodel.ReportFigure`
    """
    bokeh_version: str = ''
    script: str = ''
    figures: Tuple[BokehReportFigure, ...]
    plotly_version: str = ''
