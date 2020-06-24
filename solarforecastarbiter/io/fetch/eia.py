"""Functions to extract EIA."""

import requests
import pandas as pd


def get_eia_data(series_id, api_key, start, end):
    """
    Retrieve data from the EIA Open Data API. [1]_

    Parameters
    ----------
    series_id : string
        The series ID in the EIA API.
    api_key : string
        The API key for accessing the EIA API.
    start : pd.Timestamp
        The start timestamp in UTC.
    end : pd.Timestamp
        The end timestamp in UTC.

    Returns
    -------
    pandas.DataFrame
        The data from *start* to *end* for *series_id*.

    References
    ----------
    .. [1] https://www.eia.gov/opendata/

    Examples
    --------
    >>> series_id = "EBA.CISO-ALL.D.H"  # CAISO demand [MW]
    >>> start = pd.Timestamp("20200601T0800Z")
    >>> end = pd.Timestamp("20200602T0800Z")
    >>> api_key = "yourapikeygoeshere"
    >>> get_eia_data(series_id, api_key, start, end)
    timestamp                    value
    2020-06-01 08:00:00+00:00    22028
    2020-06-01 09:00:00+00:00    21141
    2020-06-01 10:00:00+00:00    20573
    ...

    """

    base_url = "https://api.eia.gov/series/"
    params = dict(
        api_key=api_key,
        series_id=series_id,
        start=start.strftime("%Y%m%dT%HZ"),
        end=end.strftime("%Y%m%dT%HZ"),
    )
    r = requests.get(base_url, params=params)
    r.raise_for_status()

    data = r.json()["series"][0]["data"]
    df = pd.DataFrame(data, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # replace invalid values with NaN
    # - "null" if missing
    # - "w" if witheld
    # - "*" if statistically insignificant
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    return df
