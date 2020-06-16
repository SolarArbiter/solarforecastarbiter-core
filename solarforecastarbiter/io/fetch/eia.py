"""Functions to extract EIA."""

import requests
import pandas as pd


def get_eia_data(series_id, api_key, start, end):
    """

    Parameters
    ----------
    series_id : string
        The series ID in the EIA Open Data API.
    api_key : string
        The API key for accessing the EIA Open Data API.
    start : pd.Timestamp
    end : pd.Timestamp

    Returns
    -------
    df : pandas.Series

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

    # returned values:
    # - numeric if valid
    # - "null" if missing
    # - "w" if witheld
    # - "*" if statistically insignificant
    data = r.json()["series"][0]["data"]
    df = pd.DataFrame(data, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df
