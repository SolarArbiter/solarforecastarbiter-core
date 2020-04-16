"""Example reference (net) load forecast."""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def estimate_clearsky(df, var="demand"):
    """Estimate clear-sky dynamically.

    Parameters
    ----------
    df : pandas.DataFrame
        Time-series of the target variable.
    var : str
        The target variable column.

    Returns
    -------
    cs : pandas.DataFrame
        Time-series of the clear-sky model.

    """

    # reshape to (date, hour of day)
    df.insert(df.shape[1], "hod", df.index.hour + df.index.minute / 60.0)
    df = pd.pivot_table(df, index=df.index.normalize(), columns="hod", values=var)

    # estimate clearsky for weekday vs weekend (0=Monday, 6=Sunday)
    frames = []
    for dow in [(0, 1, 2, 3, 4), (5, 6)]:

        group = df[df.index.dayofweek.isin(dow)]

        # update every week
        group = group.resample("1w", closed="right", label="right").median()

        # if missing, use the most recent weekly estimate
        group = group.resample("1D", closed="right", label="right").fillna(method="ffill")

        # remove dates not in the specified DOW
        # (e.g. don't use the weekday trend for weekend)
        group = group[group.index.dayofweek.isin(dow)]

        frames.append(group)

    df = pd.concat(frames, axis=0).sort_index().reset_index()
    df = df.melt(id_vars="timestamp", var_name="hour", value_name=f"{var}_clear")
    df["hour"] = pd.to_numeric(df["hour"])
    df["timestamp"] = df["timestamp"] + df["hour"] * np.timedelta64(1, "h")
    df = df.set_index("timestamp")
    df = df[[f"{var}_clear"]]

    return df


if __name__ == "__main__":

    # timestamp (local) + net load [MWh]
    df = pd.read_csv("load_caiso.csv", parse_dates=True, index_col=0)

    cs = estimate_clearsky(df, var="demand")
    df = df.join(cs, how="outer")
    df.insert(df.shape[1], "demand_kt", df["demand"] / df["demand_clear"])

    #horizon = "2h"  # RT market
    horizon = "26h"  # DA market (due by 10am) = noon next day

    # add Persistence forecast
    pers = df.copy()
    pers = pers.shift(periods=1, freq=horizon)
    df = df.join(pers, rsuffix="_pers")

    # add Smart Pers. forecast
    df.insert(df.shape[1], "demand_smart", df["demand_kt_pers"] * df["demand_clear"])

    # add timestamps
    df.index.name = "target_datetime"
    df.insert(df.shape[1], "forecast_datetime", df.index - pd.Timedelta(horizon))

    df = df[["demand", "demand_pers", "demand_smart"]].dropna(how="any")

    # compute error metrics
    var = "demand"
    for model in ["pers", "smart"]:
        error = (df[f"{var}_{model}"] - df[var]).values
        mae = np.mean(np.abs(error))
        mbe = np.mean(error)
        rmse = np.sqrt(np.mean(error ** 2))
        mape = np.mean(np.abs(error / df[var].values))
        print("{:<10}: {:<6}: MAE = {:>6.1f} | MBE = {:>6.1f} | RMSE = {:>6.1f} | MAPE = {:>6.1%} | max = {:>8.1f}".format(var, model, mae, mbe, rmse, mape, df[var].max()))

    #fig, axes = plt.subplots(2, 1, sharex=True)
    #ax = axes[0]
    #ax.plot(df["demand"], "-k")
    #ax.plot(df["demand_clear"], "--r")

    #ax = axes[1]
    #ax.plot(df["demand_kt"], "--b")

    #plt.show()
