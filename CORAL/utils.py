__author__ = "Jake Nunemaker"
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
import datetime as dt

import pandas as pd
import matplotlib.pyplot as plt


def get_installed_capacity_by(df, year) -> int:
    """
    Return installed capacity by specific year.

    Parameters
    ----------
    df : pd.DataFrame
    year : int
    """

    installed_capacity = df.loc[
        df["Date Finished"] < dt.datetime(year, 1, 1, 0, 0)
    ]["capacity"].sum()

    partial = df.loc[
        (df["Date Finished"] > dt.datetime(year, 1, 1, 0, 0))
        & (df["Date Started"] < dt.datetime(year, 1, 1, 0, 0))
    ]

    for _, project in partial.iterrows():
        perc = (dt.datetime(year, 1, 1, 0, 0) - project["Date Started"]) / (
            project["Date Finished"] - project["Date Started"]
        )
        installed_capacity += perc * project["capacity"]

    return installed_capacity


def squarify(data):

    out = []
    for i, (time, cap) in enumerate(data):

        out.append((time, cap))
        try:
            if cap != data[i + 1][1]:
                out.append((data[i + 1][0], cap))

        except IndexError:
            pass

    return pd.DataFrame(out, columns=["time", "capacity"])


def plot_shared_resource_capacities(
    manager, directory, ignore_cols=None, col_map=None
):

    if ignore_cols is None:
        ignore_cols = []

    if col_map is None:
        col_map = {}

    history = pd.DataFrame(manager.resource_history)
    history = history.groupby("time").tail(1)

    cols = [
        c for c in list(history.columns) if c not in ["time", *ignore_cols]
    ]

    for col in cols:
        data = list(zip(history["time"], history[col]))
        data = squarify(data)

        fig = plt.figure(figsize=(6, 2), dpi=200)
        ax = fig.add_subplot(111)

        try:
            label = col_map[col]

        except KeyError:
            label = col

        ax.plot(data["time"], data["capacity"], label=label)

        ax.set_xlabel("Time")
        ax.set_ylabel("Resource Capacity")
        ax.set_title(label)

        ax.set_ylim(0, ax.get_ylim()[1])

        fp = os.path.join(directory, f"{label}.png")
        fig.savefig(fp)
