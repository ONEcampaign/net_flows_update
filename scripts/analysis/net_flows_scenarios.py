""" """

import pandas as pd

from scripts.config import Paths


inflows_df = pd.read_csv(Paths.raw_data / "inflows_scenarios.csv")
outflows_df = pd.read_csv(Paths.raw_data / "outflows_scenarios.csv")


def add_income_level_aggregates(df):
    """ """
    agg_df = (
        df.loc[lambda d: d.country != "Developing countries"]
        .groupby(["year", "scenario", "income_level"])
        .agg({"inflows": "sum", "outflows": "sum"})
        .reset_index()
        .loc[lambda d: d.income_level != "High income"]
        .rename(columns={"income_level": "country"})
        .assign(income_level=None, continent=None, prices="current")
    )

    return pd.concat([df, agg_df])


def add_africa_aggregate(df):
    """ """
    agg_df = (
        df.loc[lambda d: d.continent == "Africa"]
        .groupby(["year", "continent", "scenario"])
        .agg({"inflows": "sum", "outflows": "sum"})
        .reset_index()
        .rename(columns={"continent": "country"})
        .assign(income_level=None, continent=None, prices="current")
    )

    return pd.concat([df, agg_df])


if __name__ == "__main__":

    scenarios = (
        pd.merge(
            inflows_df.rename(columns={"value": "inflows"})
            .drop(columns="indicator_type")
            .loc[lambda d: d.year != 2023],
            outflows_df.rename(columns={"value": "outflows"}).drop(
                columns="indicator_type"
            ),
            how="left",
        )
        .pipe(add_income_level_aggregates)
        .pipe(add_africa_aggregate)
        .assign(net_flows=lambda d: d.inflows + d.outflows)
    )

    scenarios.to_csv(Paths.raw_data / "net_flows_scenarios.csv", index=False)
