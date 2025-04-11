import pandas as pd
import numpy as np

from scripts.analysis.common import (
    all_flows_pipeline,
    AnalysisVersion,
    create_dev_countries_total,
    AVERAGE_PERIODS,
)
from scripts.config import Paths


def historical_inflows(
    debt_only: bool = False,
    version: AnalysisVersion = "total",
    china_as_counterpart_type: bool = False,
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    constant: bool = False,
) -> pd.DataFrame:

    inflows_data = all_flows_pipeline(
        as_net_flows=False,
        version=version,
        exclude_outliers=exclude_outliers,
        remove_countries_wo_outflows=remove_countries_wo_outflows,
        china_as_counterpart_type=china_as_counterpart_type,
        constant=constant,
    ).loc[lambda d: d.indicator_type == "inflow"]

    if debt_only:
        inflows_data = inflows_data.loc[
            lambda d: ~d.indicator.isin(["grants_bilateral", "grants_multilateral"])
        ]

    total_data = create_dev_countries_total(data=inflows_data)

    inflows_data = pd.concat([total_data, inflows_data], ignore_index=True)

    inflows_data = (
        inflows_data.groupby(
            [c for c in inflows_data.columns if c not in ["value", "indicator"]]
        )[["value"]]
        .sum()
        .reset_index()
    )

    if china_as_counterpart_type:
        inflows_data["counterpart_type"] = inflows_data["counterpart_type"].replace(
            {"Bilateral": "Bilateral (excl. China)", "Private": "Private (excl. China)"}
        )

    return inflows_data


def inflows_by_period(
    inflows_data: pd.DataFrame, china_as_counterpart_type: bool = True
) -> pd.DataFrame:
    """
    Group inflows data by specified AVERAGE_PERIODS and calculate average values.

    Args:
        inflows_data (pd.DataFrame): The data to group.
        china_as_counterpart_type (bool): Whether to treat China as a counterpart type.
    Returns:
        pd.DataFrame: The grouped DataFrame with average values.

    """

    inflows_data = inflows_data.copy(deep=True)

    for period, settings in AVERAGE_PERIODS.items():
        start_year, end_year = settings["years"]
        inflows_data.loc[
            (inflows_data.year >= start_year) & (inflows_data.year <= end_year),
            "period",
        ] = period

    # Remove rows with NaN in the 'period' column
    data = inflows_data.dropna(subset=["period"])

    # Grouper
    grouper = ["period", "country", "continent", "income_level", "prices"]

    if china_as_counterpart_type:
        grouper.append("counterpart_type")

    # Group by period and sum the values
    data = (
        data.groupby(grouper, observed=True, dropna=False)["value"].sum().reset_index()
    )

    data["length"] = data["period"].map(
        lambda x: AVERAGE_PERIODS[x]["length"] if x in AVERAGE_PERIODS else None
    )

    data["value"] = (data["value"] / data["length"]).round(2)

    return data.drop(columns=["length"])


def add_income_aggs(df):
    """" """

    return pd.concat([(df
                       .loc[lambda d: (d.country != 'Developing countries')&(d.income_level != "High income")]
                       .groupby(by=['year', 'income_level', 'counterpart_type'])
                       .agg({"value": "sum"})
                       .reset_index()
                       .rename(columns = {"income_level": "country"})
                       .assign(income_level = None,
                               continent = None,
                               prices="current",
                               period=np.nan
                               )
                       ), df], ignore_index=True)


def add_africa_agg(df):
    """ """

    return pd.concat([(df.loc[lambda d: (d.continent == 'Africa')]
                       .groupby(by=['year', 'continent', 'counterpart_type'])
                       .agg({"value": "sum"})
                       .reset_index()
                       .rename(columns = {"continent": "country"})
                       .assign(income_level = None,
                               continent = None,
                               prices="current",
                               period=np.nan
                               )
                       ), df], ignore_index=True)



if __name__ == "__main__":
    total_inflows = historical_inflows(debt_only=True, china_as_counterpart_type=True)

    total_inflows = total_inflows.pipe(add_income_aggs).pipe(add_africa_agg)
    total_inflows.to_csv(Paths.raw_data / "total_inflows.csv", index=False)

    # total_inflows_avg = inflows_by_period(total_inflows, china_as_counterpart_type=True)
