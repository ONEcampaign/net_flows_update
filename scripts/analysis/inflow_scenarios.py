from typing import Literal, TypeAlias

import pandas as pd

from scripts.analysis.common import (
    mask_grant_indicators,
    mask_grant_and_concessional_indicators,
    all_flows_pipeline,
    OUTPUT_GROUPER,
    create_dev_countries_total,
)

Percent: TypeAlias = int


def get_latest_inflows(
    constant: bool = False,
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    china_as_counterpart_type: bool = False,
) -> pd.DataFrame:
    """
    Get the latest inflows data
    """
    inflows_data = all_flows_pipeline(
        as_net_flows=False,
        version="total",
        exclude_outliers=exclude_outliers,
        remove_countries_wo_outflows=remove_countries_wo_outflows,
        china_as_counterpart_type=china_as_counterpart_type,
        constant=constant,
    ).loc[lambda d: d.indicator_type == "inflow"]

    total_data = create_dev_countries_total(data=inflows_data)

    inflows_data = pd.concat([total_data, inflows_data], ignore_index=True)

    # Get the latest inflows data
    latest_year = inflows_data.year.max()

    latest_inflows = inflows_data.loc[inflows_data.year == latest_year]

    return latest_inflows


def projected_scenarios(
    data: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
    reduce_by: Percent = 0,
    target_year: int = 2027,
) -> pd.DataFrame:
    """
    Project values forward to the target year, reducing the specified type
    (grants or concessional finance) linearly by the given percentage.

    Args:
        data (pd.DataFrame): Input data with 'year', 'value', and 'indicator' columns.
        version (Literal["grants", "concessional_finance"]): Which type of flows to reduce.
        reduce_by (Percent): Total percentage reduction by the target year.
        target_year (int): The year to project to.

    Returns:
        pd.DataFrame: The projected DataFrame.
    """
    latest_year = data["year"].max()

    # Extend data to target_year by duplicating latest known year
    future_years = range(latest_year + 1, target_year + 1)
    projected_rows = []

    for year in future_years:
        projected = data[data["year"] == latest_year].copy()
        projected["year"] = year
        projected_rows.append(projected)

    if projected_rows:
        data = pd.concat([data] + projected_rows, ignore_index=True)

    # Define mask for rows to reduce
    if version == "grants":
        reduce_mask = ~mask_grant_indicators(data)
    else:
        reduce_mask = ~mask_grant_and_concessional_indicators(data)

    # Apply linear reduction only to future years
    years_to_reduce = range(latest_year + 1, target_year + 1)
    n_years = target_year - latest_year
    for i, year in enumerate(years_to_reduce, start=1):
        factor = 1 - (reduce_by / 100) * (i / n_years)  # Linear interpolation
        year_mask = (data["year"] == year) & reduce_mask
        data.loc[year_mask, "value"] *= factor

    return data


def projected_inflows_scenario1(
    data: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
) -> pd.DataFrame:
    """
    SCENARIO 1: [version] flows remain constant in nominal terms.

    Args:
        data (pd.DataFrame): Input inflows data.
        version (Literal["grants", "concessional_finance"]): Which type of flows to reduce.


    Returns:
        pd.DataFrame: The projected inflows DataFrame.
    """
    return (
        projected_scenarios(data, version=version, reduce_by=0, target_year=2027)
        .groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )


def projected_inflows_scenario2(
    data: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
) -> pd.DataFrame:
    """
    SCENARIO 2: [version] flows decline by 30% in nominal terms by 2027.

    Args:
        data (pd.DataFrame): Input inflows data.
        version (Literal["grants", "concessional_finance"]): Which type of flows to reduce.
    Returns:
        pd.DataFrame: The projected inflows DataFrame.
    """
    return (
        projected_scenarios(data, version=version, reduce_by=30, target_year=2027)
        .groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )


def projected_inflows_scenario3(
    data: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
) -> pd.DataFrame:
    """
    SCENARIO 3: [version] flows decline by 50% in nominal terms by 2027.

    Args:
        data (pd.DataFrame): Input inflows data.
        version (Literal["grants", "concessional_finance"]): Which type of flows to reduce.

    Returns:
        pd.DataFrame: The projected inflows DataFrame.
    """
    return (
        projected_scenarios(data, version=version, reduce_by=50, target_year=2027)
        .groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )


if __name__ == "__main__":
    # Get the latest inflows data
    latest_inflows_data = get_latest_inflows()

    # --- Scenarios ---

    scenario1_reduced_grants = projected_inflows_scenario1(
        latest_inflows_data, version="grants"
    )
    scenario1_reduced_concessional = projected_inflows_scenario1(
        latest_inflows_data, version="concessional_finance"
    )

    scenario2_reduced_grants = projected_inflows_scenario2(
        latest_inflows_data, version="grants"
    )
    scenario2_reduced_concessional = projected_inflows_scenario2(
        latest_inflows_data, version="concessional_finance"
    )

    scenario3_reduced_grants = projected_inflows_scenario3(
        latest_inflows_data, version="grants"
    )
    scenario3_reduced_concessional = projected_inflows_scenario3(
        latest_inflows_data, version="concessional_finance"
    )
