from typing import Literal, TypeAlias

import pandas as pd
from bblocks import add_iso_codes_column

from scripts.analysis.common import (
    mask_grant_indicators,
    mask_grant_and_concessional_indicators,
    all_flows_pipeline,
    OUTPUT_GROUPER,
    create_dev_countries_total,
)
from scripts.models.seek import extract_decreases, apply_linear_reduction

from scripts.config import Paths

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


def extent_2023_data_to_2024(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extend the 2023 data to 2024 by duplicating the 2023 data.
    This is done to ensure that the data is available for the next year
    for projections and scenarios.

    Args:
        data (pd.DataFrame): Input data
    Returns:
        pd.DataFrame: The extended DataFrame with 2024 data.
    """
    # Create 2024 data based on 2023
    data24 = data.loc[lambda d: d.year == 2023].copy()
    data24.loc[:, "year"] = 2024
    return pd.concat([data, data24], ignore_index=True)


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


def projected_scenarios_with_multiplier(
    data: pd.DataFrame,
    multipliers: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
    multiplier_col: str = "realistic_multiplier",
    target_year: int = 2027,
) -> pd.DataFrame:
    """
    Project values forward to the target year using a precomputed multiplier column.

    Args:
        data (pd.DataFrame): Input data with 'year', 'value', 'indicator' columns.
        multipliers (pd.DataFrame): DataFrame with 'year', 'iso_code', and the multiplier column.
        version (Literal["grants", "concessional_finance"]): Which type of flows to reduce.
        multiplier_col (str): The name of the multiplier column to apply.
        target_year (int): The year to project to.

    Returns:
        pd.DataFrame: Projected DataFrame with values scaled by the multiplier.
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

    # Join the multipliers to the data
    data = data.merge(
        multipliers[["iso_code", "year", multiplier_col]],
        on=["iso_code", "year"],
        how="left",
    )

    # Define mask for rows where multiplier should apply
    if version == "grants":
        apply_mask = ~mask_grant_indicators(data)
    else:
        apply_mask = ~mask_grant_and_concessional_indicators(data)

    # Apply multiplier to value (default to 1 if no multiplier present)
    data["value"] = data["value"] * data[multiplier_col].where(apply_mask, 1.0).fillna(
        1.0
    )

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


from typing import Literal


def projected_inflows_scenario(
    data: pd.DataFrame,
    version: Literal["grants", "concessional_finance"],
    *,
    scenario: Literal[2, 3],
) -> pd.DataFrame:
    """
    Projected inflows scenarios.

    Args:
        data (pd.DataFrame): Input inflows data.
        version (Literal["grants", "concessional_finance"]): Type of flows to reduce.
        scenario (Literal[2, 3]): Scenario version.

    Returns:
        pd.DataFrame: The projected inflows DataFrame.
    """
    data = add_iso_codes_column(data, id_column="counterpart_area", id_type="regex")

    # Prepare decreasing countries and projection parameters based on scenario
    if scenario == 2:
        seek_scenarios = extract_decreases()
        multiplier_col = "realistic_multiplier"
        reductions = {"USA": 60, "rest": 0}
    elif scenario == 3:
        seek_scenarios = extract_decreases().pipe(
            apply_linear_reduction, reduction=0.1, start_year=2025, end_year=2027
        )
        multiplier_col = "realistic_multiplier_reduced"
        reductions = {"USA": 80, "rest": 10}
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    is_seek = data["iso_code"].isin(seek_scenarios["iso_code"])
    data_seek = data[is_seek]
    data_other = data[~is_seek]

    data_us = data_other[data_other["iso_code"] == "USA"]
    data_rest = data_other[data_other["iso_code"] != "USA"]

    projection_args = dict(version=version, target_year=2027)

    data_seek = projected_scenarios_with_multiplier(
        data_seek,
        multipliers=seek_scenarios,
        multiplier_col=multiplier_col,
        **projection_args,
    )
    data_us = projected_scenarios(
        data_us, reduce_by=reductions["USA"], **projection_args
    )
    data_rest = projected_scenarios(
        data_rest, reduce_by=reductions["rest"], **projection_args
    )

    projected = pd.concat([data_seek, data_us, data_rest], ignore_index=True)
    return (
        projected.groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )


if __name__ == "__main__":
    # Get the latest inflows data
    latest_inflows_data = get_latest_inflows()

    # --- Scenarios ---

    scenario1_reduced_concessional = projected_inflows_scenario1(
        latest_inflows_data, version="concessional_finance"
    )

    scenario2_reduced_concessional = projected_inflows_scenario(
        latest_inflows_data, version="concessional_finance", scenario=2
    )

    scenario3_reduced_concessional = projected_inflows_scenario(
        latest_inflows_data, version="concessional_finance", scenario=3
    )


    df = pd.concat(
        [
            scenario1_reduced_concessional.assign(scenario = "scenario 1"),
            scenario2_reduced_concessional.assign(scenario = "scenario 2"),
            scenario3_reduced_concessional.assign(scenario = "scenario 3"),

        ],
        ignore_index=True,
)
    df.to_csv(Paths.raw_data / "inflows_scenarios.csv", index=False)
