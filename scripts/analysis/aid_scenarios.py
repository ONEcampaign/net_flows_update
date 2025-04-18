from typing import Literal

import numpy as np
import pandas as pd
from bblocks import add_iso_codes_column
from oda_data import ODAData, set_data_path
from oda_data import donor_groupings


from scripts.config import Paths
from scripts.logger import logger
from scripts.models.seek import extract_decreases, apply_linear_reduction

set_data_path(Paths.raw_data)


def projected_oda_with_multiplier(
    data: pd.DataFrame,
    multipliers: pd.DataFrame,
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

    # Apply multiplier to value (default to 1 if no multiplier present)
    data["value"] = data["value"] * data[multiplier_col].fillna(1.0)

    return data


def projected_oda_scenarios(
    data: pd.DataFrame,
    reduce_by=0,
    target_year: int = 2027,
) -> pd.DataFrame:
    """
    Project values forward to the target year, reducing the specified type
    (grants or concessional finance) linearly by the given percentage.

    Args:
        data (pd.DataFrame): Input data with 'year', 'value', and 'indicator' columns.
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

    # Apply linear reduction only to future years
    years_to_reduce = range(latest_year + 1, target_year + 1)
    n_years = target_year - latest_year
    for i, year in enumerate(years_to_reduce, start=1):
        factor = 1 - (reduce_by / 100) * (i / n_years)  # Linear interpolation
        year_mask = data["year"] == year
        data.loc[year_mask, "value"] *= factor

    return data


def projected_inflows_scenario(
    data: pd.DataFrame,
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
    data = add_iso_codes_column(data, id_column="donor_name", id_type="regex")

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

    projection_args = dict(target_year=2027)

    data_seek = projected_oda_with_multiplier(
        data_seek,
        multipliers=seek_scenarios,
        multiplier_col=multiplier_col,
        **projection_args,
    )
    data_us = projected_oda_scenarios(
        data_us, reduce_by=reductions["USA"], **projection_args
    )
    data_rest = projected_oda_scenarios(
        data_rest, reduce_by=reductions["rest"], **projection_args
    )

    projected = pd.concat([data_seek, data_us, data_rest], ignore_index=True)
    return projected
