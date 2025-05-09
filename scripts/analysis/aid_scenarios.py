from typing import Literal

import numpy as np
import pandas as pd
from bblocks import add_iso_codes_column
from oda_data import ODAData, set_data_path
from oda_data import donor_groupings

from scripts import config
from scripts.config import Paths
from scripts.data.inflows import clean_grants_inflows_output
from scripts.logger import logger
from scripts.models.seek import extract_decreases, apply_linear_reduction

set_data_path(Paths.raw_data)


def get_historical_oda(constant: bool = False) -> pd.DataFrame:
    """
    Retrieve oda inflows from OECD ODA data.

    Args:
        - constant (bool): Whether to retrieve the data in constant or current prices.

    Returns:
        pd.DataFrame: DataFrame containing grants inflows data.
    """

    # Create an object with the basic settings
    oda = ODAData(
        donors=list(donor_groupings()["dac_countries"]) + [83],
        years=range(config.ANALYSIS_YEARS[0], config.ANALYSIS_YEARS[1] + 1),
        include_names=True,
        base_year=config.CONSTANT_BASE_YEAR if constant else None,
        prices="constant" if constant else "current",
    )

    # Load the data
    oda.load_indicator("total_oda_official_definition")

    # Retrieve and clean the data
    data = oda.get_data().assign(value=lambda d: d.value * 1e6)

    return data


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


def add_dac_total(data: pd.DataFrame) -> pd.DataFrame:
    if "realistic_multiplier" in data.columns:
        data = data.drop(columns=["realistic_multiplier"])

    total = (
        data.assign(donor_code=20001, donor_name="DAC Countries", iso_code="DAC")
        .groupby([c for c in data.columns if c not in ["value"]])
        .sum()
        .reset_index()
    )

    return pd.concat([total, data], ignore_index=True)


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
    data = add_iso_codes_column(data, id_column="donor_code", id_type="DACCode")

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

    if "realistic_multiplier_reduced" in projected.columns:
        projected = projected.drop(columns=["realistic_multiplier_reduced"])

    projected = add_dac_total(projected)

    return projected


def projections_chart() -> None:
    data = get_historical_oda()
    columns = ["year", "donor_name", "currency", "prices", "value"]
    oda_projected2 = (
        projected_inflows_scenario(data, scenario=2)
        .filter(columns)
        .rename(columns={"value": "ODA (Projected Scenario 2)"})
    )
    oda_projected3 = projected_inflows_scenario(data, scenario=3)
    oda_projected3 = oda_projected3.filter(columns).rename(
        columns={"value": "ODA (Projected Scenario 3)"}
    )
    historical = (
        data.pipe(add_dac_total).filter(columns).rename(columns={"value": "ODA"})
    )

    full = historical.merge(
        oda_projected2, how="outer", on=["year", "donor_name", "currency", "prices"]
    ).merge(
        oda_projected3, how="outer", on=["year", "donor_name", "currency", "prices"]
    )

    full.loc[
        lambda d: d.year < 2024,
        ["ODA (Projected Scenario 2)", "ODA (Projected Scenario 3)"],
    ] = pd.NA

    full_dac = full.loc[lambda d: d.donor_name == "DAC Countries"]
    full_non_dac = full.loc[lambda d: d.donor_name != "DAC Countries"]

    full = pd.concat([full_dac, full_non_dac], ignore_index=True)

    full.to_csv(Paths.output / "oda_scenarios.csv", index=False)


if __name__ == "__main__":
    projections_chart()
