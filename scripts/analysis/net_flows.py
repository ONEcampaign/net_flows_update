from typing import Literal

import pandas as pd
from bblocks import set_bblocks_data_path

from scripts.analysis.common import (
    exclude_outlier_countries,
    add_china_as_counterpart_type,
    convert_to_net_flows,
    exclude_countries_without_outflows,
)
from scripts.config import Paths
from scripts.data.inflows import get_total_inflows
from scripts.data.outflows import get_debt_service_data

set_bblocks_data_path(Paths.raw_data)

LATEST_INFLOWS: int = 2023

AnalysisVersion = Literal["total", "excluding_grants", "excluding_concessional_finance"]


def prep_flows(inflows: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the inflow data for further processing.

    This function drops rows with NaN in 'iso_code', zero in 'value', or 'World' in
    'counterpart_area'. Then, it groups the DataFrame by all columns except
    'value', and sums up 'value' within each group.

    Args:
        inflows (pd.DataFrame): The input DataFrame containing inflow data. It is expected to
         have columns including 'iso_code', 'value', and 'counterpart_area'.

    Returns:
        pd.DataFrame: The processed DataFrame.

    """
    # Drop rows with NaN in 'iso_code'
    df = inflows.dropna(subset=["iso_code"])

    # Further drop rows with zero 'value' or 'World' in 'counterpart_area'
    df = df.loc[lambda d: d.value != 0].loc[lambda d: d.counterpart_area != "World"]

    # Group by all columns except 'value', and sum up 'value' within each group
    df = (
        df.astype({"value": "float"})
        .groupby([c for c in df.columns if c != "value"], observed=True, dropna=False)[
            ["value"]
        ]
        .sum()
        .reset_index()
    )

    return df


def rename_indicators(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """
    Rename the indicators in the DataFrame.

    Maps the original indicator names to new ones based on a predefined dictionary.
    The new names are constructed by appending a suffix to the base name of each indicator.
    If an indicator does not exist in the dictionary, its original name is preserved.

    Args:
        df (pd.DataFrame): The input DataFrame with an 'indicator' column that needs to be renamed.
        suffix (str, optional): The suffix to append to the base name of each indicator.
        Defaults to an empty string.

    Returns:
        pd.DataFrame: The DataFrame with renamed indicators.
    """

    indicators = {
        "grants": f"Grants{suffix}",
        "grants_bilateral": f"Bilateral Grants{suffix}",
        "grants_multilateral": f"Multilateral Grants{suffix}",
        "bilateral": f"All bilateral{suffix}",
        "bilateral_non_concessional": f"Bilateral Non-Concessional Debt{suffix}",
        "bilateral_concessional": f"Bilateral Concessional Debt{suffix}",
        "multilateral_non_concessional": f"Multilateral Non-Concessional Debt{suffix}",
        "multilateral_concessional": f"Multilateral Concessional Debt{suffix}",
        "multilateral": f"All multilateral{suffix}",
        "bonds": f"Private - bonds{suffix}",
        "banks": f"Private  - banks{suffix}",
        "other_private": f"Private - other{suffix}",
        "other": f"Private - other{suffix}",
    }
    return df.assign(
        indicator=lambda d: d.indicator.map(indicators).fillna(d.indicator)
    )


def get_all_flows(constant: bool = False) -> pd.DataFrame:
    """
    Retrieve all inflow and outflow data, process them, and combine into a single DataFrame.

    Args:
        constant (bool, optional): A flag to indicate whether to retrieve constant inflow
        and debt service data. Defaults to False.

    Returns:
        pd.DataFrame: The combined DataFrame of processed inflow and outflow data.
    """

    # Get inflow and outflow data
    inflows = (
        get_total_inflows(constant=constant)
        .pipe(prep_flows)
        .pipe(rename_indicators, suffix="")
    )

    # Get outflow data. NOTE: the value of outflow is negative
    outflows = (
        get_debt_service_data(constant=constant)
        .pipe(prep_flows)
        .assign(value=lambda d: -d.value)
        .pipe(rename_indicators, suffix="")
    )

    # Combine inflow and outflow data
    data = (
        pd.concat([inflows, outflows], ignore_index=True)
        .drop(columns=["counterpart_iso_code", "iso_code"])
        .loc[lambda d: d.value != 0]
    )

    return data


def exclude_grant_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude grant indicators from the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.

    Returns:
        pd.DataFrame: The DataFrame with grant indicators excluded.
    """

    return data.loc[lambda d: ~d.indicator.str.contains("grant", case=False)]


def exclude_grant_and_concessional_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude grant and concessional indicators from the DataFrame,
    but keep 'Non-concessional'.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.

    Returns:
        pd.DataFrame: The DataFrame with grant and concessional indicators excluded,
                      except for 'Non-concessional'.
    """
    return data.loc[
        lambda d: ~d.indicator.str.contains("grant", case=False, regex=True)
        & ~(
            d.indicator.str.contains("concessional", case=False, regex=True)
            & ~d.indicator.str.contains("non[- ]?concessional", case=False, regex=True)
        )
    ]


def all_flows_pipeline(
    as_net_flows: bool = False,
    version: AnalysisVersion = "total",
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    china_as_counterpart_type: bool = False,
    constant: bool = False,
) -> pd.DataFrame:
    """Create a dataset with all flows for visualisation.

    Args:
        as_net_flows (bool): If True, convert inflows to net flows.
        version (str): Version of the data to use. Options are:
            - "total": All flows
            - "excluding_grants": Exclude grants
            - "excluding_concessional_finance": Exclude concessional finance (grants and
               concessional loans)
        exclude_outliers (bool): If True, exclude outlier countries (China, Ukraine, Russia).
        remove_countries_wo_outflows (bool): If True, remove countries with missing outflows data.
        china_as_counterpart_type (bool): If True, add China as a counterpart type.
        constant (bool): If True, use constant prices.

    """

    # get constant and current data
    data = get_all_flows(constant=constant).loc[lambda d: d.year <= LATEST_INFLOWS]

    if version == "excluding_grants":
        data = exclude_grant_indicators(data)
    elif version == "excluding_concessional_finance":
        data = exclude_grant_and_concessional_indicators(data)

    if exclude_outliers:
        data = exclude_outlier_countries(data)

    if remove_countries_wo_outflows:
        data = exclude_countries_without_outflows(data)

    if china_as_counterpart_type:
        data = data.pipe(add_china_as_counterpart_type)

        # Save the data
        data = (
            data.groupby(
                [c for c in data.columns if c not in ["value", "counterpart_area"]],
                observed=True,
                dropna=False,
            )["value"]
            .sum()
            .reset_index()
        )

    if as_net_flows:
        data = data.pipe(convert_to_net_flows)

    return data


def net_flows_by_country_pipeline(
    version: AnalysisVersion = "total", constant: bool = False
) -> pd.DataFrame:
    grouper = [
        "year",
        "country",
        "continent",
        "income_level",
        "prices",
        "indicator_type",
    ]

    full_data = all_flows_pipeline(
        as_net_flows=True,
        version=version,
        exclude_outliers=True,
        remove_countries_wo_outflows=True,
        china_as_counterpart_type=False,
        constant=constant,
    )

    by_country = (
        full_data.groupby(grouper, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )

    total = (
        by_country.assign(
            country="Developing countries", income_level="All", continent="World"
        )
        .groupby(grouper, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )

    return pd.concat([total, by_country], ignore_index=True)


if __name__ == "__main__":
    net_flows = net_flows_by_country_pipeline(constant=False)

    net_flows_excluding_grants = net_flows_by_country_pipeline(
        version="excluding_grants", constant=False
    )

    net_flows_excluding_concessional = net_flows_by_country_pipeline(
        version="excluding_concessional_finance", constant=False
    )
