import json
import os
from typing import Literal

import pandas as pd

from scripts.data.inflows import get_total_inflows
from scripts.data.outflows import get_debt_service_data

LATEST_INFLOWS: int = 2023

AnalysisVersion = Literal["total", "excluding_grants", "excluding_concessional_finance"]

OUTPUT_GROUPER = [
    "year",
    "country",
    "continent",
    "income_level",
    "prices",
    "indicator_type",
]

GROUPS = {
    "Developing countries": 1,
    "Low income": 2,
    "Lower middle income": 3,
    "Upper middle income": 4,
    "Africa": 7,
    "Europe": 8,
    "Asia": 9,
    "America": 10,
    "Oceania": 11,
}


def create_grouping_totals(
    data: pd.DataFrame, group_column: str, exclude_cols: list[str]
) -> pd.DataFrame:
    """Create group totals as 'country'"""

    dfs = []

    for group in data[group_column].unique():
        df_ = data.loc[lambda d: d[group_column] == group].copy()
        df_["country"] = group
        df_ = (
            df_.groupby(
                [c for c in df_.columns if c not in ["value"] + exclude_cols],
                observed=True,
                dropna=False,
            )["value"]
            .sum()
            .reset_index()
        )

        dfs.append(df_)

    groups = pd.concat(dfs, ignore_index=True)

    return pd.concat([data, groups], ignore_index=True)


def exclude_outlier_countries(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[lambda d: ~d.country.isin(["China", "Ukraine", "Russia"])]

    return data


def create_world_total(data: pd.DataFrame, name: str = "World") -> pd.DataFrame:
    """Create a world total for the data"""

    df = data.copy(deep=True)
    df["country"] = name
    df = (
        df.groupby(
            [c for c in df.columns if c not in ["income_level", "continent", "value"]],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )

    return pd.concat([data, df], ignore_index=True)


def add_china_as_counterpart_type(df: pd.DataFrame) -> pd.DataFrame:
    """Adds China as counterpart type"""

    # Get china as counterpart
    china = df.loc[lambda d: d.counterpart_area == "China"].copy()

    # Add counterpart type, by type
    china["counterpart_type"] = "China"

    # Remove China from the original data
    df = df.loc[lambda d: d.counterpart_area != "China"]

    # Concatenate the data
    return pd.concat([df, china], ignore_index=True)


def convert_to_net_flows(data: pd.DataFrame) -> pd.DataFrame:
    """Group the indicator type to get net flows"""

    data = (
        data.groupby(
            [c for c in data.columns if c not in ["value", "indicator_type"]],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )

    data["indicator_type"] = "net_flow"

    return data


def summarise_by_country(data: pd.DataFrame) -> pd.DataFrame:
    """Summarise the data by country"""

    data = (
        data.groupby(
            [
                c
                for c in data.columns
                if c
                not in [
                    "value",
                    "counterpart_area",
                    "counterpart_type",
                    "indicator",
                ]
            ],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )

    return data


def create_groupings(data: pd.DataFrame) -> pd.DataFrame:
    # Create world totals
    data_grouped = create_world_total(data, "Developing countries")

    # Create continent totals
    data_grouped = create_grouping_totals(
        data_grouped, group_column="continent", exclude_cols=["income_level"]
    )

    # Create income_level totals
    data_grouped = create_grouping_totals(
        data_grouped, group_column="income_level", exclude_cols=["continent"]
    )

    # remove individual country data
    data_grouped = data_grouped.loc[lambda d: d.country.isin(GROUPS)]

    return data_grouped


def reorder_countries(df: pd.DataFrame, counterpart_type: bool = False) -> pd.DataFrame:
    """Reorder countries by continent and income level"""

    df["order"] = df["country"].map(GROUPS).fillna(99)

    counterpart_order = {
        "Bilateral": 1,
        "Multilateral": 2,
        "Private": 3,
        "China": 4,
    }

    if counterpart_type:
        df["order_counterpart"] = (
            df["counterpart_type"].map(counterpart_order).fillna(99)
        )

    df = (
        df.sort_values(
            ["order", "country", "year", "order_counterpart"]
            if counterpart_type
            else ["order", "country", "year"]
        )
        .drop(columns=["order", "order_counterpart"] if counterpart_type else ["order"])
        .reset_index(drop=True)
    )

    return df


def update_key_number(path: str, new_dict: dict) -> None:
    """Update a key number json by updating it with a new dictionary"""

    # Check if the file exists, if not create
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump({}, f)

    with open(path, "r") as f:
        data = json.load(f)

    for k in new_dict.keys():
        data[k] = new_dict[k]

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def exclude_countries_without_outflows(data: pd.DataFrame) -> pd.DataFrame:
    df_pivot = (
        data.query("prices == 'current'")
        .groupby(["year", "country", "indicator_type"], observed=True, dropna=False)[
            "value"
        ]
        .sum()
        .reset_index()
        .pivot(index=["year", "country"], columns="indicator_type", values="value")
        .reset_index()
    )

    new_data = []

    # for each year, from the original data remove countries where "outflow" is missing
    for year in df_pivot["year"].unique():
        countries = (
            df_pivot.loc[lambda d: d.year == year]
            .loc[lambda d: d.outflow.notna()]["country"]
            .unique()
        )
        d_ = data.loc[lambda d: d.country.isin(countries)].loc[lambda d: d.year == year]
        new_data.append(d_)

    return pd.concat(new_data, ignore_index=True)


def mask_grant_indicators(data: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for rows that are NOT grant indicators.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.

    Returns:
        pd.Series: A boolean mask where True means the row is NOT a grant indicator.
    """
    return ~data.indicator.str.contains("grant", case=False)


def mask_grant_and_concessional_indicators(data: pd.DataFrame) -> pd.Series:
    """
    Return a boolean mask for rows that are NOT grant or concessional indicators,
    except it keeps 'Non-concessional'.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.

    Returns:
        pd.Series: A boolean mask where True means the row is NOT a grant or concessional
                   indicator, with 'Non-concessional' preserved.
    """
    is_grant = data.indicator.str.contains("grant", case=False)
    is_concessional = data.indicator.str.contains("concessional", case=False)
    is_non_concessional = data.indicator.str.contains(
        "non[- ]?concessional", case=False
    )

    return ~(is_grant | (is_concessional & ~is_non_concessional))


def exclude_grant_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Exclude grant indicators from the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.

    Returns:
        pd.DataFrame: The DataFrame with grant indicators excluded.
    """
    return data.loc[mask_grant_indicators(data)]


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
    return data.loc[mask_grant_and_concessional_indicators(data)]


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


def exclusions(
    data: pd.DataFrame,
    exclude_outliers: bool,
    remove_countries_wo_outflows: bool,
    china_as_counterpart_type: bool,
) -> pd.DataFrame:
    """
    Apply exclusions to the DataFrame based on specified criteria.

    Args:
        data (pd.DataFrame): The input DataFrame containing inflow and outflow data.
        exclude_outliers (bool): If True, exclude outlier countries (China, Ukraine, Russia).
        remove_countries_wo_outflows (bool): If True, remove countries with missing outflows data.
        china_as_counterpart_type (bool): If True, add China as a counterpart type.

    Returns:
        pd.DataFrame: The DataFrame after applying the specified exclusions.
    """
    if exclude_outliers:
        data = exclude_outlier_countries(data)

    if remove_countries_wo_outflows:
        data = exclude_countries_without_outflows(data)

    if china_as_counterpart_type:
        data = data.pipe(add_china_as_counterpart_type)

        data = (
            data.groupby(
                [c for c in data.columns if c not in ["value", "counterpart_area"]],
                observed=True,
                dropna=False,
            )["value"]
            .sum()
            .reset_index()
        )

    return data


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
    inflows_data = get_total_inflows(constant=constant).pipe(prep_flows)

    # Get outflow data. NOTE: the value of outflow is negative
    outflows_data = (
        get_debt_service_data(constant=constant)
        .pipe(prep_flows)
        .assign(value=lambda d: -d.value)
    )

    # Combine inflow and outflow data
    data = (
        pd.concat([inflows_data, outflows_data], ignore_index=True)
        .drop(columns=["counterpart_iso_code", "iso_code"])
        .loc[lambda d: d.value != 0]
    )

    return data


def all_flows_pipeline(
    as_net_flows: bool = False,
    version: AnalysisVersion = "total",
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    china_as_counterpart_type: bool = False,
    constant: bool = False,
    exclude_outflow_estimates: bool = True,
) -> pd.DataFrame:
    """Create a dataset with all flows for visualisation.

    Args:
        as_net_flows (bool): If True, convert inflows - outflows to net flows.
        version (str): Version of the data to use. Options are:
            - "total": All flows
            - "excluding_grants": Exclude grants
            - "excluding_concessional_finance": Exclude concessional finance (grants and
               concessional loans)
        exclude_outliers (bool): If True, exclude outlier countries (China, Ukraine, Russia).
        remove_countries_wo_outflows (bool): If True, remove countries with missing outflows data.
        china_as_counterpart_type (bool): If True, add China as a counterpart type.
        constant (bool): If True, use constant prices.
        exclude_outflow_estimates (bool): If True, exclude outflow estimates.

    """

    # get constant and current data
    data = get_all_flows(constant=constant)

    if exclude_outflow_estimates:
        data = data.loc[lambda d: d.year <= LATEST_INFLOWS]

    if version == "excluding_grants":
        data = exclude_grant_indicators(data)
    elif version == "excluding_concessional_finance":
        data = exclude_grant_and_concessional_indicators(data)

    data = exclusions(
        data,
        exclude_outliers=exclude_outliers,
        remove_countries_wo_outflows=remove_countries_wo_outflows,
        china_as_counterpart_type=china_as_counterpart_type,
    )

    if as_net_flows:
        data = data.pipe(convert_to_net_flows)

    return data


def create_dev_countries_total(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.assign(
            country="Developing countries",
            income_level="All",
            continent="World",
            iso_code="DEV",
        )
        .groupby(
            [c for c in data.columns if c != "value"],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )
