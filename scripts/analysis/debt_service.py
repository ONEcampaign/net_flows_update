from typing import Literal

import pandas as pd

from scripts.analysis.common import (
    all_flows_pipeline,
    exclude_grant_and_concessional_indicators,
    create_dev_countries_total,
    OUTPUT_GROUPER,
)

LAST_ANALYSIS_YEAR: int = 2027


def get_debt_service(
    version: Literal["total", "excluding_concessional_finance"],
    constant: bool = False,
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    china_as_counterpart_type: bool = False,
) -> pd.DataFrame:
    """
    Get outflow data (historical and projected) for the specified version.

    Args:
        version (Literal["total", "excluding_concessional_finance"]): The version of the data to retrieve.
        constant (bool): Whether to use constant values.
        exclude_outliers (bool): Whether to exclude outliers (China, Russia, Ukraine).
        remove_countries_wo_outflows (bool): Whether to remove countries without outflows.
        china_as_counterpart_type (bool): Whether to treat China as a counterpart type.
    """
    outflows_data = (
        all_flows_pipeline(
            as_net_flows=False,
            version="total",
            exclude_outliers=exclude_outliers,
            remove_countries_wo_outflows=remove_countries_wo_outflows,
            china_as_counterpart_type=china_as_counterpart_type,
            constant=constant,
            exclude_outflow_estimates=False,
        )
        .loc[lambda d: d.indicator_type == "outflow"]
        .loc[lambda d: d.year <= LAST_ANALYSIS_YEAR]
    )

    if version == "excluding_concessional_finance":
        outflows_data = exclude_grant_and_concessional_indicators(data=outflows_data)

    total_data = create_dev_countries_total(data=outflows_data)

    outflows_data = pd.concat([total_data, outflows_data], ignore_index=True)

    return (
        outflows_data.groupby(
            [
                c
                for c in outflows_data.columns
                if c not in ["indicator", "indicator_type", "value"]
            ],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
        .assign(value=lambda d: d.value * -1)
    )


def debt_service_by_period(debt_service_data: pd.DataFrame) -> pd.DataFrame:
    """
    Group debt service data by specified periods and calculate average values.

    Args:
        debt_service_data (pd.DataFrame): The debt service data to group.
    Returns:
        pd.DataFrame: The grouped DataFrame with average values.

    """

    periods = {
        "2010-2014": {"length": 5, "years": (2010, 2014)},
        "2018-2022": {"length": 5, "years": (2018, 2022)},
        "2024-2025 (projected)": {"length": 2, "years": (2024, 2025)},
    }

    for period, settings in periods.items():
        start_year, end_year = settings["years"]
        debt_service_data.loc[
            (debt_service_data.year >= start_year)
            & (debt_service_data.year <= end_year),
            "period",
        ] = period

    # Remove rows with NaN in the 'period' column
    debt_service_data = debt_service_data.dropna(subset=["period"])

    # Group by period and sum the values
    debt_service_data = (
        debt_service_data.groupby(
            [
                "period",
                "country",
                "continent",
                "income_level",
                "prices",
                "counterpart_type",
            ],
            observed=True,
            dropna=False,
        )["value"]
        .sum()
        .reset_index()
    )

    debt_service_data["length"] = debt_service_data["period"].map(
        lambda x: periods[x]["length"] if x in periods else None
    )

    debt_service_data["value"] = (
        debt_service_data["value"] / debt_service_data["length"]
    ).round(2)

    return debt_service_data.drop(columns=["length"])


if __name__ == "__main__":
    debt_service = get_debt_service(
        version="total",
        china_as_counterpart_type=True,
    )

    # To recreate chart data
    ds_by_period = debt_service_by_period(debt_service)
