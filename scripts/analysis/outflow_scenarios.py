from typing import Literal, TypeAlias

import pandas as pd

from scripts.analysis.common import (
    all_flows_pipeline,
    OUTPUT_GROUPER,
    create_dev_countries_total,
    LATEST_INFLOWS,
    exclude_grant_and_concessional_indicators,
)

from scripts.config import Paths

Percent: TypeAlias = int


def get_wb_projected_outflows(
    version: Literal["total", "excluding_concessional_finance"],
    constant: bool = False,
    exclude_outliers: bool = True,
    remove_countries_wo_outflows: bool = True,
    china_as_counterpart_type: bool = False,
) -> pd.DataFrame:
    """
    Get outflow projections data
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
        .loc[lambda d: d.year > LATEST_INFLOWS]
    )

    if version == "excluding_concessional_finance":
        outflows_data = exclude_grant_and_concessional_indicators(data=outflows_data)

    total_data = create_dev_countries_total(data=outflows_data)

    outflows_data = pd.concat([total_data, outflows_data], ignore_index=True)

    return (
        outflows_data.groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )


if __name__ == "__main__":
    # --- Scenarios ---

    scenario_total_outflows = get_wb_projected_outflows(
        version="total",
        exclude_outliers=True,
        remove_countries_wo_outflows=True,
        china_as_counterpart_type=False,
    )

    scenario_excluding_concessional_finance = get_wb_projected_outflows(
        version="excluding_concessional_finance",
        exclude_outliers=True,
        remove_countries_wo_outflows=True,
        china_as_counterpart_type=False,
    )

    scenario_total_outflows.to_csv(Paths.raw_data / "outflows_scenarios.csv", index=False)
