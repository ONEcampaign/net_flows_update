import pandas as pd

from scripts.analysis.common import (
    all_flows_pipeline,
    AnalysisVersion,
    create_dev_countries_total,
)


def historical_inflows(
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


if __name__ == "__main__":
    total_inflows = historical_inflows(china_as_counterpart_type=True)
