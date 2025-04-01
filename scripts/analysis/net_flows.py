from typing import Literal

import pandas as pd
from bblocks import set_bblocks_data_path

from scripts.analysis.common import (
    convert_to_net_flows,
    all_flows_pipeline,
    AnalysisVersion,
    OUTPUT_GROUPER,
    create_dev_countries_total,
)
from scripts.config import Paths

set_bblocks_data_path(Paths.raw_data)


def net_flows_by_country_pipeline(
    version: AnalysisVersion = "total",
    as_net_flows: bool = True,
    constant: bool = False,
) -> pd.DataFrame:
    """Create a dataset with all flows for visualisation.

    Args:
        version (str): Version of the data to use. Options are:
            - "total": All flows
            - "excluding_grants": Exclude grants
            - "excluding_concessional_finance": Exclude concessional finance (grants and
               concessional loans)
        as_net_flows (bool): If True, convert inflows - outflows to net flows.
        constant (bool): If True, use constant prices.

    """

    full_data = all_flows_pipeline(
        as_net_flows=as_net_flows,
        version=version,
        exclude_outliers=True,
        remove_countries_wo_outflows=True,
        china_as_counterpart_type=False,
        constant=constant,
    )

    total_data = create_dev_countries_total(data=full_data)

    data = pd.concat([total_data, full_data], ignore_index=True)

    data = (
        data.groupby(OUTPUT_GROUPER, observed=True, dropna=False)["value"]
        .sum()
        .reset_index()
    )

    return data


if __name__ == "__main__":
    # Get all flows and net flows
    all_flows = net_flows_by_country_pipeline(as_net_flows=False)
    inflows = all_flows.query("indicator_type == 'inflow'")
    outflows = all_flows.query("indicator_type == 'outflow'")
    net_flows = all_flows.pipe(convert_to_net_flows)

    # Exclude grants
    all_flows_excluding_grants = net_flows_by_country_pipeline(
        version="excluding_grants", as_net_flows=False
    )
    inflows_excluding_grants = all_flows_excluding_grants.query(
        "indicator_type == 'inflow'"
    )
    outflows_excluding_grants = all_flows_excluding_grants.query(
        "indicator_type == 'outflow'"
    )
    net_flows_excluding_grants = all_flows_excluding_grants.pipe(convert_to_net_flows)

    # Exclude concessional finance
    all_flows_excluding_concessional = net_flows_by_country_pipeline(
        version="excluding_concessional_finance", as_net_flows=False
    )
    inflows_excluding_concessional = all_flows_excluding_concessional.query(
        "indicator_type == 'inflow'"
    )
    outflows_excluding_concessional = all_flows_excluding_concessional.query(
        "indicator_type == 'outflow'"
    )
    net_flows_excluding_concessional = all_flows_excluding_concessional.pipe(
        convert_to_net_flows
    )
