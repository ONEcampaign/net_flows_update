from typing import Literal

import pandas as pd
from bblocks import (
    add_iso_codes_column,
    set_bblocks_data_path,
    WorldEconomicOutlook,
)
from pydeflate import imf_gdp_deflate

from scripts import config

set_bblocks_data_path(config.Paths.raw_data)


def calculate_growth_rate(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        value=lambda d: d.groupby("iso_code", dropna=False)["value"].pct_change()
    ).dropna(subset=["value"])


def calculate_deflator(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy(deep=True)

    data["value"] = data.groupby("iso_code", dropna=False, observed=True)[
        "value"
    ].transform(lambda x: (1 + x).cumprod())

    return data


def rebase_value(data: pd.DataFrame, year: int) -> pd.DataFrame:
    data["value"] = data.groupby("iso_code")["value"].transform(
        lambda x: x / x.loc[data.year == year].sum()
    )

    return data


def _get_weo_indicator(indicator: str) -> pd.DataFrame:
    weo = WorldEconomicOutlook()

    weo.load_data(indicator)

    return (
        weo.get_data()
        .assign(year=lambda d: d.year.dt.year)
        .dropna(subset=["iso_code"])
        .loc[lambda d: d.year >= config.ANALYSIS_YEARS[0]]
    )


def _get_deflator_from_indicator(indicator: str, base_year: int) -> pd.DataFrame:
    df = (
        _get_weo_indicator(indicator)
        .pipe(calculate_growth_rate)
        .pipe(calculate_deflator)
        .pipe(rebase_value, year=base_year)
        .assign(value=lambda d: d.value.astype(float))
    )

    return df.filter(["iso_code", "year", "value"])


def get_current_gdp(base: int = 2023):
    return _get_deflator_from_indicator(indicator="NGDP", base_year=base)


def constant_gdp_growth_index(base: int = 2023):
    return _get_deflator_from_indicator(indicator="NGDP_R", base_year=base)


def get_constant_deflators(base: int = 2023):
    df = _get_weo_indicator(indicator="NGDP_D").pipe(rebase_value, year=base)

    return df.filter(["iso_code", "year", "value"])


def extend_deflators_to_year(
    data: pd.DataFrame, last_year: int, rolling_window: int
) -> pd.DataFrame:
    """This function creates rows for each donor for the missing years between the
    max year in the data and the last year specified in the arguments. The value is
    rolling average of the previous 3 years"""

    def fill_with_rolling_average(
        idx, group: pd.DataFrame, rolling_window: int = 3
    ) -> pd.DataFrame:
        # calculate yearly diff
        group["yearly_diff"] = group["value"].diff()

        new_index = pd.Index(
            range(group.year.max() - rolling_window, last_year + 1), name="year"
        )

        new_df = group.set_index("year").reindex(new_index)
        new_df["yearly_diff"] = (
            new_df["yearly_diff"].rolling(window=rolling_window).mean()
        )
        new_df["yearly_diff"] = new_df["yearly_diff"].ffill()
        new_df["value"] = new_df.value.fillna(
            (new_df["value"].shift(1).ffill() + new_df["yearly_diff"].shift(1).cumsum())
        )

        new_df = new_df.drop(columns=["yearly_diff"])
        group = group.drop(columns=["yearly_diff"])

        new_df[["iso_code"]] = idx

        new_df = new_df.loc[lambda d: d.index > group.year.max()]

        group = group
        new_df = new_df.reset_index()

        group = pd.concat([group, new_df], ignore_index=False)
        return group

    data = data

    dfs = []

    for group_idx, group_data in data.groupby(
        ["iso_code"], dropna=False, observed=True
    ):
        dfs.append(
            fill_with_rolling_average(
                idx=group_idx, group=group_data, rolling_window=rolling_window
            )
        )

    return pd.concat(dfs, ignore_index=True)


def future_exchange_deflators(base_year: int = 2023) -> pd.DataFrame:
    weo = WorldEconomicOutlook()
    weo.load_data(["NGDPD", "NGDP"])

    national = weo.get_data("NGDP").filter(["iso_code", "year", "value"])
    dollars = weo.get_data("NGDPD").filter(["iso_code", "year", "value"])

    data = national.merge(dollars, on=["year", "iso_code"], suffixes=("_lcu", "_usd"))
    data["year"] = data["year"].dt.year

    data["exchange"] = data["value_usd"] / data["value_lcu"]

    base_data = (
        data.loc[lambda d: d.year == base_year]
        .filter(["iso_code", "exchange"])
        .rename(columns={"exchange": "base_exchange"})
    )

    data = data.merge(base_data, on=["iso_code"])

    data["exchange_deflator"] = 100 * data["exchange"] / data["base_exchange"]

    return data.filter(["year", "iso_code", "exchange_deflator"])


def get_future_gni(
    gni_data: pd.DataFrame,
    prices: Literal["constant", "current"],
    base_year: int = 2023,
) -> pd.DataFrame:

    max_in_data = gni_data.year.max()
    latest = gni_data.loc[lambda d: d.year == max_in_data].filter(["iso_code", "value"])

    if prices == "constant":
        growth = constant_gdp_growth_index(base=base_year)
    else:
        growth = get_current_gdp(base=base_year)

    growth = growth.loc[lambda d: d.year > max_in_data].rename(
        columns={"value": "growth_ratio"}
    )

    growth = growth.merge(latest, on=["iso_code"])
    growth[f"{prices}_usd_gni"] = growth["value"] * growth["growth_ratio"]

    return growth.filter(["year", "iso_code", f"{prices}_usd_gni"])


def _get_oda_indicator(
    indicator: str, start_year: int, end_year: int, base_year: int | None
) -> pd.DataFrame:
    from oda_data import set_data_path, ODAData

    set_data_path(config.Paths.raw_data)
    oda = ODAData(
        years=range(start_year, end_year + 1),
        base_year=base_year,
        prices="constant" if base_year else "current",
    )
    oda.load_indicator(indicator)

    data = oda.get_data()

    data = data.pipe(
        add_iso_codes_column, id_column="donor_code", id_type="DACCode"
    ).loc[lambda d: d.iso_code.str.len() == 3]

    return data


def current_deflator_series(
    base_year: int = 2023,
    start_year: int = config.ANALYSIS_YEARS[0],
    end_year: int = 2029,
) -> pd.DataFrame:
    """"""

    # get  gni
    gni = _get_oda_indicator(
        start_year=start_year, end_year=base_year, indicator="gni", base_year=None
    )

    # get GNI deflators
    gdp_deflators = get_current_gdp(base=base_year).rename(
        columns={"value": "lcu_lcu_deflator"}
    )

    # get future gni
    future_gni = get_future_gni(gni, prices="current", base_year=base_year)

    gni = pd.concat(
        [
            gni.filter(["year", "iso_code", "value"]).rename(
                columns={"value": f"current_usd_gni"}
            ),
            future_gni,
        ],
        ignore_index=True,
    )

    gni = gni.merge(
        gdp_deflators,
        on=["year", "iso_code"],
        how="left",
    )

    return gni


def constant_deflator_series(
    base_year: int = 2023,
    start_year: int = config.ANALYSIS_YEARS[0],
    end_year: int = 2029,
):
    # get gni
    gni = _get_oda_indicator(
        start_year=start_year,
        end_year=base_year,
        indicator="gni",
        base_year=base_year,
    )

    # get future gni
    future_gni = get_future_gni(gni, prices="constant", base_year=base_year)

    gni = pd.concat(
        [
            gni.filter(["year", "iso_code", "value"]).rename(
                columns={"value": f"constant_usd_gni"}
            ),
            future_gni,
        ],
        ignore_index=True,
    )

    # Unique iso_codes
    iso_codes = gni["iso_code"].unique()

    # Create a dataframe with the Cartesian product of years and iso_codes
    data = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [range(start_year, end_year + 1), iso_codes], names=["year", "iso_code"]
        )
    ).reset_index()

    # Add 'value' column
    data["value"] = 100

    # Deflate usd
    usd_data = (
        imf_gdp_deflate(data=data, base_year=base_year, to_current=True)
        .dropna(subset=["value"])
        .rename(columns={"value": "usd_usd_deflator"})
    )

    lcu_data = (
        imf_gdp_deflate(
            data=data,
            base_year=base_year,
            source_currency="LCU",
            target_currency="LCU",
            id_column="iso_code",
            to_current=True,
        )
        .dropna(subset=["value"])
        .rename(columns={"value": "lcu_lcu_deflator"})
    )

    # future LCU (e.g price deflators)
    future_lcu = lcu_data.loc[lambda d: d.year >= 2023]

    ex_defl = future_exchange_deflators(2023)

    # usd deflators
    future_usd = future_lcu.merge(ex_defl, on=["year", "iso_code"])
    future_usd["usd_usd_deflator"] = (
        100 * future_usd["lcu_lcu_deflator"] / future_usd["exchange_deflator"]
    )

    future_usd = future_usd.loc[lambda d: d.year > 2023].filter(
        ["year", "iso_code", "usd_usd_deflator"]
    )
    future_lcu = future_lcu.loc[lambda d: d.year > 2023].filter(
        ["year", "iso_code", "lcu_lcu_deflator"]
    )

    usd = pd.concat(
        [usd_data.loc[lambda d: d.year <= 2023], future_usd], ignore_index=True
    )
    lcu = pd.concat(
        [lcu_data.loc[lambda d: d.year <= 2023], future_lcu], ignore_index=True
    )

    data = usd.merge(lcu, on=["year", "iso_code"], how="left").merge(
        gni, on=["year", "iso_code"], how="left"
    )

    values = ["usd_usd_deflator", "lcu_lcu_deflator", "constant_usd_gni"]

    data[values] = data[values].astype(float).round(4)

    return data


if __name__ == "__main__":
    df = current_deflator_series()
