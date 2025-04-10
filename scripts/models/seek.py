import pandas as pd
from bblocks import add_iso_codes_column

from scripts.config import Paths

SEEK_FILE = "202502_seek_emeea.xlsx"
IDX = ["year", "donor"]

indicators = {
    "gni": ["gni", "gni_downside", "gni_realistic", "gni_upside"],
    "oda": ["oda_downside", "oda_realistic", "oda_upside"],
    "oda_gni": ["oda_gni_downside", "oda_gni_realistic", "oda_gni_upside"],
    "bilateral": ["bilateral_upside", "bilateral_realistic"],
    "multilateral": ["multilateral_upside", "multilateral_realistic"],
    "idrc": ["idrc", "idrc_downside"],
    "scholarship": ["scholarship"],
    "administrative": ["administrative"],
    "development_awareness": ["development_awareness"],
    "eui_contributions": ["eui_contributions"],
    "ukraine": ["ukraine", "ukraine_downside"],
    "health_bilateral": ["health_bilateral_upside", "health_bilateral_realistic"],
    "agriculture_bilateral": [
        "agriculture_bilateral_upside",
        "agriculture_bilateral_realistic",
    ],
    "wash_bilateral": ["wash_bilateral_upside", "wash_bilateral_realistic"],
    "wb_multilateral": ["wb_multilateral_upside", "wb_multilateral_realistic"],
    "rdb_multilateral": ["rdb_multilateral_upside", "rdb_multilateral_realistic"],
    "un_multilateral": ["un_multilateral_upside", "un_multilateral_realistic"],
}


def clean_column_names(columns: pd.Index) -> pd.Index:
    """Clean column names: lower case, replace spaces, remove parenthetical info, etc"""
    return (
        columns.str.lower()
        .str.replace(" ", "_")
        .str.replace(r"\(.*?\)", "", regex=True)
        .str.strip("_")
        .str.replace("bl", "bilateral", regex=False)
        .str.replace("ml", "multilateral", regex=False)
        .str.replace("ag", "agriculture", regex=False)
        .str.replace("_value", "", regex=False)
    )


def read_projections() -> pd.DataFrame:
    """Read the SEEK projections data."""
    data = pd.read_excel(
        Paths.models / SEEK_FILE, sheet_name="Projections constant 2023 price"
    )

    # Clean column names
    data.columns = clean_column_names(data.columns)

    return data


def get_seek_indicator(indicator: str) -> pd.DataFrame:
    df = read_projections().filter(IDX + indicators[indicator])

    # Remove the indicator prefix from column names
    df.columns = [c.replace(f"{indicator}_", "") for c in df.columns]

    # Rename the column that matches the indicator to "realistic"
    df.columns = ["realistic" if indicator == c else c for c in df.columns]
    return df


def load_deflators():
    return pd.read_excel(Paths.models / "deflators_one.xlsx")[
        ["year", "iso_code", "usd_usd_deflator"]
    ]


def extract_decreases():
    """Calculate year-on-year multiplicative factors from 2023 baseline values for each scenario."""

    data = (
        get_seek_indicator("oda")
        .pipe(add_iso_codes_column, id_column="donor", id_type="regex")
        .merge(load_deflators(), how="left", on=["year", "iso_code"])
    )

    scenarios = ["downside", "realistic", "upside"]

    for col in scenarios:
        data[col] = data[col] * data["usd_usd_deflator"]

    # Get the 2023 values per iso_code and scenario
    baseline = data[data["year"] == 2023][["iso_code"] + scenarios].set_index(
        "iso_code"
    )

    # Merge baseline back into the full dataset
    for scenario in scenarios:
        data = data.merge(
            baseline[[scenario]].rename(columns={scenario: f"{scenario}_2023"}),
            how="left",
            on="iso_code",
        )
        # Compute multiplicative factor
        data[f"{scenario}_multiplier"] = data[scenario] / data[f"{scenario}_2023"]

    return data.filter(["year", "iso_code", "donor", "realistic_multiplier"])


def apply_linear_reduction(
    data: pd.DataFrame,
    reduction: float,
    start_year: int,
    end_year: int,
    multiplier_col: str = "realistic_multiplier",
    output_col: str = "realistic_multiplier_reduced",
) -> pd.DataFrame:
    """Apply a linear reduction to a multiplier column over a given time range,
    and keep the final reduction constant beyond the end year.

    Args:
        data: DataFrame with at least 'year', 'iso_code', and the multiplier column.
        reduction: Total amount to reduce the multiplier by over the period.
        start_year: The year to start applying the reduction.
        end_year: The year to finish applying the reduction.
        multiplier_col: The name of the column with the original multiplier.
        output_col: The name of the new column to store the reduced multiplier.

    Returns:
        DataFrame with an additional column applying the linear reduction.
    """
    data = data.copy()
    data[output_col] = data[multiplier_col]

    num_years = end_year - start_year + 1
    if num_years <= 0:
        raise ValueError("end_year must be greater than or equal to start_year.")

    for i, year in enumerate(range(start_year, end_year + 1)):
        year_reduction = reduction * ((i + 1) / num_years)
        mask = data["year"] == year
        data.loc[mask, output_col] = data.loc[mask, multiplier_col] - year_reduction

    # Apply full reduction from end_year + 1 onwards
    mask_later = data["year"] > end_year
    data.loc[mask_later, output_col] = data.loc[mask_later, multiplier_col] - reduction

    return data
