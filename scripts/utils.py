import pandas as pd
from bblocks import add_income_level_column
from pydeflate import set_pydeflate_path, imf_gdp_deflate

from scripts import config
from scripts.data.common import clean_debtors, clean_creditors, add_counterpart_type

set_pydeflate_path(config.Paths.raw_data)


def to_constant_prices(data: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """
    This method takes in a pandas DataFrame 'data' and an integer 'base_year' as input parameters.
    It returns a new pandas DataFrame with constant prices. It uses IMF WEO data to
    deflate the data.

    Args:
        data (pd.DataFrame): The  DataFrame containing the data to be converted to constant prices.
        base_year (int): The base year against which the prices will be deflated.

    Returns:
        pd.DataFrame: A new DataFrame with constant prices.

    """

    # Pass the data to the deflate function and assign a prices column
    data = imf_gdp_deflate(data=data, base_year=base_year, year_column="year").assign(
        prices="constant"
    )

    return data


def clean_debt_output(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the output data frame by replacing bad characters and
    cleaning debtors and creditors.

    Args:
        data (pd.DataFrame): The input data frame containing the data to be cleaned.


    """
    # replace bad characters
    data["counterpart_area"] = data["counterpart_area"].str.strip()

    # clean debtors
    data = clean_debtors(data, "country")

    # clean creditors
    data = clean_creditors(data, "counterpart_area")

    # Convert the year to an integer
    data.year = data.year.dt.year

    # add income level
    data = add_income_level_column(data, id_column="iso_code", id_type="ISO3")

    # drop missing values and values which are zero
    data = data.dropna(subset=["value"]).loc[lambda d: d.value != 0]

    # add counterpart type
    data = add_counterpart_type(data)

    return data
