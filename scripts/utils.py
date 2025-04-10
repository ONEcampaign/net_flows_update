from functools import lru_cache

import pandas as pd
from bblocks import add_income_level_column
from pydeflate import set_pydeflate_path, imf_gdp_deflate
import bblocks_data_importers as bbdata

# import lru_cache
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


def custom_sort(df: pd.DataFrame, col: str, custom_list: list) -> pd.DataFrame:
    """Custom sort function for a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame to sort.
        col (str): The column name to sort by.
        custom_list (list): The custom order for sorting.

    Returns:
        The sorted DataFrame.

    """
    def sorting_key(value):
        # If the value is in the custom list, return its index, otherwise return a large number
        return (custom_list.index(value) if value in custom_list else len(custom_list), str(value))

    # Sort the DataFrame using the custom key
    df = df.loc[sorted(df.index, key=lambda x: sorting_key(df.loc[x, col]))]
    return df.reset_index(drop=True)

@lru_cache
def get_gni():
    """Get a dataframe with GNI values"""

    wb = bbdata.WorldBank()

    return wb.get_data("NY.GNP.ATLS.CD").loc[:, ['year', 'entity_code', 'value']].rename(columns = {"value":'gni'})

def add_gni(df):
    """ """

    gni = get_gni()

    # Merge the GNI data with the original DataFrame
    return df.merge(gni, how='left', on=["year", "entity_code"])

@lru_cache
def get_gni_pc():
    """Get a dataframe with GNI per capita values"""

    wb = bbdata.WorldBank()

    # Get GNI per capita data from World Bank
    return wb.get_data("NY.GNP.PCAP.CD").loc[:, ['year', 'entity_code', 'value']].rename(columns = {"value":'gni_pc'})

def add_gni_pc(df):
    """ """

    # Get GNI per capita data
    gni_pc = get_gni_pc()

    # Merge the GNI per capita data with the original DataFrame
    return df.merge(gni_pc, how='left', on=["year", "entity_code"])
