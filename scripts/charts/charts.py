"""Module to create the chart data for the page"""

import pandas as pd

from scripts.config import Paths
from scripts.utils import custom_sort


net_flows_data = pd.read_csv(Paths.raw_data / "net_flows.csv")


def add_income_level_aggregates(df):

    agg_df = (df
              .loc[lambda d: d.country != "Developing countries"]
              .groupby(['year', 'income_level', 'indicator_type', 'flow_type', "prices"], dropna=True)
              .agg({"value": "sum"})
              .reset_index()
              .rename(columns = {"income_level": "country"})
              .assign(income_level = None,
                      continent = None
                      )

              )

    return pd.concat([df, agg_df.loc[lambda d: d.country != "High income"]], ignore_index=True)


def add_africa_aggregate(df):

    afr_agg = (df
               .loc[lambda d: d.continent == "Africa"]
               .groupby(['year', 'indicator_type', 'flow_type', "prices"], dropna=True)
               .agg({"value": "sum"})
               .reset_index()
               .assign(income_level = None,
                       continent = None,
                       country = "Africa"
                       )
               )

    return pd.concat([df, afr_agg], ignore_index=True)


def chart_1():
    """ Create data for chart 1: net flows comparison with/without concessional finance """

    df = (net_flows_data
     .pipe(add_africa_aggregate)
     .pipe(add_income_level_aggregates)
     .loc[lambda d: d.indicator_type == 'net_flow']
     .pivot(index = ['year', 'country'], columns = "flow_type", values="value")
     .reset_index()
     .pipe(custom_sort, "country", ['Developing countries', "Africa", "Low income", "Lower middle income", "Upper middle income"])
          )

    # download data
    (df
     .rename(columns = {"all": "net flows including aid and concessional finance", "excluding_concessional": "net flows excluding aid and concessional finance"})
     .assign(unit = "current US$")
     .to_csv(Paths.output / "chart_1_download.csv", index=False)
     )

    # chart data
    (df
     .rename(columns = {"all": "all net flows", "excluding_concessional": "net flows excluding aid and concessional finance"})
     .to_csv(Paths.output / "chart_1.csv", index=False)
     )

