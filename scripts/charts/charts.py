"""Module to create the chart data for the page"""

import pandas as pd
import country_converter as coco

from scripts.config import Paths
from scripts.utils import custom_sort, add_gni, add_gni_pc


net_flows_data = pd.read_csv(Paths.raw_data / "net_flows.csv")
inflows_data = pd.read_csv(Paths.raw_data / "total_inflows.csv")
outflows_avg_data = pd.read_csv(Paths.raw_data / "debt_service_by_period.csv")

def chart_1():
    """ Create data for chart 1: net flows comparison with/without concessional finance """

    df = (net_flows_data
     # .pipe(add_africa_aggregate)
     # .pipe(add_income_level_aggregates)
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
     .rename(columns = {"all": "all net flows", "excluding_concessional": "net flows excluding concessional finance"})
     .to_csv(Paths.output / "chart_1.csv", index=False)
     )


def chart_2():
    """Connected dot African countries net flows in 2023"""

    df = (net_flows_data
          .loc[lambda d: (d.indicator_type == "net_flow")&(d.continent == "Africa")&(d.year == 2023), ['year', 'country', 'income_level', 'value', 'flow_type']]
          .assign(entity_code = lambda d: coco.CountryConverter().pandas_convert(d.country))
          .pipe(add_gni)
          .assign(value_pct_gni = lambda d: (d.value/d.gni)*100)
          .pipe(add_gni_pc)
          .sort_values(by = ['flow_type', 'gni_pc'])
          .dropna(subset="value_pct_gni")
          .assign(flow_type = lambda d: d.flow_type.map({"all": "all flows", "excluding_concessional": "excluding concessional finance"}))

          )

    # download data
    (df
     .rename(columns = {"value_pct_gni": "net flows as % of GNI",
                        "gni_pc": "GNI per capita, Atlas method (current US$)",
                       "value": "net flows (current US$)",
                          "gni": "GNI, Atlas method (current US$)",
                       })

     .to_csv(Paths.output / "chart_2_download.csv", index=False)
     )

    # chart data
    (pd.concat([pd.DataFrame([[""] * 1], columns=['country']), # add an empty row at the top
                df])
    .assign(value = lambda d: round(d.value/1e9,2))
     .to_csv(Paths.output / "chart_2.csv", index=False)
     )


def chart_3():
    """Line chart new debt inflows"""

    df = (inflows_data.pivot(index=['year', 'country'], columns='counterpart_type', values='value')
     .reset_index()
     .pipe(custom_sort, "country", ['Developing countries', "Africa", "Low income", "Lower middle income", "Upper middle income"])
     )

    df.to_csv(Paths.output / "chart_3.csv", index=False)
    df.assign(prices = "current").to_csv(Paths.output / "chart_3_data.csv", index=False)


def chart_4():
    """ """

    o1 = (outflows_avg_data
          .loc[lambda d: d.period.isin(['2010-2014', '2018-2022'])]
          .assign(code = lambda d: d.counterpart_type + " " + "data")
          )

    o2 = (outflows_avg_data
          .loc[lambda d: d.period.isin(['2018-2022', "2024-2025 (projected)"])]
          .assign(code = lambda d: d.counterpart_type)
          )

    df = (pd.concat([o1, o2])
     .pivot(index=['country', 'period', 'counterpart_type'], columns = 'code', values='value')
     .reset_index()
     .pipe(custom_sort, "country", ['Developing countries', "Africa", "Low income", "Lower middle income", "Upper middle income"])
     )
    df.to_csv(Paths.output / "chart_4.csv", index=False)

    # download data
    outflows_avg_data.loc[:, ['period', 'country', 'counterpart_type', 'prices', 'value']].to_csv(Paths.output / "chart_4_download.csv", index=False)




if __name__ == "__main__":
    chart_1()
    chart_2()
    chart_3()
    chart_4()

