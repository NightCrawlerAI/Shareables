import pandas as pd
import numpy as np
from dataflux import x2_int as x2
import pyodbc
import dataframe_image as dfi
from datetime import datetime, date
# from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import win32com.client as win32
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import subprocess
import time
from seaborn import set_style

set_style('dark')

def e360_holidays():
    import holidays
    import numpy as np
    from dateutil.easter import easter
    from pandas.tseries.offsets import BDay

    focus_years = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034]

    us_holidays = holidays.UnitedStates(years=focus_years,
                                        observed=True)

    for year in focus_years:
        us_holidays.update({pd.to_datetime(easter(year)-BDay(1)).date(): "Good Friday"})

    focus_holidays = [holiday for holiday in us_holidays.items()]

    e360_holidays = []
    for index, holiday in enumerate(focus_holidays):
        if 'Veterans' in holiday[1] or 'Columbus' in holiday[1]:
            continue
        else:
            e360_holidays.append(holiday[0])

    office_closed_days = np.busdaycalendar(holidays=e360_holidays)

    return office_closed_days


US_BUS_DAY = CustomBusinessDay(calendar=e360_holidays())
today = date.today() #- 1 * US_BUS_DAY
current_day = today.strftime('%m-%d-%Y')
next_bus_day = today + 1 * US_BUS_DAY
next_bus_day = next_bus_day.strftime('%m-%d-%Y')
class Strategy_VaR:
    @staticmethod
    def numfmt(x, pos):
        '''custom plot formatter function: divide by 1000'''
        s = f'{round(x / 1000000,2)}M'
        return s


    def __init__(self, strategy):
        self.strategy = strategy #This is the strategy name that will be used to pull the data from the database

        self.cnxn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            r'SERVER=e360-db01;'
            r'DATABASE=Voltage;'
            r'Trusted_Connection=yes;'
        )

        self.yfmt = tkr.FuncFormatter(self.numfmt)

    def __repr__(self):
        return f'This is a Strategy VaR object for {self.strategy}'

    def get_strategy_var_data(self):
        if self.strategy != 'PJM+Henry':
            strategy_var_q = f'''select as_of_date, historical_date, sum(hist_pnl) hist_pnl, 
            percentile_cont(.01)
            within group (order by sum(hist_pnl))
            over (partition by as_of_date) percentile_99,
            percentile_cont(.03)
            within group (order by sum(hist_pnl))
            over (partition by as_of_date) percentile_97,
            percentile_cont(.05)
            within group (order by sum(hist_pnl))
            over (partition by as_of_date) percentile_95
            from hvar_detail_vectors_percent 
            where as_of_date between '01-JAN-2023' and dateadd(day, 1, cast(getdate() as date))
            and var_ticker in (select distinct var_ticker from e360_master_mapping where strat_lvl1 = '{self.strategy}')
            and historical_date >= dateadd(day, -181, as_of_date)
            group by as_of_date, historical_date
            order by 1, 2'''

            strategy_var = pd.read_sql(strategy_var_q, self.cnxn)
        else:
            strategy_var_q = '''
            with prod_groups as (select var_ticker, strat_lvl1 as e360_product_group
                    from e360_master_mapping
                  group by var_ticker, strat_lvl1), 
            focus as (select as_of_date, historical_date, sum(hist_pnl) hist_pnl, map.e360_product_group 
            	from hvar_detail_vectors_percent hvar
            	inner join prod_groups map 
            	on hvar.var_ticker = map.var_ticker
            	where hvar.as_of_date >= '01-JAN-2023'
            	and historical_date >= dateadd(day, -181, hvar.as_of_date)
            	and hist_pnl is not null
            	and e360_product_group in ('PJM+M3 Strategy', 'Henry Hub')
            	group by as_of_date, historical_date, e360_product_group),
            final as (select as_of_date, historical_date, sum(hist_pnl) hist_pnl from focus 
            group by as_of_date, historical_date),
            thomas as (select as_of_date, historical_date, hist_pnl,
            	percentile_cont(.01)
            	within group (order by hist_pnl)
            	over (partition by as_of_date) percentile_99,
            	percentile_cont(.03)
            	within group (order by hist_pnl)
            	over (partition by as_of_date) percentile_97,
            	percentile_cont(.05)
            	within group (order by hist_pnl)
            	over (partition by as_of_date) percentile_95
            	from final)
            select distinct as_of_date, percentile_99, percentile_97, percentile_95 from thomas
            order by 1
            '''

            strategy_var = pd.read_sql(strategy_var_q, self.cnxn)

        return strategy_var

    def get_strategy_pnl_data(self):
        if self.strategy != 'PJM+Henry':
            strategy_pnl_q = f'''
            with base as (
            select pos.*, map.strat_lvl1 e360_product_group, map.strat_lvl2 e360_product_subgroup from position_snapshot_v3 pos
            left join e360_master_mapping map
                on pos.primary_product_code=map.molecule_product_code
             where as_of_date between '01-JAN-2023' and dateadd(day, -1, cast(getdate() as date))
             ) 
             select * from base where e360_product_group = '{self.strategy}'
            '''

            strategy_pnl = pd.read_sql(strategy_pnl_q, self.cnxn)

        else:
            strategy_pnl_q = '''
            with base as (
                    select pos.*, map.strat_lvl1 e360_product_group, map.strat_lvl2 e360_product_subgroup from position_snapshot_v3 pos
                	left join e360_master_mapping map
                		on pos.primary_product_code=map.molecule_product_code
                     where as_of_date between '01-JAN-2023' and dateadd(day, -1, cast(getdate() as date))
                     ) 
                     select * from base where e360_product_group in ('PJM+M3 Strategy', 'Henry Hub')
            '''

            strategy_pnl = pd.read_sql(strategy_pnl_q, self.cnxn)
        return strategy_pnl

    def create_strategy_groupbys(self, var_df, pnl_df):
        var_grouped = var_df.groupby('as_of_date')[['percentile_99', 'percentile_95']].mean()
        var_grouped['percentile_99 pos'] = var_grouped.percentile_99.abs()
        var_grouped['percentile_95 pos'] = var_grouped.percentile_95.abs()

        pnl_grouped = pnl_df.groupby('as_of_date')[['base_currency_mtm_change']].sum()

        var_plot_series = var_grouped[var_grouped.index.isin(pnl_grouped.index)]
        pnl_plot_series = pnl_grouped[pnl_grouped.index.isin(var_grouped.index)]

        return var_plot_series, pnl_plot_series

    def create_plot_slices(self, var_df, pnl_df):
        var_5_day = var_df * np.sqrt(5)

        pnl_5_day = pnl_df.rolling(5).sum().rename(columns={'base_currency_mtm_change': 'rolling_pnl'})
        cum_pnl = pnl_df.cumsum().rename(columns={'base_currency_mtm_change': 'total_pnl'})

        return var_5_day, pnl_5_day, cum_pnl


    def plot_5_day_rolling(self, var_df, pnl_df):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        data = pd.concat([var_df, pnl_df], axis=1)[
            ['percentile_99', 'percentile_95', 'rolling_pnl', 'percentile_99 pos', 'percentile_95 pos']]
        data.plot(kind='line', ax=ax1, stacked=False, color=['red', 'blue', 'black', 'red', 'blue'])
        ax1.legend(data.iloc[:, :3], loc='upper center', frameon=False, ncol=3)
        ax1.set_ylabel('5 Day Rolling HVaR', labelpad=15, fontsize=12)
        ax1.yaxis.set_major_formatter(self.yfmt)

        # ax1.set_xlabel('Date', fontsize=12)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

        plt.title(f'{self.strategy.upper()} | HVaR vs. PnL (Rolling 5 Days)')
        plt.savefig(fr"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_{strategy}.png")
        # plt.show()

    def create_cumulative_plot(self, var_df, pnl_df):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        var_df.iloc[:, 2:].plot(kind='line', ax=ax1, stacked=False, color=['green', 'purple'])
        ax1.legend(loc='lower left')
        ax1.set_ylabel('Daily HVaR', labelpad=15, fontsize=12)
        ax1.yaxis.set_major_formatter(self.yfmt)
        # ax1.set_xlabel('Date', fontsize=12)

        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.fill_between(var_df.index, var_df['percentile_99 pos'], var_df['percentile_95 pos'])

        ax2 = ax1.twinx()

        pnl_df.plot(kind='line', ax=ax2, linestyle='dashdot', label='cumulative_pnl', color='red')
        ax2.set_ylabel('Cumulative PnL', rotation=270, labelpad=15, fontsize=12)
        ax2.yaxis.set_major_formatter(self.yfmt)

        plt.title(f'{self.strategy.upper()} | HVaR vs. PnL')
        plt.savefig(fr"P:\KJ\Adhocs\thomas_hvar_vs_pnl_{strategy}.png")
        # plt.show()

    def run(self):
        strategy_var = self.get_strategy_var_data()
        strategy_pnl = self.get_strategy_pnl_data()
        strategy_var_series, strategy_pnl_series = self.create_strategy_groupbys(strategy_var, strategy_pnl)
        var_5_day, pnl_5_day, cum_pnl = self.create_plot_slices(strategy_var_series, strategy_pnl_series)
        self.plot_5_day_rolling(var_5_day, pnl_5_day)
        self.create_cumulative_plot(strategy_var_series, cum_pnl)


if __name__ == '__main__':
    strategies = ['Basis', 'ERCOT', 'Environmental', 'Global NG',
                  'Henry Hub', 'Nepool', 'PJM+M3 Strategy', 'West Strategy', 'PJM+Henry']

    for strategy in strategies:
        var = Strategy_VaR(strategy)
        print(var)
        var.run()

