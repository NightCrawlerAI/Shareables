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
bat_date = today.strftime('%Y-%m-%d')
mtd_start = today.replace(day=1).strftime('%Y-%m-%d')
ytd_start = pd.to_datetime('01/01/2024').date().strftime('%Y-%m-%d')


cnxn = pyodbc.connect(
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=e360-db01;'
    r'DATABASE=Voltage;'
    r'Trusted_Connection=yes;'
)


ytd_pnl = x2.PositionSnapshotV3(ytd_start, current_day)
mtd_pnl = x2.PositionSnapshotV3(mtd_start, current_day)
curr_pnl = x2.PositionSnapshotV3(current_day, current_day)
novtd_pnl = x2.PositionSnapshotV3('11-01-2023', current_day)
# print(curr_pnl)


def format_df(df):
    search_cols = ['as_of_date', 'contract_month', 'expiry_date', 'tenor',
                   'trade_date', 'option_strip', 'expiration_date', 'Date',
                   'trading_date', 'start_date', 'end_date', 'report_date']
    for col in df.columns:
        if col in search_cols:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %X %p').dt.date
    return df


psg_map = pd.read_sql("select molecule_product_code, strat_lvl2, strat_lvl1 from e360_master_mapping", cnxn)
groups = dict(zip(psg_map.molecule_product_code, psg_map.strat_lvl1))
sub_groups = dict(zip(psg_map.molecule_product_code, psg_map.strat_lvl2))


historical_var_series_q = f'''
select as_of_date, historical_date, sum(hist_pnl) hist_pnl, 
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
where as_of_date between '01-JAN-2023' and '{next_bus_day}'
and historical_date >= dateadd(day, -181, as_of_date)
group by as_of_date, historical_date
order by as_of_date
'''

hist_var_series = pd.read_sql(historical_var_series_q, cnxn)

series_var = hist_var_series.groupby('as_of_date')[['percentile_99', 'percentile_95']].mean()
series_var['percentile_99 pos'] = series_var.percentile_99.abs()
# series_var['percentile_97 pos'] = series_var.percentile_97.abs()
series_var['percentile_95 pos'] = series_var.percentile_95.abs()

yearly_pnl = format_df(x2.PositionSnapshotV3('01/01/2023', current_day))
yearly_pnl['e360_product_group'] = yearly_pnl.primary_product_code.map(groups)
yearly_pnl['e360_product_subgroup'] = yearly_pnl.primary_product_code.map(sub_groups)

series_pnl = yearly_pnl.groupby('as_of_date')[['base_currency_mtm_change']].sum()

var_plot_series = series_var[~np.logical_not(series_var.index.isin(series_pnl.index))]
pnl_plot_series = series_pnl[~np.logical_not(series_pnl.index.isin(series_var.index))].rename(columns = {'base_currency_mtm_change' : 'daily_pnl'})

thomas_5_day_var = var_plot_series * np.sqrt(5)
thomas_5_day_pnl = pnl_plot_series.rolling(5).sum().rename(columns = {'daily_pnl' : 'rolling_pnl'})
thomas_cum_pnl = pnl_plot_series.cumsum().rename(columns = {'daily_pnl' : 'total_pnl'})

def numfmt(x, pos):
    '''custom plot formatter function: divide by 1,000,000'''
    s = f'{round(x / 1000000,2)}M'
    return s

yfmt = tkr.FuncFormatter(numfmt)


def plot_5_day_rolling(var_df, pnl_df, strategy):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    data = pd.concat([var_df, pnl_df], axis=1)[['percentile_99', 'percentile_95', 'rolling_pnl', 'percentile_99 pos', 'percentile_95 pos']]
    data.plot(kind='line', ax=ax1, stacked=False, color=['red', 'blue', 'black', 'red', 'blue'])
    ax1.legend(data.iloc[:, :3], loc='upper center', frameon=False, ncol=3)
    ax1.set_ylabel('5 Day Rolling HVaR', labelpad=15, fontsize=12)
    ax1.yaxis.set_major_formatter(yfmt)

    # ax1.set_xlabel('Date', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # ax2 = ax1.twinx()

    # thomas_5_day_pnl.plot(kind='line', ax=ax2, linestyle='dashdot', color = 'r')
    # ax2.set_ylabel('5 Day Rolling PnL', rotation =270, labelpad=20)
    # ax2.yaxis.set_major_formatter(yfmt)

    plt.title(f'{strategy.upper()} | HVaR vs. PnL (Rolling 5 Days)')
    plt.savefig(fr"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_{strategy}.png")
    # plt.show()

plot_5_day_rolling(thomas_5_day_var, thomas_5_day_pnl, 'Portfolio Total')


def create_cumulative_plot(var_df, pnl_df, strategy):
    fig, ax1 = plt.subplots(figsize=(10,6))

    var_df.iloc[:,2:].plot(kind='line', ax=ax1, stacked = False, color = ['green', 'purple'] )
    ax1.legend(loc = 'lower left')
    ax1.set_ylabel('Daily HVaR', labelpad = 15,fontsize = 12)
    ax1.yaxis.set_major_formatter(yfmt)
    # ax1.set_xlabel('Date', fontsize = 12)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation = 45)
    ax1.fill_between(var_df.index, var_df['percentile_99 pos'], var_df['percentile_95 pos'])

    ax2 = ax1.twinx()

    pnl_df.plot(kind='line', ax=ax2, linestyle='dashdot', label = 'cumulative_pnl', color = 'red')
    ax2.set_ylabel('Cumulative PnL', rotation =270, labelpad=15, fontsize = 12)
    ax2.yaxis.set_major_formatter(yfmt)

    plt.title(f'{strategy.upper()} | HVaR vs. PnL')
    plt.savefig(fr"P:\KJ\Adhocs\thomas_hvar_vs_pnl_{strategy}.png")
    # plt.show()

create_cumulative_plot(var_plot_series, thomas_cum_pnl, 'Portfolio Total')


subprocess.run(['python', 'Strategy_VaR.py'])


global_hvar_q = f"""with prod_groups as (select var_ticker, strat_lvl2 as e360_product_subgroup
        from e360_master_mapping
      group by var_ticker, strat_lvl2) 
select as_of_date, historical_date, sum(hist_pnl) hist_pnl, map.e360_product_subgroup 
from hvar_detail_vectors_percent hvar
inner join prod_groups map 
on hvar.var_ticker = map.var_ticker
where hvar.as_of_date in ('{current_day}','{next_bus_day}')
and historical_date >= dateadd(day, -181, hvar.as_of_date)
and hist_pnl is not null
group by as_of_date, historical_date, e360_product_subgroup
order by 2"""

global_total_hvar_q = f"""with prod_groups as (select var_ticker
        from e360_master_mapping
      group by var_ticker) 
select as_of_date, historical_date, sum(hist_pnl) hist_pnl
from hvar_detail_vectors_percent hvar
inner join prod_groups map 
on hvar.var_ticker = map.var_ticker
where hvar.as_of_date in ('{current_day}','{next_bus_day}')
and historical_date >= dateadd(day, -181, hvar.as_of_date)
and hist_pnl is not null
group by as_of_date, historical_date
order by 2"""

global_hvar = pd.read_sql(global_hvar_q,cnxn)
global_total_hvar = pd.read_sql(global_total_hvar_q,cnxn)

hvar = global_hvar.loc[global_hvar.as_of_date == pd.to_datetime(next_bus_day).date()]
prior_hvar = global_hvar.loc[global_hvar.as_of_date == pd.to_datetime(current_day).date()]

total_hvar = global_total_hvar.loc[global_total_hvar.as_of_date == pd.to_datetime(next_bus_day).date()]
prior_total_hvar = global_total_hvar.loc[global_total_hvar.as_of_date == pd.to_datetime(current_day).date()]


def create_var_by_grouping(var_df, total_df, group_col, index_name='Product Subgroup'):
    '''
    var_df : dataframe that contains the var data by desired grouping
    total_df: dataframe that contains the ungrouped var data hist_pnl vectors
    group_col : string of the column label that is the unique key for grouping
    index_name : string of the desired index column's name (Product Group)
    '''
    var_dict_99 = {}
    var_dict_97 = {}
    var_dict_95 = {}

    prod_groups = list(var_df[group_col].unique())

    for prod_group in prod_groups:
        sub_var_df = var_df[var_df[group_col] == prod_group]

        var_99 = abs(sub_var_df.hist_pnl.quantile(0.01))
        var_dict_99.update({prod_group: var_99})

        var_97 = abs(sub_var_df.hist_pnl.quantile(0.03))
        var_dict_97.update({prod_group: var_97})

        var_95 = abs(sub_var_df.hist_pnl.quantile(0.05))
        var_dict_95.update({prod_group: var_95})

    var_dict_99.update({'Total': abs(total_df.hist_pnl.quantile(.01))})
    var_dict_97.update({'Total': abs(total_df.hist_pnl.quantile(.03))})
    var_dict_95.update({'Total': abs(total_df.hist_pnl.quantile(.05))})

    if index_name == 'Product Subgroup':
        test0 = pd.DataFrame(index=var_dict_99.keys(), data=var_dict_99.values(), columns=['Subgrp 99 HVaR'])
        test05 = pd.DataFrame(index=var_dict_97.keys(), data=var_dict_97.values(), columns=['Subgrp 97 HVaR'])
        test1 = pd.DataFrame(index=var_dict_95.keys(), data=var_dict_95.values(), columns=['Subgrp 95 HVaR'])
        test = pd.concat([test0, test05, test1], axis=1)
        test.index.name = index_name
        return test
    else:
        gtest0 = pd.DataFrame(index=var_dict_99.keys(), data=var_dict_99.values(), columns=['Grp 99 HVaR'])
        gtest05 = pd.DataFrame(index=var_dict_97.keys(), data=var_dict_97.values(), columns=['Grp 97 HVaR'])
        gtest1 = pd.DataFrame(index=var_dict_95.keys(), data=var_dict_95.values(), columns=['Grp 95 HVaR'])
        gtest = pd.concat([gtest0, gtest05, gtest1], axis=1)
        gtest.index.name = index_name
        return gtest



#
# # Compute HVaR by Product SubGroup
# final_dict_99 = {}
# prod_groups = list(hvar.e360_product_subgroup.unique())
# for prod_group in prod_groups:
#     var_df = hvar[hvar.e360_product_subgroup == prod_group]
#     var_99 = abs(var_df.hist_pnl.quantile(0.01))
#     final_dict_99.update({prod_group: var_99})
#
# final_dict_99.update({'Total': abs(total_hvar.hist_pnl.quantile(0.01))})
#
# final_dict_97 = {}
# prod_groups = list(hvar.e360_product_subgroup.unique())
# for prod_group in prod_groups:
#     var_df = hvar[hvar.e360_product_subgroup == prod_group]
#     var_97 = abs(var_df.hist_pnl.quantile(0.03))
#     final_dict_97.update({prod_group: var_97})
#
# final_dict_97.update({'Total': abs(total_hvar.hist_pnl.quantile(0.03))})
#
# final_dict_95 = {}
# prod_groups = list(hvar.e360_product_subgroup.unique())
# for prod_group in prod_groups:
#     var_df = hvar[hvar.e360_product_subgroup == prod_group]
#     var_95 = abs(var_df.hist_pnl.quantile(0.05))
#     final_dict_95.update({prod_group: var_95})
#
# final_dict_95.update({'Total': abs(total_hvar.hist_pnl.quantile(0.05))})

# Compute HVaR by Product Group
global_g_hvar_q = f"""with prod_groups as (select var_ticker, strat_lvl1 as e360_product_group
        from e360_master_mapping
      group by var_ticker, strat_lvl1) 
select as_of_date, historical_date, sum(hist_pnl) hist_pnl, map.e360_product_group 
from hvar_detail_vectors_percent hvar
inner join prod_groups map 
on hvar.var_ticker = map.var_ticker
where hvar.as_of_date in ('{current_day}','{next_bus_day}')
and historical_date >= dateadd(day, -181, hvar.as_of_date)
and hist_pnl is not null
group by as_of_date, historical_date, e360_product_group
order by 2"""

global_g_hvar = pd.read_sql(global_g_hvar_q,cnxn)

g_hvar = global_g_hvar.loc[global_g_hvar.as_of_date == pd.to_datetime(next_bus_day).date()]
prior_g_hvar = global_g_hvar.loc[global_g_hvar.as_of_date == pd.to_datetime(current_day).date()]

hvar_today = create_var_by_grouping(hvar,total_hvar,'e360_product_subgroup')
g_hvar_today = create_var_by_grouping(g_hvar, total_hvar, 'e360_product_group', 'Product Group')
hvar_yesterday = create_var_by_grouping(prior_hvar, prior_total_hvar, 'e360_product_subgroup')
g_hvar_yesterday = create_var_by_grouping(prior_g_hvar, prior_total_hvar, 'e360_product_group', 'Product Group')
#
# g_final_dict_99 = {}
# prod_groups = list(g_hvar.e360_product_group.unique())
# for prod_group in prod_groups:
#     var_df = g_hvar[g_hvar.e360_product_group == prod_group]
#     var_99 = abs(var_df.hist_pnl.quantile(0.01))
#     g_final_dict_99.update({prod_group: var_99})
#
# g_final_dict_99.update({'Total': abs(total_hvar.hist_pnl.quantile(0.01))})
#
# g_final_dict_97 = {}
# prod_groups = list(g_hvar.e360_product_group.unique())
# for prod_group in prod_groups:
#     var_df = g_hvar[g_hvar.e360_product_group == prod_group]
#     var_97 = abs(var_df.hist_pnl.quantile(0.03))
#     g_final_dict_97.update({prod_group: var_97})
#
# g_final_dict_97.update({'Total': abs(total_hvar.hist_pnl.quantile(0.03))})
#
# g_final_dict_95 = {}
# prod_groups = list(g_hvar.e360_product_group.unique())
# for prod_group in prod_groups:
#     var_df = g_hvar[g_hvar.e360_product_group == prod_group]
#     var_95 = abs(var_df.hist_pnl.quantile(0.05))
#     g_final_dict_95.update({prod_group: var_95})
#
# g_final_dict_95.update({'Total': abs(total_hvar.hist_pnl.quantile(0.05))})

thomas_new = pd.concat([g_hvar_yesterday, g_hvar_today], axis = 1)

thomas_new['Grp 99 HVaR DoD % Δ']=thomas_new.iloc[:,[0,3]].pct_change(axis=1).iloc[:,1]*100
thomas_new['Grp 97 HVaR DoD % Δ']=thomas_new.iloc[:,[1,4]].pct_change(axis=1).iloc[:,1]*100
thomas_new['Grp 95 HVaR DoD % Δ']=thomas_new.iloc[:,[2,5]].pct_change(axis=1).iloc[:,1]*100

thomas_new['Grp 99 HVaR DoD Net Δ']=thomas_new.iloc[:,[0,3]].diff(axis=1).iloc[:,1]
thomas_new['Grp 97 HVaR DoD Net Δ']=thomas_new.iloc[:,[1,4]].diff(axis=1).iloc[:,1]
thomas_new['Grp 95 HVaR DoD Net Δ']=thomas_new.iloc[:,[2,5]].diff(axis=1).iloc[:,1]

row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental', 'Global NG', 'Henry Hub', 'Nepool', 'PJM+M3 Strategy', 'West Strategy', 'Total']
thomas_new = thomas_new.reindex(row_order)

thomas_sendout = thomas_new.iloc[:,6:9]
thomas_final = thomas_sendout.fillna(0).style.format('{:.2f}%', na_rep='')

rows_to_style = thomas_new.index != 'Total'
thomas_final2 = thomas_new.iloc[:,9:].fillna(0).style.background_gradient(subset=pd.IndexSlice[rows_to_style, :],\
                                                                         cmap='RdYlBu').format('{:,.2f}', na_rep = '')


thomas_sub_new = pd.concat([hvar_yesterday, hvar_today], axis = 1)

thomas_sub_new['Subgrp 99 HVaR DoD % Δ']=thomas_sub_new.iloc[:,[0,3]].pct_change(axis=1).iloc[:,1]*100
thomas_sub_new['Subgrp 97 HVaR DoD % Δ']=thomas_sub_new.iloc[:,[1,4]].pct_change(axis=1).iloc[:,1]*100
thomas_sub_new['Subgrp 95 HVaR DoD % Δ']=thomas_sub_new.iloc[:,[2,5]].pct_change(axis=1).iloc[:,1]*100

thomas_sub_new['Subgrp 99 HVaR DoD Net Δ']=thomas_sub_new.iloc[:,[0,3]].diff(axis=1).iloc[:,1]
thomas_sub_new['Subgrp 97 HVaR DoD Net Δ']=thomas_sub_new.iloc[:,[1,4]].diff(axis=1).iloc[:,1]
thomas_sub_new['Subgrp 95 HVaR DoD Net Δ']=thomas_sub_new.iloc[:,[2,5]].diff(axis=1).iloc[:,1]

new_row_order = ['Midwest', 'Mountain', 'South Central', 'WTI', 'ERC-N On', 'ERC-N Off', 'CCA', 'WCA', 'REC', 'RGGI',
'Dutch TTF', 'Henry Hub', 'Mass Hub On', 'PJM On', 'PJM Off',
'East Basis', 'West Power On', 'West Power Off', 'Pacific Basis',
 'Total']
thomas_sub_new = thomas_sub_new.reindex(new_row_order)

sub_rows_to_style = thomas_sub_new.index != 'Total'
thomas_sub_sendout = thomas_sub_new.iloc[:,6:9]
thomas_sub_final = thomas_sub_sendout.fillna(0).style.format('{:.2f}%', na_rep='')

thomas_sub_final2 = thomas_sub_new.iloc[:,9:].fillna(0).style.background_gradient(subset=pd.IndexSlice[sub_rows_to_style, :],\
                                                                         cmap='RdYlBu').format('{:,.2f}', na_rep = '')


dfi.export(thomas_final, r"P:\KJ\Adhocs\thomas_dod_group_var.png")
dfi.export(thomas_final2, r"P:\KJ\Adhocs\thomas_dod_group_var_net_chng.png")
dfi.export(thomas_sub_final, r"P:\KJ\Adhocs\thomas_dod_subgroup_var.png")
dfi.export(thomas_sub_final2, r"P:\KJ\Adhocs\thomas_dod_subgroup_var_net_chng.png")

#Compute YTD, MTD, EOD PnLs
target_grps = ['Crude', 'Environmental', 'ERCOT', 'Global NG', 'Henry Hub', 'PJM+M3 Strategy', 'West Strategy', 'Nepool', 'Basis']
count = 1
for pnl in [ytd_pnl, mtd_pnl, curr_pnl, novtd_pnl]:
    # print(count)
    # print(pnl)
    pnl['e360_product_group'] = pnl.primary_product_code.map(groups)
    pnl['e360_product_subgroup'] = pnl.primary_product_code.map(sub_groups)
    pnl.rename(columns={
        'e360_product_group': 'Product Group',
        'e360_product_subgroup': 'Product SubGroup',
        'base_currency_mtm_change': 'EOD PnL'
    }, inplace=True)

    if count == 1:
        thomas_ytd = pnl.loc[pnl['Product Group'].isin(target_grps), :].groupby(['Product Group', 'Product SubGroup'])[
            'EOD PnL'].sum().to_frame()
    elif count == 2:#pnl.equals(mtd_pnl):
        thomas_mtd = pnl.loc[pnl['Product Group'].isin(target_grps), :].groupby(['Product Group', 'Product SubGroup'])[
            'EOD PnL'].sum().to_frame()
    elif count == 3:
        thomas_curr = pnl.loc[pnl['Product Group'].isin(target_grps), :].groupby(['Product Group', 'Product SubGroup'])[
            'EOD PnL'].sum().to_frame()
    else:
        thomas_novtd = pnl.loc[pnl['Product Group'].isin(target_grps), :].groupby(['Product Group', 'Product SubGroup'])[
            'EOD PnL'].sum().to_frame()

    count += 1

# print(thomas_curr)
# import sys
# sys.exit(1)

def create_base_df(thomas):
    # Recalculate the sum of EOD PnL for updated groups
    grouped_sum = thomas.groupby(level='Product Group')['EOD PnL'].sum().reset_index(name='Sum of Product Group')
    thomas_reset = thomas.reset_index()
    thomas_merged = thomas_reset.merge(grouped_sum, on='Product Group')
    thomas_merged.set_index(['Product Group', 'Product SubGroup'], inplace=True)
    thomas_merged.loc[thomas_merged['Sum of Product Group'].duplicated(), 'Sum of Product Group'] = np.nan

    # Calculate and Add Total PnL at the Bottom
    total_pnl = pd.DataFrame({
        'EOD PnL': thomas['EOD PnL'].sum(),
        'Sum of Product Group': thomas['EOD PnL'].sum()
    }, index=pd.MultiIndex.from_tuples([('Total', 'Total')]))

    thomas_merged = pd.concat([thomas_merged, total_pnl])
    thomas_merged.index.names = ['Product Group', 'Product Subgroup']

    return thomas_merged


final_ytd = create_base_df(thomas_ytd).rename(columns = {
    'EOD PnL':'YTD Subgroup PnL',
    'Sum of Product Group':'YTD Group PnL'
})
final_mtd = create_base_df(thomas_mtd).rename(columns = {
    'EOD PnL':'MTD Subgroup PnL',
    'Sum of Product Group':'MTD Group PnL'
})
final_curr = create_base_df(thomas_curr).rename(columns = {
    'EOD PnL':'EOD Subgroup PnL',
    'Sum of Product Group':'EOD Group PnL'
})
final_novtd = create_base_df(thomas_novtd).rename(columns = {
    'EOD PnL':'NovTD Subgroup PnL',
    'Sum of Product Group':'NovTD Group PnL'
})


col_order_legacy = ['EOD Subgroup PnL', 'EOD Group PnL', 'MTD Subgroup PnL', 'MTD Group PnL',
                    'YTD Subgroup PnL', 'YTD Group PnL', 'NovTD Subgroup PnL', 'NovTD Group PnL']
final_df = final_ytd.join([final_mtd, final_curr, final_novtd])[col_order_legacy]


# gtest0 = pd.DataFrame(index=g_final_dict_99.keys(), data = g_final_dict_99.values(), columns = ['Grp 99 HVaR'])
# gtest05 = pd.DataFrame(index=g_final_dict_97.keys(), data = g_final_dict_97.values(), columns = ['Grp 97 HVaR'])
# gtest1 = pd.DataFrame(index=g_final_dict_95.keys(), data=g_final_dict_95.values(), columns = ['Grp 95 HVaR'])
# gtest = pd.concat([gtest0, gtest05, gtest1], axis = 1)
# gtest.index.name = 'Product Group'
#
# test0 = pd.DataFrame(index=final_dict_99.keys(), data = final_dict_99.values(), columns = ['Subgrp 99 HVaR'])
# test05 = pd.DataFrame(index=final_dict_97.keys(), data = final_dict_97.values(), columns = ['Subgrp 97 HVaR'])
# test1 = pd.DataFrame(index=final_dict_95.keys(), data=final_dict_95.values(), columns = ['Subgrp 95 HVaR'])
# test = pd.concat([test0, test05, test1], axis = 1)
# test.index.name = 'Product Subgroup'

multi_index_map = pd.read_sql("select distinct strat_lvl2, strat_lvl1 from e360_master_mapping order by 2", cnxn)
sub_to_group_map = dict(zip(multi_index_map.strat_lvl2, multi_index_map.strat_lvl1))
sub_to_group_map.update({'Total' : 'Total'})


gtest_reset = g_hvar_today.reset_index()
test_reset = hvar_today.reset_index()

test_reset['Product Group'] = test_reset['Product Subgroup'].map(sub_to_group_map)

# Merge the DataFrames
combined_df = pd.merge(gtest_reset, test_reset, on='Product Group', how='outer')

# Set the index to be hierarchical post-merge
combined_df.set_index(['Product Group', 'Product Subgroup'], inplace=True)

# Sort the index to ensure it's ordered after the operations
combined_df.sort_index(inplace=True)

duplicates = combined_df[['Grp 99 HVaR', 'Grp 97 HVaR', 'Grp 95 HVaR']].duplicated()
combined_df[['Grp 99 HVaR', 'Grp 97 HVaR', 'Grp 95 HVaR']] = combined_df[['Grp 99 HVaR', 'Grp 97 HVaR', 'Grp 95 HVaR']].mask(duplicates, np.NaN)

combined_df['Grp 99/95 HVaR (%)'] = (combined_df['Grp 99 HVaR']/combined_df['Grp 95 HVaR'])*100
combined_df['Subgrp 99/95 HVaR (%)'] = (combined_df['Subgrp 99 HVaR']/combined_df['Subgrp 95 HVaR'])*100

testing2 = final_df.join(combined_df)
final_df2 = testing2.copy()

new_cols = ['EOD Subgroup PnL', 'MTD Subgroup PnL', 'YTD Subgroup PnL', 'NovTD Subgroup PnL',
            'Subgrp 99 HVaR', 'Subgrp 97 HVaR', 'Subgrp 95 HVaR', 'Subgrp 99/95 HVaR (%)',
            'EOD Group PnL', 'MTD Group PnL', 'YTD Group PnL', 'NovTD Group PnL',
            'Grp 99 HVaR', 'Grp 97 HVaR', 'Grp 95 HVaR', 'Grp 99/95 HVaR (%)'
            ]

# new_row_order = [('Basis', 'Midwest'), ('Basis', 'Mountain'), ('Basis', 'South Central'), ('Crude', 'WTI'), ('ERCOT', 'ERC-N On'),
#  ('ERCOT', 'ERC-N Off'), ('Carbon Allowance', 'CCA'), ('Carbon Allowance', 'WCA'), ('Environmental', 'REC'),
#  ('Global NG', 'Dutch TTF'), ('Henry Hub', 'Henry Hub'), ('Nepool', 'Mass Hub On'), ('PJM+M3 Strategy', 'PJM On'), ('PJM+M3 Strategy', 'PJM Off'),
#  ('PJM+M3 Strategy', 'East Basis'), ('West Strategy', 'West Power On'), ('West Strategy', 'West Power Off'), ('West Strategy', 'Pacific Basis'),
#  ('Total', 'Total')]

final_df2 = final_df2[new_cols]

def style_sub_columns(s):
    # s is a Series representing a row, with the column names as the index
    styles = []
    for col_name in s.index:
        if 'Group' in col_name:
            styles.append('font-weight: bold; font-size: 12px')  # italic text for 'Group PnL' columns
        elif 'Subgroup' in col_name:
            val = s[col_name]
            color = 'darkred' if val < 0 else 'darkgreen'  # dark red for negative, dark green for positive
            styles.append(f'background-color: {color}; color: white')  # font color is white
        elif 'HVaR' in col_name:
            styles.append('font-weight: bold; color: darkblue; font-size: 12px')
        else:
            styles.append('')  # default style for other columns
    return styles

# Apply the styles to the DataFrame
thomas_styled = final_df2.style\
    .apply(style_sub_columns, axis=1)\
    .format('{:,.0f}', na_rep='')\
    .set_table_styles([
        {'selector': 'th.level0', 'props': [('vertical-align', 'top')]}
    ])

dfi.export(thomas_styled, r"P:\KJ\Adhocs\thomas_eod_pnl.png")



final_df3 = final_df2.dropna(subset='YTD Group PnL').reset_index(drop = True, level = 1).iloc[:,8:]
# print(final_df3)
# import sys
# sys.exit(1)
def style_grp_columns(s):
    # s is a Series representing a row, with the column names as the index
    styles = []
    for col_name in s.index:
        if 'PnL' in col_name:
            val = s[col_name]
            color = 'darkred' if val < 0 else 'darkgreen'  # dark red for negative, dark green for positive
            styles.append(f'background-color: {color}; color: white; font-style: italic; font-size: 12px')  # font color is white
        elif 'HVaR' in col_name:
            styles.append('font-weight: bold; color: darkblue; font-size: 12px')
        else:
            styles.append('')  # default style for other columns
    return styles

thomas_grp_styled = final_df3.style\
    .apply(style_grp_columns, axis=1)\
    .format('{:,.0f}', na_rep='')\
    .set_table_styles([
        {'selector': 'th.level0', 'props': [('vertical-align', 'top')]}
    ])

dfi.export(thomas_grp_styled, r"P:\KJ\Adhocs\thomas_eod_pnl_group.png")



def create_corr_matrix(hvar_df):
    base_case = hvar_df.fillna(0).iloc[:, 1:]

    pivoted_df = base_case.groupby(list(base_case.columns[[0, 2]]))['hist_pnl'].sum().unstack().fillna(0)

    correlation_matrix = pivoted_df.corr().fillna(0)
    correlation_matrix.index.name = None

    styled_corr_matrix = correlation_matrix.style.background_gradient(cmap='coolwarm').format('{:.2f}')

    return styled_corr_matrix

subgroup_corr = create_corr_matrix(hvar)
dfi.export(subgroup_corr, r"P:\KJ\Adhocs\thomas_eod_pnl_subgroup_corr.png")

group_corr = create_corr_matrix(g_hvar)
dfi.export(group_corr, r"P:\KJ\Adhocs\thomas_eod_pnl_group_corr.png")



#For PJM+Henry Only
def compute_var(series, percentile):
    '''percentile must be a float between 0 and 1'''
    return abs(series.quantile(percentile))

def compute_sum(df, inclusive=None):
    if not inclusive:
        return df.loc[df['Product Group'].isin(['PJM+M3 Strategy', 'Henry Hub']), 'EOD PnL'].sum()
    else:
        return df.loc[np.logical_not(df['Product Group'].isin(['PJM+M3 Strategy', 'Henry Hub'])), 'EOD PnL'].sum()



focus_hvar_q = f"""with prod_groups as (select var_ticker, strat_lvl1 as e360_product_group
        from e360_master_mapping
      group by var_ticker, strat_lvl1), 
focus as (select as_of_date, historical_date, sum(hist_pnl) hist_pnl, map.e360_product_group 
from hvar_detail_vectors_percent hvar
inner join prod_groups map 
on hvar.var_ticker = map.var_ticker
where hvar.as_of_date = '{next_bus_day}'
and historical_date >= dateadd(day, -181, hvar.as_of_date)
and hist_pnl is not null
and e360_product_group in ('PJM+M3 Strategy', 'Henry Hub')
group by as_of_date, historical_date, e360_product_group)
select as_of_date, historical_date, sum(hist_pnl) hist_pnl from focus 
group by as_of_date, historical_date
order by 2"""


rest_hvar_q = f"""
with prod_groups as (select var_ticker, strat_lvl1 as e360_product_group
        from e360_master_mapping
      group by var_ticker, strat_lvl1), 
focus as (select as_of_date, historical_date, sum(hist_pnl) hist_pnl, map.e360_product_group 
from hvar_detail_vectors_percent hvar
inner join prod_groups map 
on hvar.var_ticker = map.var_ticker
where hvar.as_of_date = '{next_bus_day}'
and historical_date >= dateadd(day, -181, hvar.as_of_date)
and hist_pnl is not null
and e360_product_group not in ('PJM+M3 Strategy', 'Henry Hub')
group by as_of_date, historical_date, e360_product_group)
select as_of_date, historical_date, sum(hist_pnl) hist_pnl from focus 
group by as_of_date, historical_date
order by 2
"""

focus_hvar = pd.read_sql(focus_hvar_q,cnxn)
rest_hvar = pd.read_sql(rest_hvar_q,cnxn)


pjm_henry_line = focus_hvar#.loc[focus_hvar.e360_product_group.isin(['PJM', 'Henry Hub'])]
remain_line = rest_hvar#.loc[np.logical_not(rest_hvar.e360_product_group.isin(['PJM', 'Henry Hub']))]

ph_var_99 = compute_var(pjm_henry_line.hist_pnl, .01)
ph_var_97 = compute_var(pjm_henry_line.hist_pnl, .03)
ph_var_95 = compute_var(pjm_henry_line.hist_pnl, .05)
ph_var_99_95 = ph_var_99/ph_var_95 * 100

ph_ytd_pnl = compute_sum(ytd_pnl)
ph_mtd_pnl = compute_sum(mtd_pnl)
ph_eod_pnl = compute_sum(curr_pnl)

remain_99 = compute_var(remain_line.hist_pnl, .01)
remain_97 = compute_var(remain_line.hist_pnl, .03)
remain_95 = compute_var(remain_line.hist_pnl, .05)
remain_99_95 = remain_99/remain_95 * 100

remain_ytd = compute_sum(ytd_pnl, True)
remain_mtd = compute_sum(mtd_pnl, True)
remain_eod = compute_sum(curr_pnl, True)

ph_df = pd.DataFrame(index = ['PJM+Henry', 'Rest of Portfolio'],
                     data = np.array([[ph_eod_pnl, ph_mtd_pnl, ph_ytd_pnl,ph_var_99, ph_var_97, ph_var_95, ph_var_99_95],
                                      [remain_eod, remain_mtd, remain_ytd, remain_99, remain_97, remain_95, remain_99_95]]).reshape(2,7),
                     columns = ['EOD PnL', 'MTD PnL', 'YTD PnL', '99 HVaR', '97 HVaR', '95 HVaR', '99/95 HVaR (%)'])

ph_df_styled = ph_df.style\
    .apply(style_grp_columns, axis=1)\
    .format('{:,.0f}', na_rep='')\
    .set_table_styles([
        {'selector': 'th.level0', 'props': [('vertical-align', 'top')]}
    ])

dfi.export(ph_df_styled, r"P:\KJ\Adhocs\thomas_eod_pnl_group_pjm_henry_df.png")


def send_email(image_paths):
    """Sends an email with the attached images inline"""
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = f'Experimental VaR Report: EOD PnL Agg : {current_day}'
    mail.To = 'tnedunthally@e360power.com; ksidhu@e360power.com; hshah@e360power.com'
    mail.CC = 'jpenelas@e360power.com; jshrewsbury@e360power.com; spatel@e360power.com; swatkins@e360power.com'
    mail.BCC = 'kjones@e360power.com'

    for idx, file in enumerate(image_paths):
        globals()[f'attachment{idx+1}'] = mail.Attachments.Add(file)
        globals()[f'cid{idx+1}'] = f'my_image_id{idx+1}'
        globals()[f'attachment{idx+1}'].PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F",globals()[f'cid{idx+1}'])

    mail.HTMLBody = f"""
        <html>
          <body>
          NOTE: The below HVaR data uses a 130-day or 6-month lookback period. 
            <h1 style="font-size:160%;text-align:center;">Group View</h1>
            <p><img src="cid:{cid1}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">PJM + Henry Only</h1>
            <p><img src="cid:{cid3}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">Subgroup View</h1>
            <p><img src="cid:{cid2}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">Group Correlation</h1>
            <p><img src="cid:{cid4}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">Subgroup Correlation</h1>
            <p><img src="cid:{cid5}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">DoD Δ in Group HVaR</h1>
            <p><img src="cid:{cid8}"></p>
            &nbsp;
            <p><img src="cid:{cid6}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">DoD Δ in Subgroup HVaR</h1>
            <p><img src="cid:{cid9}"></p> 
            &nbsp;
            <p><img src="cid:{cid7}"></p>
            &nbsp;
            <h1 style="font-size:160%;text-align:center;">HVaR vs. PnL</h1>
            <div>
                <p>
                    <img src="cid:{cid10}">
                    <img src="cid:{cid11}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid12}">
                    <img src="cid:{cid13}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid14}">
                    <img src="cid:{cid15}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid16}">
                    <img src="cid:{cid17}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid18}">
                    <img src="cid:{cid19}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid20}">
                    <img src="cid:{cid21}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid22}">
                    <img src="cid:{cid23}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid24}">
                    <img src="cid:{cid25}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid26}">
                    <img src="cid:{cid27}">
                </p>
                &nbsp;
                <p>
                    <img src="cid:{cid28}">
                    <img src="cid:{cid29}">
                </p>
            </div>   
            <br>
          </body>
        </html>
        """

    mail.Send()

image_list = [r"P:\KJ\Adhocs\thomas_eod_pnl_group.png", r"P:\KJ\Adhocs\thomas_eod_pnl.png",
              r"P:\KJ\Adhocs\thomas_eod_pnl_group_pjm_henry_df.png", r"P:\KJ\Adhocs\thomas_eod_pnl_group_corr.png",
              r"P:\KJ\Adhocs\thomas_eod_pnl_subgroup_corr.png",  r"P:\KJ\Adhocs\thomas_dod_group_var.png",
              r"P:\KJ\Adhocs\thomas_dod_subgroup_var.png",
              r"P:\KJ\Adhocs\thomas_dod_group_var_net_chng.png", r"P:\KJ\Adhocs\thomas_dod_subgroup_var_net_chng.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Portfolio Total.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Portfolio Total.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Basis.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Basis.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_ERCOT.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_ERCOT.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Environmental.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Environmental.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Global NG.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Global NG.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Henry Hub.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Henry Hub.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_Nepool.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_Nepool.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_PJM.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_PJM.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_West Power.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_West Power.png",
              r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_PJM+Henry.png", r"P:\KJ\Adhocs\thomas_hvar_vs_pnl_5day_PJM+Henry.png",]


send_email(image_list)