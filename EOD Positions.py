from dataflux import x2_int as x2
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from risk_tools import get_dates, format_df, e360_holidays
from pandas.tseries.offsets import CustomBusinessDay
import db
import dataframe_image as dfi
import win32com.client as win32
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from datetime import datetime, timedelta



today, last_bus_day = get_dates()#today_override = '01/04/2024')



positions = format_df(x2.PositionSnapshotV3(last_bus_day.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
print(last_bus_day, today)



cnxn = db._connect()
mapping = pd.read_sql('''select *, 
case 
	when e360_product_subgroup = 'PJM WH RT On' then 'PJM RT Strategy'
	when e360_product_subgroup = 'PJM WH RT Off' then 'PJM RT Strategy'
	when e360_product_subgroup in (select distinct e360_product_subgroup from e360_master_mapping where e360_product_group = 'PJM' and e360_product_subgroup not in ('PJM WH RT On', 'PJM WH RT Off'))
		then 'Rest of PJM'
	when e360_product_subgroup = 'East' then 'East Basis'
	else strat_lvl1 end strat_lvl3
from e360_master_mapping''', cnxn)



prod_groups = dict(zip(mapping.molecule_product_code, mapping.strat_lvl3))



def style_rowwise_total(df, cmap='RdYlGn'):
    # Exclude the 'Total' row and column before normalization
    df_no_total = df.copy()#drop(index='Total', columns='Total', errors='ignore')
    
    # Initialize the colormap with the specified color map
    norm = TwoSlopeNorm(vmin=df_no_total.min().min(),vcenter=df_no_total.median().median(), vmax=df_no_total.max().max())
    cm = ScalarMappable(norm=norm, cmap=cmap)
    
    # Function to apply color to each cell
    def apply_color(val):
        if pd.isna(val) or val == 'Total':
            return ''
        else:
            color = cm.to_rgba(val, bytes=False)
            return f'background-color: rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]})'

    # Normalize each row (excluding 'Total') and apply the coloring function
    styled_df = df.applymap(apply_color)

    return styled_df



def add_report_columns(df):
    df['Strategy'] = df.primary_product_code.map(prod_groups)
    df['Quarter'] = pd.PeriodIndex(df.contract_month, freq='Q')
    return df

add_report_columns(positions)



def create_report_pivots(df, values_col, mom_rep = False):
    if not mom_rep:
        curr_df = df.query("as_of_date == @today")
        prior_df = df.query("as_of_date == @last_bus_day")
    
        curr_pt = curr_df.pivot_table(values_col, 'Strategy', 'Quarter', 'sum', 0, True, margins_name = 'Total')
        prior_pt = prior_df.pivot_table(values_col, 'Strategy', 'Quarter', 'sum', 0, True, margins_name = 'Total')

        return curr_pt, prior_pt
    else:
        return df.pivot_table(values_col, 'Strategy', 'Quarter', 'sum', 0, True, margins_name = 'Total')



cp_pt, pp_pt = create_report_pivots(positions, 'base_currency_mtm_change')



dod_chng = cp_pt #- pp_pt

if 'Crude' in dod_chng.index:
    new_row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
else:
    new_row_order = ['Basis', 'ERCOT', 'Environmental',
   'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
    
dod_chng = dod_chng.reindex(new_row_order)

dod_styled = dod_chng.fillna(0).style.set_caption("Today's PnL").format('{:,.0f}').apply(style_rowwise_total, axis=None)
dfi.export(dod_styled, r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\dod_change_heatmap.png")
dod_styled


# ### Day over Day Position Change

dod_positions = format_df(x2.PositionSnapshotV3(last_bus_day.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
print(last_bus_day, today)



add_report_columns(dod_positions)


cp_pt_dod, pp_pt_dod = create_report_pivots(dod_positions, 'delta_position')




dod_chng = cp_pt_dod - pp_pt_dod

if 'Crude' in dod_chng.index:
    new_row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
else:
    new_row_order = ['Basis', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
    
dod_chng = dod_chng.reindex(new_row_order)
    


def style_rowwise(df, cmap='RdYlGn'):
    # Exclude the 'Total' row before normalization
    df_no_total = df.drop(index='Total', errors='ignore')
    
    # Find the max absolute value for normalization
    max_val = df_no_total.max().max()
    center = df_no_total.median().median()
    min_val = df_no_total.min().min()
    
    # Initialize the TwoSlopeNorm with the center at zero
    norm = TwoSlopeNorm(vmin=min_val, vcenter=center, vmax=max_val)
    cm = ScalarMappable(norm=norm, cmap=cmap)
    
    # Function to apply color to each cell
    def apply_color(val):
        if pd.isna(val) or val == 'Total':
            return ''
        else:
            color = cm.to_rgba(val, bytes=False)
            return f'background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})'

    # Normalize each row (excluding 'Total') and apply the coloring function
    styled_df = df.applymap(apply_color)
    
    # Ensure the 'Total' row is styled with a neutral color
    if 'Total' in df.index:
        styled_df.loc['Total'] = 'background-color: none'
#     if 'Total' in df.columns:
#         styled_df['Total'] = 'background-color: none'
    
    return styled_df



test_df = dod_chng.fillna(0)
dod_styled = test_df.style.set_caption('DoD Δ in Position').format('{:,.0f}').apply(style_rowwise, axis=None)
dfi.export(dod_styled, r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\dod_change_table.png")
dod_styled


# ### Month over Month Charts

today, last_bus_day = get_dates()#today_override='12/21/2023')

US_BUS_DAY = CustomBusinessDay(calendar=e360_holidays())

last_bus_day = ((today - timedelta(days = today.day)) - US_BUS_DAY).date()

real_mom_positions = format_df(x2.PositionSnapshotV3(last_bus_day.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')))
positions



print(last_bus_day, today)



cp_pt_mom_real = create_report_pivots(add_report_columns(real_mom_positions), 'base_currency_mtm_change', True)



real_mom_chng = cp_pt_mom_real #- pp_pt_mom_real 
real_mom_chng.fillna(0, inplace = True)

if 'Crude' in real_mom_chng.index:
    new_row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
else:
    new_row_order = ['Basis', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
    
real_mom_chng = real_mom_chng.reindex(new_row_order)
    
final_df = real_mom_chng.style.set_caption("MTD PnL").format('{:,.0f}').apply(style_rowwise_total, axis=None)
dfi.export(final_df, r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\mom_change_heatmap.png")
final_df



cp_pt_mom_real_pos, pp_pt_mom_real_pos = create_report_pivots(real_mom_positions, 'delta_position')




real_pos_mom_chng = cp_pt_mom_real_pos - pp_pt_mom_real_pos

if 'Crude' in real_pos_mom_chng.index:
    new_row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
else:
    new_row_order = ['Basis', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
    
real_pos_mom_chng = real_pos_mom_chng.reindex(new_row_order)
    
final_df2 = real_pos_mom_chng.fillna(0).style.set_caption('Δ in Position from Start of Month').format('{:,.0f}').apply(style_rowwise, axis=None)
dfi.export(final_df2, r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\mom_change_table.png")
final_df2


# ### Current Positions


curr_pos = cp_pt_dod.copy()

if 'Crude' in curr_pos.index:
    new_row_order = ['Basis', 'Crude', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
else:
    new_row_order = ['Basis', 'ERCOT', 'Environmental',
       'Global NG', 'Henry Hub', 'Nepool', 'PJM RT Strategy', 'Rest of PJM', 'East Basis', 'West Strategy', 'Total']
    
curr_pos = curr_pos.reindex(new_row_order)


final_df = curr_pos.fillna(0)
curr_pos_styled = final_df.style.set_caption('Current Position').format('{:,.0f}').apply(style_rowwise, axis=None)
dfi.export(curr_pos_styled, r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\curr_pos_table.png")


def send_email(image_paths):
    """Sends an email with the attached images inline"""
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = f"EOD PnL & DoD Position Changes: {today.strftime('%b-%d-%Y')}"
    mail.To = 'tnedunthally@e360power.com; ksidhu@e360power.com; hshah@e360power.com'
    mail.CC = 'spatel@e360power.com; swatkins@e360power.com; jpenelas@e360power.com; jshrewsbury@e360power.com'
    mail.BCC = 'kjones@e360power.com'

    for idx, file in enumerate(image_paths):
        globals()[f'attachment{idx+1}'] = mail.Attachments.Add(file)
        globals()[f'cid{idx+1}'] = f'my_image_id{idx+1}'
        globals()[f'attachment{idx+1}'].PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F",globals()[f'cid{idx+1}'])

    mail.HTMLBody = f"""
        <html>
          <body>
            <h1 style="font-size:150%;text-align:center;">DoD</h1>
            <div>
                <img src="cid:{cid1}">
                <br>
                <br>
                <br>
                <br>                
                <img src="cid:{cid2}">
                <br>
            </div> 
            <br>
            <h1 style="font-size:150%;text-align:center;">MTD</h1>
            <div>
                <img src="cid:{cid3}">
                <br>
                <br>
                <br>
                <br> 
                <img src="cid:{cid4}">
                <br>
            </div>  
            <br>
            <h1 style="font-size:150%;text-align:center;">Current</h1>
            <div>
                <img src="cid:{cid5}">
            </div>  
            <br>
          </body>
        </html>
        """

    mail.Send()

image_list = [r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\dod_change_heatmap.png", r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\dod_change_table.png",
              r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\mom_change_heatmap.png", r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\mom_change_table.png",
              r"\\e360-fs01\Shares\E360Power Docs\KJ\Adhocs\curr_pos_table.png"] 


send_email(image_list)

