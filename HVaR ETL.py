# import pyodbc
# from sqlalchemy import create_engine
import pandas as pd 
import numpy as np
from dataflux import x2_int as x2
# from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import datetime


pd.options.display.float_format = '{:,.2f}'.format

def e360_holidays():
    import holidays
    import numpy as np

    us_holidays = holidays.UnitedStates(years=[2023, 2024, 2025, 2026, 2027, 2028,
                                               2029, 2030, 2031, 2032, 2033, 2034],
                                        observed=True)

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
today = datetime.date.today()
result = today #+ 1 * US_BUS_DAY
next_bus_day = result.strftime('%m-%d-%Y')
print(next_bus_day)

# cnxn = pyodbc.connect(
#     r'DRIVER={ODBC Driver 17 for SQL Server};'
#     r'SERVER=e360-db01;'
#     r'DATABASE=Voltage;'
#     r'Trusted_Connection=yes;'
# )


def compute_etl(losses, confidence_level):
    var_threshold = np.percentile(losses, 100 - confidence_level)
    tail_losses = [loss for loss in losses if loss < var_threshold]
    print(f'These are the tail losses for {losses.name}: {tail_losses}')
    if len(tail_losses) == 0:
        return None
    else:
        return np.mean(tail_losses)


date = next_bus_day

df = x2.percent_var_vectors(date)
# df = pd.read_sql(f"""
#     select as_of_date, historical_date, bucket_name, company, sum(hist_pnl) hist_pnl from hvar_vectors_percent
#         where as_of_date = '{date}'
#         and company != 'PACELINE'
#         group by as_of_date, company, historical_date, bucket_name
#     union all
#     select as_of_date, historical_date, bucket_name, 'Total Portfolio' as company, sum(hist_pnl) hist_pnl
#     from hvar_vectors_percent
#         where as_of_date = '{date}'
#         group by as_of_date, historical_date, bucket_name""", cnxn)

final_base = df.pivot_table('hist_pnl', index = ['company', 'historical_date'], columns = 'bucket_name', aggfunc = 'sum', sort = False)
print(final_base)

all_losses = []
etl_final = pd.DataFrame()

for company in final_base.index.get_level_values('company').unique():
    company_losses = final_base.loc[company]
    all_losses = [company_losses[column] for column in company_losses.columns]

    confidence_level = 99
    etl_values = [compute_etl(losses, confidence_level) for losses in all_losses]
    cols = [loss.name for loss in all_losses]
    as_of_date = [date for loss in all_losses]

    etl = pd.DataFrame({'as_of_date': as_of_date,
                        'bucket_name': cols,
                        'etl_cvar': etl_values,
                        'company': company})

    etl.fillna(0, inplace=True)

    etl_final = pd.concat([etl_final, etl], axis=0)
    # etl_final.to_json()

x2.upload('HVaR ETL Loader', etl_final)

