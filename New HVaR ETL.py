import pyodbc
import pandas as pd
import numpy as np
from dataflux import x2_int as x2
# from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import datetime
import sys


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

today = datetime.date.today()
result = today #+ 1 * US_BUS_DAY
# result = pd.to_datetime('12/05/2023')
next_bus_day = result.strftime('%m-%d-%Y')

class ETLProcessor:
    def __init__(self):
        self.cnxn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            r'SERVER=e360-db01;'
            r'DATABASE=Voltage;'
            r'Trusted_Connection=yes;'
        )
        self.date_range = pd.bdate_range(next_bus_day, next_bus_day)
        self.date_range = self.date_range.map(lambda x: x.strftime('%m-%d-%Y'))

    @staticmethod
    def compute_etl(losses, confidence_level):
        var_threshold = np.percentile(losses, 100 - confidence_level)
        tail_losses = [loss for loss in losses if loss < var_threshold]
        print(f'These are the tail losses for {losses.name}: {tail_losses}')
        if len(tail_losses) == 0:
            return None
        else:
            return np.mean(tail_losses)

    def query_data(self, date): #All query_data methods in the subclasses below should be converted to X2 pulls
        raise NotImplementedError

    def process_data(self, df, date):
        raise NotImplementedError

    def upload_data(self, processed_data):
        x2.upload(self.loader_name, processed_data)

    def run(self):
        try:
            for date in self.date_range:
                df = self.query_data(date)
                processed_data = self.process_data(df, date)
                self.upload_data(processed_data)
        except:
            sys.exit(1)


class ByProductGroupProcessor(ETLProcessor):
    loader_name = 'PG HVaR ETL Loader'

    def query_data(self, date):
        query_str = f"""
        declare @as_of_date date = '{date}';
        select * from kj_hvar_by_pg
        where as_of_date = @as_of_date
        and company != 'PACELINE'
        union all
        select as_of_date, historical_date, 'Total Portfolio' as company, sum(hist_pnl) hist_pnl, e360_product_group 
            from kj_hvar_by_pg
                where as_of_date = @as_of_date
            group by as_of_date, historical_date, e360_product_group
            order by 5, 3, 2
        """
        return pd.read_sql(query_str, self.cnxn)

    def process_data(self, df, date):
        final_base = df.pivot_table('hist_pnl', index=['company', 'historical_date'], columns='e360_product_group',
                                    aggfunc='sum', sort=False).fillna(0)
        all_losses = []

        etl_final = pd.DataFrame()

        for company in final_base.index.get_level_values('company').unique():
            company_losses = final_base.loc[company]
            all_losses = [company_losses[column] for column in company_losses.columns]

            confidence_level = 99
            etl_values = [self.compute_etl(losses, confidence_level) for losses in all_losses]
            cols = [loss.name for loss in all_losses]
            as_of_date = [date for loss in all_losses]

            etl = pd.DataFrame({'as_of_date': as_of_date,
                                'e360_product_group': cols,
                                'etl_cvar': etl_values,
                                'company': company})

            etl.fillna(0, inplace=True)

            etl_final = pd.concat([etl_final, etl], axis=0)
        return etl_final


class ByProductCurveGroupProcessor(ETLProcessor):
    loader_name = 'PG CG HVaR ETL Loader'

    def query_data(self, date):
        query_str = f"""
        declare @as_of_date date = '{date}';
        select * from kj_hvar_by_pg_cg
            where as_of_date = @as_of_date
            and company != 'PACELINE'
        union all
        select as_of_date, historical_date, 'Total Portfolio' as company, sum(hist_pnl) hist_pnl, e360_product_group, curve_group
            from kj_hvar_by_pg_cg
                where as_of_date = @as_of_date
            group by as_of_date, historical_date, e360_product_group, curve_group
            order by 5, 6, 3, 2
        """
        return pd.read_sql(query_str, self.cnxn)

    def process_data(self, df, date):
        final_base = df.pivot_table('hist_pnl', index=['company', 'historical_date'],
                                    columns=['e360_product_group', 'curve_group'],
                                    aggfunc='sum', sort=False).fillna(0)
        all_losses = []

        etl_final = pd.DataFrame()

        for company in final_base.index.get_level_values('company').unique():
            company_losses = final_base.loc[company]
            all_losses = [company_losses[column] for column in company_losses.columns]

            confidence_level = 99
            etl_values = [self.compute_etl(losses, confidence_level) for losses in all_losses]
            cols = [loss.name for loss in all_losses]
            as_of_date = [date for loss in all_losses]

            etl = pd.DataFrame({'as_of_date': as_of_date,
                                'e360_product_and_curve_group': cols,
                                'etl_cvar': etl_values,
                                'company': company})

            etl.fillna(0, inplace=True)

            etl_final = pd.concat([etl_final, etl], axis=0)

        etl_final[['e360_product_group', 'curve_group']] = etl_final['e360_product_and_curve_group'].apply(pd.Series)
        etl_final.drop(columns='e360_product_and_curve_group', inplace=True)
        etl_final.curve_group = etl_final.curve_group.replace('\r', '', regex=True)
        return etl_final


class ByProductSubgroupProcessor(ETLProcessor):
    loader_name = 'SG HVaR ETL Loader'

    def query_data(self, date):
        query_str = f"""
        declare @as_of_date date = '{date}';
        select * from kj_hvar_by_sg
        where as_of_date = @as_of_date
        and company != 'PACELINE'
        union all
        select as_of_date, historical_date, 'Total Portfolio' as company, sum(hist_pnl) hist_pnl, e360_product_subgroup 
            from kj_hvar_by_sg
                where as_of_date = @as_of_date
            group by as_of_date, historical_date, e360_product_subgroup
            order by 5, 3, 2
        """
        return pd.read_sql(query_str, self.cnxn)

    def process_data(self, df, date):
        final_base = df.pivot_table('hist_pnl', index=['company', 'historical_date'], columns='e360_product_subgroup',
                                    aggfunc='sum', sort=False).fillna(0)
        all_losses = []

        etl_final = pd.DataFrame()

        for company in final_base.index.get_level_values('company').unique():
            company_losses = final_base.loc[company]
            all_losses = [company_losses[column] for column in company_losses.columns]

            confidence_level = 99
            etl_values = [self.compute_etl(losses, confidence_level) for losses in all_losses]
            cols = [loss.name for loss in all_losses]
            as_of_date = [date for loss in all_losses]

            etl = pd.DataFrame({'as_of_date': as_of_date,
                                'e360_product_subgroup': cols,
                                'etl_cvar': etl_values,
                                'company': company})

            etl.fillna(0, inplace=True)

            etl_final = pd.concat([etl_final, etl], axis=0)
        return etl_final


class BySubCurveGroupProcessor(ETLProcessor):
    loader_name = 'SG CG HVaR ETL Loader'

    def query_data(self, date):
        query_str = f"""
        declare @as_of_date date = '{date}';
        select * from kj_hvar_by_sg_cg
            where as_of_date = @as_of_date
            and company != 'PACELINE'
        union all
        select as_of_date, historical_date, 'Total Portfolio' as company, sum(hist_pnl) hist_pnl, e360_product_subgroup, curve_group
            from kj_hvar_by_sg_cg
                where as_of_date = @as_of_date
            group by as_of_date, historical_date, e360_product_subgroup, curve_group
            order by 5, 6, 3, 2
        """
        return pd.read_sql(query_str, self.cnxn)

    def process_data(self, df, date):
        final_base = df.pivot_table('hist_pnl', index=['company', 'historical_date'],
                                    columns=['e360_product_subgroup', 'curve_group'],
                                    aggfunc='sum', sort=False).fillna(0)
        all_losses = []

        etl_final = pd.DataFrame()

        for company in final_base.index.get_level_values('company').unique():
            company_losses = final_base.loc[company]
            all_losses = [company_losses[column] for column in company_losses.columns]

            confidence_level = 99
            etl_values = [self.compute_etl(losses, confidence_level) for losses in all_losses]
            cols = [loss.name for loss in all_losses]
            as_of_date = [date for loss in all_losses]

            etl = pd.DataFrame({'as_of_date': as_of_date,
                                'e360_sub_and_curve_group': cols,
                                'etl_cvar': etl_values,
                                'company': company})

            etl.fillna(0, inplace=True)

            etl_final = pd.concat([etl_final, etl], axis=0)

        etl_final[['e360_product_subgroup', 'curve_group']] = etl_final['e360_sub_and_curve_group'].apply(pd.Series)
        etl_final.drop(columns='e360_sub_and_curve_group', inplace=True)
        etl_final.curve_group = etl_final.curve_group.replace('\r', '', regex=True)
        return etl_final


# Run the ETL processes
processors = [ByProductGroupProcessor(), ByProductCurveGroupProcessor(), ByProductSubgroupProcessor(), BySubCurveGroupProcessor()]
for processor in processors:
    processor.run()
