import dataframe_image as dfi
import datetime
from dateutil.relativedelta import *
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import BMonthEnd
import pandas as pd
import numpy as np
from dataflux import x2_int as x2


def get_business_day():
    result = today - 1 * US_BUS_DAY
    last_bus_day = result.strftime('%m-%d-%Y')
    curr_bus_day = today.strftime('%m-%d-%Y')
    return last_bus_day, curr_bus_day


def get_num_working_days_next_month():
    next_month_start = (pd.Timestamp.now() + pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
    next_month_end = (pd.to_datetime(next_month_start) + BMonthEnd(1)).strftime('%Y-%m-%d')
    date_range = pd.date_range(start=next_month_start, end=next_month_end, freq=US_BUS_DAY)
    num_working_days_next_month = len(date_range) * 16
    return num_working_days_next_month


def format_df(df):
    date_cols = ['as_of_date', 'contract_month', 'expiry_date', 'tenor', 'trade_date', 'option_strip',
                 'expiration_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%m/%d/%Y %X %p').dt.date
    return df


def get_settles_by_date(last_bus_day):
    settles = x2.Get_ICE_Settles_by_Date(last_bus_day)
    settles = format_df(settles)
    return settles


def get_prompt_settle(settles, ticker):
    prompt_settle = settles[(settles.exchange_code == ticker) & (
                settles.tenor == today.replace(day=1) + relativedelta(months=1))].reset_index(drop=True)
    return prompt_settle


def get_pin_risk_data(day, ticker):
    prod_opts = x2.pin_risk_report(day, ticker + '.O')
    prod_opts = format_df(prod_opts)

    prod_fp = x2.pin_risk_report(day, ticker)
    prod_fp = format_df(prod_fp)
    return prod_opts, prod_fp


def get_opt_prices(last_bus_day, ticker):
    opt_prices = x2.ICE_Option_Prices(ticker, last_bus_day, last_bus_day)
    opt_prices = format_df(opt_prices)
    return opt_prices


def get_underlying_mw(prod_fp):
    underlying_mw = prod_fp.contracts.sum()
    return underlying_mw


def get_contract_month_codes(prod_fp):
    contract_month_codes = {
        1: 'F',
        2: 'G',
        3: 'H',
        4: 'J',
        5: 'K',
        6: 'M',
        7: 'N',
        8: 'Q',
        9: 'U',
        10: 'V',
        11: 'X',
        12: 'Z'
    }

    # Extract month and map to month code
    month_num = prod_fp.contract_month[0].month
    month_code = contract_month_codes[month_num]

    # Extract year in YY format
    contract_year = prod_fp.contract_month[0].strftime('%y')
    return month_code, contract_year


def get_focus_strikes_and_hypothetical_settles(prod_opts, prompt_settle):
    focus_strikes = list(prod_opts.strike_price.unique())
    hypothetical_settles = [price + offset for price in focus_strikes for offset in [-0.01, 0.01]]

    if ticker == 'PMI':
        hypothetical_settles = [hypo_set for hypo_set in hypothetical_settles if (
                    (hypo_set >= prompt_settle.price[0] - 20.01) & (hypo_set <= prompt_settle.price[0] + 20.01))]
    else:
        hypothetical_settles = [hypo_set for hypo_set in hypothetical_settles] #if ((hypo_set >= prompt_settle.price[0] - 75.01) & (hypo_set <= prompt_settle.price[0] + 75.01))
    return focus_strikes, hypothetical_settles


def get_lookup_table(prod_opts):
    lookup_table = prod_opts.groupby(['strike_price', 'put_call_flag'], as_index=0)['contracts'].sum()

    lookup_table.put_call_flag = [lookup_table.put_call_flag[i][0].upper() for i in range(len(lookup_table))]
    lookup_table.strike_price = [lookup_table.strike_price[i].astype('float64') for i in range(len(lookup_table))]
    lookup_table['opt_uid'] = [str(lookup_table.strike_price[i]) + lookup_table.put_call_flag[i] for i in
                               range(len(lookup_table))]
    return lookup_table


def get_prompt_opt_prices(opt_prices, today, focus_strikes):
    # The below datetime needs to be the dynamic prompt month
    prompt_opt_prices = opt_prices[(opt_prices.option_strip == today.replace(day=1) + relativedelta(months=1))
                                   & (opt_prices.strike_price.isin(focus_strikes))].reset_index(drop=True,
                                                                                                inplace=False)

    prompt_opt_prices['opt_uid'] = [str(prompt_opt_prices.strike_price[i]) + prompt_opt_prices.put_or_call[i] for i in
                                    range(len(prompt_opt_prices))]

    prompt_opt_prices = prompt_opt_prices[prompt_opt_prices.opt_uid.isin(lookup_table.opt_uid)].reset_index(drop=True,
                                                                                                            inplace=False)

    return prompt_opt_prices


def get_test(prod_opts):
    test = prod_opts.groupby(['strike_price', 'put_call_flag'], as_index=False)['contracts'].sum()
    return test


def calculate_intrinsic_prem(test, hypothetical_settles, num_working_days_next_month):
    for i in range(len(test)):
        for j in hypothetical_settles:
            if test.put_call_flag[i] == 'put':
                if j - test.strike_price[i] < 0:
                    test.loc[i, j] = abs(j - test.strike_price[i]) * test.contracts[i] * num_working_days_next_month
                else:
                    test.loc[i, j] = 0
            else:
                if test.strike_price[i] - j < 0:
                    test.loc[i, j] = abs(j - test.strike_price[i]) * test.contracts[i] * num_working_days_next_month
                else:
                    test.loc[i, j] = 0

    intrinsic_prem = test.iloc[:, 3:].sum(axis=0)
    return intrinsic_prem.astype('int64')


def get_test1(prompt_opt_prices):
    test1 = prompt_opt_prices.groupby(['strike_price', 'put_or_call'], as_index=False)['settlement_price'].mean()
    return test1


def calculate_extrinsic_prem(test1, hypothetical_settles, num_working_days_next_month):
    for i in range(len(test1)):
        for j in hypothetical_settles:
            if test1.put_or_call[i] == 'P':
                if j < test1.strike_price[i]:
                    test1.loc[i, j] = (test1.settlement_price[i] - abs(j - test1.strike_price[i])) * test.contracts[
                        i] * num_working_days_next_month
                else:
                    test1.loc[i, j] = test1.settlement_price[i] * test.contracts[i] * num_working_days_next_month
            else:
                if test1.strike_price[i] < j:
                    test1.loc[i, j] = (test1.settlement_price[i] - abs(j - test1.strike_price[i])) * test.contracts[
                        i] * num_working_days_next_month
                else:
                    test1.loc[i, j] = test1.settlement_price[i] * test.contracts[i] * num_working_days_next_month
    ext_prem = test1.iloc[:, 3:].sum(axis=0)
    return ext_prem.astype('int64')


def create_df_sets(prod_opts):
    df_sets = [prod_opts.loc[prod_opts.company == name, :].reset_index(drop=True) for name in
               list(prod_opts.company.unique())]
    return df_sets


def process_portfolio(df_sets, hypothetical_settles):
    portfolio = pd.DataFrame()
    # For each company in the dataset
    for df in df_sets:
        # For each row in the company
        for row in np.arange(len(df)):
            # If the row is a put
            if df.loc[row, 'put_call_flag'] == 'put':
                # If the row is a short
                if df.loc[row, 'contracts'] < 0:
                    # For every potential underlying settle @ +/- $0.01
                    for expectation in hypothetical_settles:
                        # If the hypothetical settle is larger than the row's strike
                        if expectation > df.loc[row, 'strike_price']:
                            df.loc[row, expectation] = 0
                        else:
                            df.loc[row, expectation] = df.loc[row, 'contracts'] * -1
                # If the row is a long
                else:
                    # For every potential underlying settle @ +/- $0.01
                    for expectation in hypothetical_settles:
                        # If the hypothetical settle is smaller than the row's strike
                        if expectation < df.loc[row, 'strike_price']:
                            df.loc[row, expectation] = df.loc[row, 'contracts'] * -1
                        else:
                            df.loc[row, expectation] = 0
            # If the row is a call
            else:
                # If the row is a long
                if df.loc[row, 'contracts'] > 0:
                    # For every potential underlying settle @ +/- $0.01
                    for expectation in hypothetical_settles:
                        # If the hypothetical settle is larger than the row's strike
                        if expectation > df.loc[row, 'strike_price']:
                            df.loc[row, expectation] = df.loc[row, 'contracts']
                        else:
                            df.loc[row, expectation] = 0
                # If the row is a short
                else:
                    # For every potential underlying settle @ +/- $0.01
                    for expectation in hypothetical_settles:
                        # If the hypothetical settle is smaller than the row's strike
                        if expectation < df.loc[row, 'strike_price']:
                            df.loc[row, expectation] = 0
                        else:
                            df.loc[row, expectation] = df.loc[row, 'contracts']
        portfolio = pd.concat([portfolio, df], axis=0)
    return portfolio


def calculate_monthly_exercise_mw(portfolio):
    monthly_exercise_mw = portfolio.iloc[:, 6:].sum(axis=0)
    return monthly_exercise_mw.astype('int64')


def calculate_net_exposure(monthly_exercise_mw, underlying_mw):
    net_exposure = monthly_exercise_mw + underlying_mw
    return net_exposure.astype('int64')


def set_fixed_price(underlying_mw, hypothetical_settles):
    fixed_price = [underlying_mw for settles in hypothetical_settles]
    return fixed_price


def calculate_underlying_pnl(underlying_mw, num_working_days_next_month, hypothetical_settles, prompt_settle):
    underlying_pnl = [int((underlying_mw * num_working_days_next_month) * (hypo_set - prompt_settle.price[0])) for
                      hypo_set in hypothetical_settles]
    return underlying_pnl


def create_net_position_df(monthly_exercise_mw, fixed_price, net_exposure, intrinsic_prem, ext_prem,
                           underlying_pnl, hypothetical_settles, month_code, contract_year, prompt_settle):
    last_fp_settle = [prompt_settle.price[0] for row in range(7)]
    total_pnl = intrinsic_prem.values + ext_prem.values + underlying_pnl

    net_position = pd.DataFrame(
        data=[monthly_exercise_mw.values, fixed_price, net_exposure.values, intrinsic_prem.values,
              ext_prem.values, underlying_pnl, total_pnl],
        index=['Monthly Exercise (MW)', 'Underlying (MW)', 'Net Position (MW)',
               '(Intrinsic Px @ Exp - Strike Px) * Position = Intrinsic Premium',
               '(Option Prem - Intrinsic Px) * Position = Extrinsic Premium',
               '(Strike Px - Mkt Px @ Exp) * Underlying Position = Underlying PnL',
               'Total PnL'],
        columns=hypothetical_settles
    )

    net_position['Contract Month'] = month_code + contract_year
    net_position.insert(0, 'Contract Month', net_position.pop('Contract Month'))

    net_position['Underlying Last Settle Price'] = last_fp_settle
    net_position.insert(0, 'Underlying Last Settle Price', net_position.pop('Underlying Last Settle Price'))

    net_position.iloc[:, 2:] = net_position.iloc[:, 2:].applymap(lambda x: "{:,.0f}".format(x))
    return net_position


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

tickers = ['PMI']

last_bus_day, curr_bus_day = get_business_day()
settles = get_settles_by_date(last_bus_day)
num_working_days_next_month = get_num_working_days_next_month()

# today = pd.to_datetime('02-10-2024').date()


all_net_positions = []

for ticker in tickers:
    print(ticker)
    prompt_settle = get_prompt_settle(settles, ticker)

    prod_opts, prod_fp = get_pin_risk_data(last_bus_day, ticker)

    opt_prices = get_opt_prices(last_bus_day, ticker)

    underlying_mw = get_underlying_mw(prod_fp)
    month_code, contract_year = get_contract_month_codes(prod_fp)

    focus_strikes, hypothetical_settles = get_focus_strikes_and_hypothetical_settles(prod_opts, prompt_settle)
    lookup_table = get_lookup_table(prod_opts)

    prompt_opt_prices = get_prompt_opt_prices(opt_prices, today, focus_strikes)

    test = get_test(prod_opts)
    test1 = get_test1(prompt_opt_prices)

    intrinsic_prem = calculate_intrinsic_prem(test, hypothetical_settles, num_working_days_next_month)
    ext_prem = calculate_extrinsic_prem(test1, hypothetical_settles, num_working_days_next_month)

    df_sets = create_df_sets(prod_opts)
    portfolio = process_portfolio(df_sets, hypothetical_settles)

    monthly_exercise_mw = calculate_monthly_exercise_mw(portfolio)
    net_exposure = calculate_net_exposure(monthly_exercise_mw, underlying_mw)

    fixed_price = set_fixed_price(underlying_mw, hypothetical_settles)

    underlying_pnl = calculate_underlying_pnl(underlying_mw, num_working_days_next_month, hypothetical_settles,
                                              prompt_settle)

    net_position = create_net_position_df(monthly_exercise_mw, fixed_price, net_exposure, intrinsic_prem, ext_prem,
                                          underlying_pnl, hypothetical_settles, month_code, contract_year,
                                          prompt_settle)

    dfi.export(net_position, fr'\\10.200.10.17\Share\Vertex\Risk\\{ticker.lower()}_monthly_pin_risk.png')
    all_net_positions.append(net_position)

counter = 0
for net_position in all_net_positions:
    print(f'{tickers[counter]} Monthly Option Exposure')
    print(net_position)
    counter += 1