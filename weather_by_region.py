import sys

from dataflux import x2_int as x2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import win32com.client as win32
from datetime import datetime, timedelta

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay

US_BUS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def create_plot(ax, whole_actual, actual_data, forecast_data,
                forecast_data_yesterday, forecast_data_prev2,  forecast_data_prev3,
                forecast_data_prev4, forecast_data_prev5, region):
    """Creates a plot for a given region"""
    # Define date ranges for different columns
    date_ranges = [
        ("pop_cdd", actual_data, 'black', '-', 3),
        ("pop_cdd_10y", whole_actual, 'purple', ':', 5),
        ("pop_cdd", forecast_data, 'red', '-', 3),
        ("pop_cdd", forecast_data_yesterday, 'blue', '--', 1),
        ("pop_cdd", forecast_data_prev2, 'green', '--', 1),
        ("pop_cdd", forecast_data_prev3, 'pink', '--', 1),
        ("pop_cdd", forecast_data_prev4, 'orange', '--', 1),
        ("pop_cdd", forecast_data_prev5, 'grey', '--', 1),
    ]


    # Select rows from index 1 to 17

    color_to_label = {
        'black': 'pop_cdd_obs',
        'red': 'pop_cdd_fc',
        'green': '2 days ago_fc',
        'pink': '3 days ago_fc',
        'orange': '4 days ago_fc',
        'grey': '5 days ago_fc',
        'blue': 'yesterday_fc',
        ('black', ':'): 'pop_cdd_10y_obs'
    }

    # Plot the data according to the specified date ranges
    for col, data_to_plot, color, linestyle, lw in date_ranges:
        print(f"Plotting: {col}")
        label = color_to_label.get((color, linestyle), color_to_label.get(color, col))
        ax.plot(data_to_plot['observation_date'], data_to_plot[col], label=label, color=color, linestyle=linestyle,
                linewidth=lw)

    # Configure x-axis and labels
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%m/%d/%Y"))
    ax.set_title(f'Weather Data Visualization for {region}')
    ax.set_xlabel('Observation Date')
    ax.set_ylabel('Degree Day')
    ax.legend()
    ax.grid(axis='x', linestyle='--', linewidth=0.5)
    ax.grid(axis='y', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', rotation=60, labelsize=10)


def fetch_data(from_date, to_date, region, as_of_date, as_of_year):
    """Fetches weather data for a given region"""
    ddf = x2.DegreeDayDataISOCWG(from_date, to_date, region, as_of_date, as_of_year)
    weather_df = ddf[['observation_date', 'pop_cdd', 'pop_cdd_10y', 'pop_cdd_30y']]
    weather_df.loc[:, 'observation_date'] = pd.to_datetime(weather_df['observation_date'], format="%m/%d/%Y %I:%M:%S %p")
    return weather_df


def send_email(image_path):
    """Sends an email with the attached image"""
    current_date = datetime.today().strftime('%Y-%m-%d')
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.Subject = f'CWG POP CDD by Power Regions for {current_date}'
    # mail.To = 'trading@e360power.com'
    mail.BCC = 'kjones@e360power.com'
    attachment = mail.Attachments.Add(image_path)
    cid = "my_image_id"
    attachment.PropertyAccessor.SetProperty("http://schemas.microsoft.com/mapi/proptag/0x3712001F", cid)

    mail.HTMLBody = f"""
        <html>
          <body>
            <p></p>
            <img src="cid:{cid}">
          </body>
        </html>
        """
    mail.Send()


def create_subplots():
    """Create subplots and figure."""
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=[25, 25])
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    return fig, axes


def plot_regions(fig, axes, regions):
    """Plot weather data for all regions."""
    current_date = datetime.today()
    one_week_ago = current_date - timedelta(weeks=1)
    yesterday = current_date - timedelta(days=1)
    five_days_ago = current_date - timedelta(days=5)

    # Define a list of previous business days for forecast data retrieval
    prev_BUS_DAYs = [
        current_date - n * US_BUS_DAY for n in range(2, 7)
    ]

    # ... [snip]

    for idx, region in enumerate(regions):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        # Get actual weather data
        weather_df_actual = fetch_data(one_week_ago.strftime("%Y-%m-%d"),
                                       current_date.strftime("%Y-%m-%d"),
                                       region, current_date.strftime("%Y-%m-%d"), current_date.strftime("%Y"))
        weather_df_actual['pop_cdd_10y'].fillna(method='ffill', inplace=True)
        weather_df_actual['pop_cdd_10y'].fillna(method='bfill', inplace=True)
        print(weather_df_actual)

        # Get yesterday weather data
        weather_df_yesterday = fetch_data(one_week_ago.strftime("%Y-%m-%d"),
                                          yesterday.strftime("%Y-%m-%d"),
                                          region, yesterday.strftime("%Y-%m-%d"), yesterday.strftime("%Y"))

        weather_df_yesterday.reset_index(drop=True, inplace=True)
        weather_df_actual.reset_index(drop=True, inplace=True)

        forecast_data_yesterday = weather_df_yesterday[weather_df_actual['observation_date'] > prev_BUS_DAYs[0]]

        # Fetch and select forecast data for previous days
        prev_forecast_datas = []
        for prev_DAY in prev_BUS_DAYs[1:]:
            weather_df_prev = fetch_data(one_week_ago.strftime("%Y-%m-%d"),
                                         yesterday.strftime("%Y-%m-%d"),
                                         region, prev_DAY.strftime("%Y-%m-%d"), prev_DAY.strftime("%Y"))
            forecast_data_prev = weather_df_prev[weather_df_actual['observation_date'] > prev_DAY]
            prev_forecast_datas.append(forecast_data_prev)

        # Split weather_df_actual into actual and forecast based on the current date
        actual_data = weather_df_actual[weather_df_actual['observation_date'] < yesterday]
        forecast_data = weather_df_actual[weather_df_actual['observation_date'] > yesterday]

        # ... [snip]

        create_plot(ax, weather_df_actual, actual_data, forecast_data,
                    forecast_data_yesterday, *prev_forecast_datas, region)

    # Removing empty subplots
    for idx in range(len(regions), 15):
        fig.delaxes(axes.flatten()[idx])

    plt.tight_layout()
    return fig


def save_and_email_figure(fig):
    """Save the figure and send it via email."""
    image_path = rf"C:\Users\kjones\Documents\ERC_Weather\images\weather_regions.png"
    fig.savefig(image_path)
    send_email(image_path)
    print("Email sent successfully!")


def main():
    regions = ["ERCOT", "SERC", "SPP",
               "MISO", "Upper Miso", "Lower Miso",
               "PJM", "West PJM", "East PJM",
               "CAISO", "WECC", "NEPOOL",
               "AESO", "NYISO", "TVA"]

    fig, axes = create_subplots()
    fig = plot_regions(fig, axes, regions)  # Corrected this line to pass the proper arguments
    save_and_email_figure(fig)


if __name__ == "__main__":
    main()
