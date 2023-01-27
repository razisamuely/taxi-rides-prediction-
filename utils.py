import warnings
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

warnings.filterwarnings('ignore')


def plot_values_on_daily_level(df, value_column, time_column, roll_avg_days):
    """
    This function plots the values of a given column (value_column) on a daily level. It also calculates the rolling average
    of the values over a specified number of days (roll_avg_days) and plots that as well.
    df: DataFrame - DataFrame containing the data to be plotted
    value_column: str - Name of the column containing the values to be plotted
    time_column: str - Name of the column containing the time/date information
    roll_avg_days: int - Number of days to use for calculating the rolling average
    """

    sns.set(rc={'figure.figsize': (12, 5)})

    df_daily_rides = df.groupby([df[time_column].dt.date]).sum()
    df_daily_rides = df_daily_rides.reset_index().rename(columns={time_column: "days"})
    df_daily_rides[f"roll_avg_{roll_avg_days}_days"] = df_daily_rides.rolling(roll_avg_days).mean()

    df_melt = df_daily_rides.melt('days', var_name='cols', value_name='vals')
    sns.lineplot(
        data=df_melt,
        x="days", y="vals", hue="cols"
    ).set_title(f"{value_column} on daily level")

    return df_daily_rides


def sample_and_plot_histogram(df, per_of_sample, value_columns, number_of_samples):
    """
    This function samples a specified percentage (per_of_sample) of the data and plots histograms of the specified
    value_columns. It repeats this process a specified number of times (number_of_samples)
    df: DataFrame - DataFrame containing the data to be plotted
    per_of_sample: float - percentage of the data to be sampled (between 0 and 1)
    value_columns: list of str - List of column names containing the values to be plotted in the histograms
    number_of_samples: int - Number of times to sample and plot the histograms
    """

    n_sampels = int(df.shape[0] * per_of_sample)

    for i in range(number_of_samples):
        df_sampled = df[value_columns].sample(n=n_sampels)
        sns.displot(df_sampled, kde=True, )
        plt.title(str(i))
        plt.ylim([0, 1300])
        plt.xlim([-10000, 150000])
        plt.show()
        clear_output(wait=True)


def queries_before_join(df_times, df_hourly_rides):
    # Number of unique times in df_times
    print("Number of unique time points in df_times:", len(df_times.datetime.unique()))

    # Number of unique times in df_hourly_rides
    print("\nNumber of unique time points in df_hourly_rides :", len(df_hourly_rides.created_on_hour.unique()))

    # Test for duplicates in `df_times`
    print("\nDuplicates in `df_times`:", df_times.datetime.duplicated().sum())

    # How many out of df_hourly_rides is in df_times ?
    print("\now many out of df_hourly_rides is in df_times ? ",
          df_hourly_rides.created_on_hour.isin(df_times.datetime).sum())

    # Out of the df_times observations, what is the distribution of the times that do not appear in the df_hourly_rides?
    print("\nOut of the df_times observations, what is the distribution "
          "of the times that do not appear in the df_hourly_rides?\n",
          df_times[~df_times.datetime.isin(df_hourly_rides.created_on_hour)].datetime.dt.to_period('M').value_counts())


def add_time_unit_lag(df, time_unit, lag):
    df_lag = pd.concat([df[['datetime']] + pd.Timedelta(f"{lag}{time_unit}"),
                        df[['rides']]],
                       axis=1
                       )

    df_lag.rename(columns={"rides": f"rides_last_{time_unit}_{lag}",
                           "datetime": f"datetime_{time_unit}_{lag}"},
                  inplace=True)

    df = pd.merge(df,
                  df_lag,
                  left_on='datetime',
                  right_on=f"datetime_{time_unit}_{lag}",
                  how="left")

    return df


def fit_predict_baseline_a(df_fit, df_predict, groupby_cols, value_col):
    # Calculating average per hour and day of week
    dow_hour_pred = df_fit.groupby(groupby_cols).agg({"rides": "mean"})

    # Assiging predictions
    df_actual_vs_pred = pd.merge(df_predict,
                                 dow_hour_pred,
                                 left_on=groupby_cols,
                                 right_on=groupby_cols,
                                 how="left")[["datetime", f"{value_col}_x", f"{value_col}_y"]]

    # Renaming columns
    df_actual_vs_pred.rename(columns={f"{value_col}_x": "actual", f"{value_col}_y": "predicted"}, inplace=True)

    return df_actual_vs_pred


def plot_predicted_vs_actual(df, fig_size=(9, 4)):
    mae = mean_absolute_error(df.actual,
                              df.predicted)
    sns.set(rc={'figure.figsize': fig_size})
    df_melt = df.melt('datetime', var_name='cols', value_name='vals')
    ax = sns.lineplot(
        data=df_melt,
        x="datetime", y="vals", hue="cols"
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, ha="right")
    ax.set_title(f"predicted VS actual, MAE ={int(mae)}")


def plot_predicted_vs_actual_for_different_intervals(df,
                                                     test,
                                                     days_back_range,
                                                     groupby_cols,
                                                     value_col):
    mae_list = [np.nan] * 46
    best_mae = 1000
    best_mae_interval = days_back_range

    for i in range(days_back_range, 0, -1):

        train = df[(df.datetime >= test.datetime.min() - pd.Timedelta(days=i)) & \
                   (df.datetime < test.datetime.min())]

        df_actual_vs_pred = fit_predict_baseline_a(df_fit=train,
                                                   df_predict=test,
                                                   groupby_cols=groupby_cols,
                                                   value_col=value_col)

        if df_actual_vs_pred.predicted.isna().sum() < 1:
            mae = mean_absolute_error(df_actual_vs_pred.actual,
                                      df_actual_vs_pred.predicted)
            if mae < best_mae:
                best_mae = mae
                best_mae_interval = i

            mae_list[i - 1] = mae
            fig, ax = plt.subplots(2, 1)

            ax[0].plot(mae_list)
            ax[0].set_title(f"Mean Absolute Error {int(mae)}")
            ax[0].set_xlabel("Days back interval")
            ax[0].set_ylabel("MAE")
            ax[0].invert_xaxis()

            plot_predicted_vs_actual(df=df_actual_vs_pred,
                                     fig_size=(12, 5))
            plt.tight_layout()
            plt.show();
            clear_output(wait=True)

    return {"best_mae": best_mae, "best_mae_interval": best_mae_interval}


def fit_predict_baseline_b(df, df_predict, k_last_hours, value_col):
    df_predict = df_predict.set_index("datetime")
    train = df[(df.datetime >= df_predict.index.min() - pd.Timedelta(hours=k_last_hours))].set_index("datetime")
    df_predict[f"predicted"] = train[[value_col]].rolling(f'{k_last_hours}h').mean().shift(periods=1)
    df_actual_vs_pred = df_predict.rename(columns={value_col: "actual"})

    mae = mean_absolute_error(df_actual_vs_pred.actual,
                              df_actual_vs_pred.predicted)

    return mae, df_actual_vs_pred


# def plot_predicted_vs_actual_for_different_intervals_bl_b(df,
#                                                           test,
#                                                           days_back_range,
#                                                           groupby_cols,
#                                                           value_col):
