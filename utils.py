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
    """
    This function prints out various information about the two dataframes `df_times` and `df_hourly_rides`
    before they are joined. The information includes:
    - Number of unique time points in df_times
    - Number of unique time points in df_hourly_rides
    - Test for duplicates in `df_times`
    - How many out of df_hourly_rides is in df_times?
    - Out of the df_times observations, what is the distribution of the times that do not appear in the df_hourly_rides?

    Parameters:
    - df_times (DataFrame): Dataframe containing the time points
    - df_hourly_rides (DataFrame): Dataframe containing the hourly rides

    """

    # Number of unique times in df_times
    print("Number of unique time points in df_times:", len(df_times.datetime.unique()))

    # Number of unique times in df_hourly_rides
    print("\nNumber of unique time points in df_hourly_rides :", len(df_hourly_rides.created_on_hour.unique()))

    # Test for duplicates in `df_times`
    print("\nDuplicates in `df_times`:", df_times.datetime.duplicated().sum())

    # How many out of df_hourly_rides is in df_times ?
    print("\nHow many out of df_hourly_rides is in df_times ? ",
          df_hourly_rides.created_on_hour.isin(df_times.datetime).sum())

    # Out of the df_times observations, what is the distribution of the times that do not appear in the df_hourly_rides?
    print("\nOut of the df_times observations, what is the distribution "
          "of the times that do not appear in the df_hourly_rides?\n",
          df_times[~df_times.datetime.isin(df_hourly_rides.created_on_hour)].datetime.dt.to_period('M').value_counts())


def add_time_unit_lag(df, time_unit, lag, value_column):
    """
    This function adds a time unit lag to a specified value column in a given dataframe.
    A new dataframe is created with the original datetime and value column, shifted by the specified lag and time unit.
    The original dataframe is then merged with the new dataframe on the datetime column.

    Parameters:
    - df (DataFrame): Dataframe to add the lag to
    - time_unit (str): Time unit to shift by ('s','m','h', 'd')
    - lag (int): Number of time units to shift by
    - value_column (str): Name of the value column to shift
    """

    df_lag = pd.concat([df[['datetime']] + pd.Timedelta(f"{lag}{time_unit}"),
                        df[[value_column]]],
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
    """
    This function takes in two dataframes, df_fit and df_predict, and calculates the average value per hour and day of week
    in the df_fit dataframe using the groupby_cols and value_col specified. It then assigns these predictions to the
    df_predict dataframe and renames the columns to "actual" and "predicted".

    Parameters:
    df_fit (DataFrame): The dataframe containing the data to fit the model on
    df_predict (DataFrame): The dataframe containing the data to make predictions on
    groupby_cols (list): A list of columns to groupby and calculate the average on
    value_col (str): The column containing the values to calculate the average of

    Returns:
    DataFrame: A dataframe containing the actual and predicted values
    """

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
    """
    This function plots the predicted vs actual values
    :param df: dataframe containing actual and predicted values
    :param fig_size: figure size of the plot
    :return:
    """

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
    """
    This function plots the predicted vs actual values for different intervals
    :param df: dataframe containing the data
    :param test: dataframe containing the test data
    :param days_back_range: range of days to test the intervals
    :param groupby_cols: columns to group the data by
    :param value_col: column containing the values to predict
    :return: dictionary containing the best mean absolute error and the interval it was achieved at
    """

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


def calculate_rolling_mean(df, value_col, index_col, mean_lag, time_unit):
    """
    The calculate_rolling_mean function takes in a dataframe df,
    a column name value_col which represents the value to calculate the rolling mean of,
    a column name index_col representing the index to roll over,
    a mean_lag representing the number of time units to average over,
    and a time_unit representing the unit of time to use
    (e.g. 'h' for hours, 'd' for days). It returns the rolling mean of the value_col shifted by one period.

    Parameters:
    df (pandas DataFrame): The dataframe on which the calculation should be performed
    value_col (str): The name of the column for which the rolling mean should be calculated
    index_col (str): The name of the column on which the rolling mean should be based
    mean_lag (int): Number of time units to use for the rolling mean calculation
    time_unit (str): The unit of time to use for the rolling mean calculation

    Returns:
    pandas Series: The rolling mean of the given value column
    """
    return df[[value_col, index_col]].rolling(f'{mean_lag}{time_unit}', on=index_col).mean() \
        .shift(periods=1)[value_col]


def fit_predict_baseline_b(df, df_predict, k_last_hours, value_col):
    """The fit_predict_baseline_b function takes in a dataframe df and a dataframe df_predict,
    a k_last_hours representing the number of hours to use as a lag,
    and a value_col representing the value to calculate the rolling mean of.
    The function then applies the rolling mean calculation using the calculate_rolling_mean function,
    and predicts the df_predict using the train dataset,
    then it returns the mean absolute error (MAE) of the
    predicted values and the dataframe containing the actual and predicted values.

    Parameters:
        df (pandas DataFrame): The training data
        df_predict (pandas DataFrame): The test data on which the prediction should be performed
        k_last_hours (int): Number of hours to use for the rolling mean calculation
        value_col (str): The name of the column for which the rolling mean should be calculated

    Returns:
        tuple: (MAE, DataFrame with actual and predicted values)

    """

    train = df[(df.datetime >= df_predict["datetime"].min() - pd.Timedelta(hours=k_last_hours))]
    # df_predict[f"predicted"] = train[[value_col]].rolling(f'{k_last_hours}h').mean().shift(periods=1)

    df_predict[f"predicted"] = calculate_rolling_mean(df=train,
                                                      value_col=value_col,
                                                      index_col="datetime",
                                                      mean_lag=k_last_hours,
                                                      time_unit='h')

    df_actual_vs_pred = df_predict.rename(columns={value_col: "actual"})

    mae = mean_absolute_error(df_actual_vs_pred.actual,
                              df_actual_vs_pred.predicted)

    return mae, df_actual_vs_pred


def plot_predicted_vs_actual_for_different_intervals_bl_b(df,
                                                          test,
                                                          hours_back_range,
                                                          value_col,
                                                          fig_size=(9, 6)):
    """The plot_predicted_vs_actual_for_different_intervals_bl_b function takes in a dataframe df, a dataframe test,
    a range of hours hours_back_range, a value column value_col,
    and a fig_size tuple representing the size of the plotted figure.
    It then plots predicted vs actual values for different intervals of hours_back_range
    and also plots a trend of mean absolute error for each interval.
    It returns the best mean absolute error and the best interval.

    """

    sns.set(rc={'figure.figsize': fig_size})

    mae_list = [np.nan] * hours_back_range
    best_mae = 1000
    best_mae_interval = hours_back_range
    for i in range(hours_back_range, 0, -1):
        mae, df_actual_vs_pred = fit_predict_baseline_b(df=df,
                                                        df_predict=test,
                                                        k_last_hours=i,
                                                        value_col=value_col)
        mae_list[i - 1] = mae

        if mae < best_mae:
            best_mae = mae
            best_mae_interval = i

        df_melt = df_actual_vs_pred[[f"predicted", "actual", "datetime"]].melt('datetime',
                                                                               var_name='cols',
                                                                               value_name='vals')

        fig, ax = plt.subplots(2, 1)

        sns.lineplot(
            data=df_melt,
            x="datetime", y="vals", hue="cols", ax=ax[0]
        ).set(title=f"predicted VS actual last {i} hours, MAE ={int(mae)}")

        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=15, ha="right")

        ax[1].plot(mae_list)

        ax[1].set_title("Mean Absolute Error Trend")
        ax[1].set_xlabel("Hours back interval")
        ax[1].set_ylabel("MAE")
        plt.xlim(-1, hours_back_range)
        plt.ylim(-1, 250)
        plt.tight_layout()
        ax[1].invert_xaxis()
        plt.show()

        clear_output(wait=True)

    return {"best_mae": best_mae, "best_mae_interval": best_mae_interval}


def data_processing_ml_approach(df, value_column, cat_cols, lag_features, avg_features, drop_na_target=True):
    # Add lag features
    for time_unit, lag in lag_features:
        df = add_time_unit_lag(df=df,
                               time_unit=time_unit,
                               lag=lag,
                               value_column=value_column)
    # Add rolling mean features
    for mean_lag, time_unit in avg_features:
        df[f'roll_avg_{mean_lag}{time_unit}_'] = calculate_rolling_mean(df=df,
                                                                        value_col=value_column,
                                                                        index_col="datetime",
                                                                        mean_lag=mean_lag,
                                                                        time_unit=time_unit)

    # Convert columns to categorical
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # Filtering na target
    if drop_na_target:
        df = df[~df[value_column].isna()]

    return df


def plot_history(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def plot_predicte_vs_actual(dataset, trainPredict, testPredict, look_back=1):
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def plot_feature_importance(model, X_test):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(7, 3))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance');
