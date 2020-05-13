from os import path
from pathlib import Path
import pickle
from car_purchase_help.utils import clean_input, format_user_input, format_file_name
from car_purchase_help.data_visualization import save_lin_reg_plot
from car_purchase_help.data_processing import remove_outliers
from car_purchase_help import constants
from car_purchase_help.linear_regression import Linear_Regression
import pandas as pd
import numpy as np


def fit_lin_regression(
    df: pd.DataFrame,
    manufacturer: str,
    model: str,
    year: float,
    save_plots: bool = False,
) -> str:
    """
    Given a car from a certain year fits a linear regression for milage
    against price and saves the model in a pkl file. For debugging purposes
    a plot of the linear regression is saved also
    :param df: dataframe containing the data (assuemd to be clean and formatted)
    :param manufacturer: manufacturer of the vechicle
    :param model: model of the vehicle
    :param year: year of manufacture
    :return: string indicating the status of the training
    """

    # Clean the input details and check whether a pkl file already exists for
    # the car
    manufacturer, model, year, _ = clean_input(manufacturer, model, year, 1000)
    model_file = Path(f"../models/{manufacturer}_{model}_{year}.pkl")
    if model_file.exists():
        return (
            "There is already a saved model for this vehicle, "
            "delete the pkl file if refitting necessary"
        )

    # Get the data for the vehicle given year from the dataframe
    data = df[
        (df["manufacturer"] == manufacturer)
        & (df["model"] == model)
        & (df["year"] == year)
    ]

    # Remove rows that contain outliers in the price column
    data = remove_outliers(data)

    # Check if there are enough data points to justify a regression to be fit
    if len(data) < constants.MIN_POINTS_TO_FIT:
        return (
            f"There were less than {constants.MIN_POINTS_TO_FIT} entries so did not fit"
        )

    # Format, then fit the regression
    X = data["odometer"].values.reshape(len(data), 1)
    y = data["price"].values.reshape(len(data), 1)
    linreg = Linear_Regression()
    linreg.fit(X, y)

    # Save a scatter plot of the data and the regression line TAKES TOO MUCH MEMORY CURRENTLY
    if save_plots:
        save_lin_reg_plot(manufacturer, model, year, X, y, linreg.get_y_preds())

    # Save the linear regression object by pickling the binary
    with open(model_file, "wb") as f:
        pickle.dump(linreg, f)
    return "Regression fit and saved successfully"


def predict_price(manufacturer: str, model: str, year: float, odometer: float):
    """
    Given a car from a certain year and mileage this returns the typical (according
    to linear regression) value it would be listed for on Craigslist
    :param df: dataframe containing the full car data
    :param manufacturer: manufacturer of the vehicle
    :param model: model of the vechicle
    :param year: year of manufacture of the vehicle
    :return: predicited price of the vechicle according to the data, however,
    if not enough data return -1, if regression fails -2, 
    """

    # Clean the input from the user and check that a pkl file exists
    manufacturer, model, year, odometer = clean_input(
        manufacturer, model, year, odometer
    )
    model_file = Path(f"models/{manufacturer}_{model}_{year}.pkl")
    assert path.isfile(model_file), "No regression model for this car from that year"

    # Load the linear regression and make the prediction
    with open(model_file, "rb") as f:
        linreg = pickle.load(f)
    prediction = linreg.predict(x=odometer)

    # Check the prediction returned a value and return it
    assert prediction != None, "Prediction failed"
    return max(prediction, 0), linreg.get_mean_absolute_residual()


def get_advice(
    predicted_price: float, listed_price: float, mean_absolute_residual: float
) -> str:
    """
    Takes the difference between the price predicted for the vehicle and the listed price
    and gives a recommendation compared to the mean absolute residual for the model
    :param predicted_price: the price the model predicted for the vehicle
    :param listed_price: the price the vehicle is listed for on the internet
    :param mean_absolute_residual: average distance from training data to the regression line
    :return: a text recommendation to be provided to the user
    """
    # The factor is how many RESIDUAL_FACTOR_DIFFs the prediction is above or
    # below the linear regression prediction at that point.
    # eg if listing is for 1,000 and the model predicts 2,000 and the
    # mean absolute residual is 500 then the price is 2 RESIDUAL_FACTOR_DIFFs
    # below the price. So this seems like a good deal.
    factor = (predicted_price - listed_price) / mean_absolute_residual
    if factor > 2 * constants.RESIDUAL_FACTOR_DIFF:
        return "This appears to be a <b>very good</b> deal."
    elif (
        1 * constants.RESIDUAL_FACTOR_DIFF
        < factor
        <= 2 * constants.RESIDUAL_FACTOR_DIFF
    ):
        return "This appears to be a <b>good</b> deal."
    elif abs(factor) <= 1 * constants.RESIDUAL_FACTOR_DIFF:
        return "This appears to be a <b>fair price for the car"
    elif (
        -2 * constants.RESIDUAL_FACTOR_DIFF
        <= factor
        < -1 * constants.RESIDUAL_FACTOR_DIFF
    ):
        return "This appears to be a <b>bad</b> deal."
    else:
        return "This appears to be a <b>very bad</b> deal."
