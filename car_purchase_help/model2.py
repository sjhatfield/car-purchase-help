from os import path
from pathlib import Path
import pickle
from car_purchase_help.utils import clean_input
from car_purchase_help import constants


def predict_mileage_cost(manufacturer: str, model: str, year: int):
    """
    Given a vehicle from a certain year predict the amount
    of value it loses for each 10,000 miles driven
    :param manufacturer: the maker of the car
    :param model: the model of car
    :param year: the year the car was released
    :return: amount the value will decrease every 10,000 miles driven
    """
    manufacturer, model, year, _ = clean_input(manufacturer, model, year, 1000)
    model_file = Path(f"models/{manufacturer}_{model}_{year}.pkl")
    # assert path.isfile(model_file), "No regression model for this car from that year"

    # Load the linear regression and make the prediction
    with open(model_file, "rb") as f:
        linreg = pickle.load(f)

    initial_value = linreg.predict(x=0)
    end_value = linreg.predict(x=constants.MILEAGE_DECREASE_PERIOD)
    decrease = initial_value - end_value
    assert decrease > 0, "Mileage cost prediction failed"

    return (
        "For every 10,000 additional miles this car is driven"
        f", its value will decrease by approximately <b>${round(decrease, 2)}</b>."
    )
