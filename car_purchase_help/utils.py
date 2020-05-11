import pandas as pd
from car_purchase_help import constants


def clean_input(
    manufacturer: str, model: str, year: float, odometer: float
) -> (str, str, float, float):
    """
    Takes the input from the user and removes whitespace, punctuation and makes all
    test lowercase which is what is expected by the model. Also verifies that the vehicle
    details are present in the dataframe
    :param manufacturer: manufacturer of the vechicle
    :param model: model of the vehicle
    :param year: year of manufacture
    :return: cleaned and verified details that the user entered
    """
    manufacturer = manufacturer.strip().lower()
    model = model.strip().lower().replace("/", "").replace("\\", "")
    year = int(year)
    odometer = int(odometer)

    assert manufacturer in constants.MANUFACTURERS, "Manufacturer does not exist"
    # Over 28,000 models to check. Unsure how to check without loading the dataframe
    # assert model in constants.MODELS, "Model does not exist"
    assert year in constants.YEARS, "Year not valid, must be > 1980 and < 2019"
    assert (
        0 < odometer < constants.MAX_ODOMETER
    ), f"Mileage not valid, must be > 0 and < {constants.MAX_ODOMETER}"

    return manufacturer, model, year, odometer


def format_user_input(text: str) -> (str, str, int, int, int):
    """"
    Take the user input of `manufacturer, model, year, mileage, listed price` and seperates by comma
    then cleans
    :param text: `manufacturer, model, year, mileage, listed price`
    :return: tuple of cleaned manufacturer, model, year, mileage, listed price
    """
    text = text.split(",")
    assert len(text) == 5, (
        "Incorrect number of features provided. Should be 5 split by commas:"
        " manufacturer, model, year, mileage, listed price"
    )
    for t in text:
        assert len(t.strip()) > 0, "One of your fields is blank"

    return (
        text[0],
        text[1],
        int(text[2].strip()),
        int(text[3].strip()),
        int(text[4].strip()),
    )


def format_file_name(text: str):
    """
    Formats a filename to be savable by removing problem characters
    :param text: to be formatted
    :return: formatted string
    """
    return text.replace("/", "").replace("\\", "")
