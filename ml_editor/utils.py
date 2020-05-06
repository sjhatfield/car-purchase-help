import string
import pandas as pd
from ml_editor import constants


def clean_input(
    df: pd.DataFrame, manufacturer: str, model: str, year: float, odometer: float
) -> (str, str, float, float):
    """
    Takes the input from the user and removes whitespace, punctuation and makes all
    test lowercase which is what is expected by the model. Also verifies that the vehicle
    details are present in the dataframe
    :param df: dataframe containing the data (assuemd to be clean and formatted)
    :param manufacturer: manufacturer of the vechicle
    :param model: model of the vehicle
    :param year: year of manufacture
    :return: cleaned and verified details that the user entered
    """
    manufacturer = (
        manufacturer.strip()
        .translate(str.maketrans("", "", string.punctuation))
        .lower()
    )
    model = model.strip().translate(str.maketrans("", "", string.punctuation)).lower()
    year = int(year)
    odometer = int(odometer)

    assert manufacturer in df["manufacturer"].unique(), "Manufacturer does not exist"
    assert model in df["model"].unique(), "Model does not exist"
    assert year in df["year"].unique(), "Year not valid, must be > 1980 and < 2019"
    assert (
        0 < odometer < constants.MAX_ODOMETER
    ), f"Mileage not valid, must be > 0 and < {constants.MAX_ODOMETER}"

    return manufacturer, model, year, odometer


def remove_outliers(car_details: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a dataframe just containing the details for a single 
    car (manufacturer and model) for a specific year and
    """
    return
