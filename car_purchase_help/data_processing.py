from car_purchase_help import constants

from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd
from collections import defaultdict


def format_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleanup data
    :param df: raw DataFrame
    :return: processed DataFrame
    """
    # Dropping the county column as it is completely missing
    df = df.drop(["county"], axis=1)

    # Setting types and fill NAs
    df["id"].fillna(-1, inplace=True)
    df["id"] = df["id"].astype(int)
    df["year"].fillna(-1, inplace=True)
    df["year"] = df["year"].astype(int)
    df["odometer"].fillna(-1, inplace=True)
    df["odometer"] = df["odometer"].astype(int)
    df["description"].fillna("", inplace=True)
    df["lat"].fillna(-1, inplace=True)
    df["long"].fillna(-1, inplace=True)

    # Use the provided ID as the index
    df.set_index("id", inplace=True, drop=True)

    # Limiting the range of the numerical columns to remove clear outliers
    df = df[(df["year"] > 1980) & (df["year"] < 2020)]
    df = df[(df["odometer"] > 1000) & (df["odometer"] < 300000)]
    df = df[df["price"] >= 500]

    return df


def split_by_description(
    formatted_df: pd.DataFrame,
    description_column: str = "description",
    test_size: float = constants.TEST_PROPORTION,
    random_state: int = constants.RANDOM_STATE,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits the dataframe into a training set and testing set. Keeps rows
    with the same description all together in the training or test set. This
    is because certian companies who sell a large amount of cars on Craigslist
    use the same description. This would be an example of data leakage from
    training set to test set.
    :param formatted_df: dataframe that has already been passed through the formatter
    """

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    splits = splitter.split(formatted_df, groups=formatted_df[description_column])
    train_idx, test_idx = next(splits)
    return formatted_df.iloc[train_idx, :], formatted_df.iloc[test_idx, :]


def remove_outliers(df: pd.DataFrame, column: str = "price", mode: str = "IQR"):
    """
    Removes outliers from the provided column of the dataframe. Outliers
    are values that are outside 3 standard deviations of the mean of the column
    :param df: dataframe assumed to be clean and formatted
    :param column: column to remove the outliers from. Usually this will be price
    :return: the same dataframe with rows removed which contain an outlier for the provided column
    """
    mode = mode.upper()
    assert mode in ["IQR", "SD"], (
        "mode is either IQR for outside 1.5 IQR method"
        " or SD for outside 3 standard deviations from the mean method"
    )
    if mode == "SD":
        return df[((df[column] - df[column].mean()) / df[column].std()).abs() < 3]
    else:
        Q3 = df[column].quantile(0.75)
        Q1 = df[column].quantile(0.25)
        IQR = Q3 - Q1
        return df[(df[column] < Q3 + 1.5 * IQR) & (df[column] > Q1 - 1.5 * IQR)]


def find_manufacturer(car_text: str, manufacturer_list: list):
    """
    :param car_text: string of car make and model information of the form `make-model` where
        make may contain hyphes or spaces.
    :param manfucaturer_list: list containing the manufacturers to check
    :return: if the car has manufacturer in the list then returns the 
        manufacturer, else returns None
    """
    split = car_text.split("-")
    possible_manufacturer_space = split[0]
    possible_manufacturer_hyphen = split[0]
    for chunk in split[1:]:
        if possible_manufacturer_space in manufacturer_list:
            return possible_manufacturer_space
        if possible_manufacturer_hyphen in manufacturer_list:
            return possible_manufacturer_hyphen
        possible_manufacturer_space = (
            possible_manufacturer_space + " " + possible_manufacturer_space
        )
        possible_manufacturer_hyphen = (
            possible_manufacturer_hyphen + "-" + possible_manufacturer_hyphen
        )
    return None


def find_model(car_text: str, model_list: list):
    """
    :param car_text: string of car make and model information of the form `make-model` where
        make may contain hyphes or spaces.
    :param model_list: list containing all the models in the Craiglist dataset
    :return: if the model is in the list return it's text
        properly formatted, else return None
    """
    split = car_text.split("-")
    possible_model_space = split[-1]
    possible_model_hyphen = split[-1]
    for chunk in split[-2::-1]:
        if possible_model_space in model_list:
            return possible_model_space
        if possible_model_hyphen in model_list:
            return possible_model_hyphen
        possible_model_hyphen = chunk + " " + possible_model_hyphen
        possible_model_hyphen = chunk + "-" + possible_model_hyphen
    return None


def clean_similar_cars_dict(similar_cars: dict, df: pd.DataFrame):
    """
    Takes the dictionary of similar cars and from the list of similar cars removes
    any which are not in the Craiglist dataset.
    :param similar_cars: This is dictionary where the keys are strings of the form
        `make,model` and the values are sets of strings of similar cars. The similar 
        car strings are of the form `make-model` where the make and model can contain
        hyphens also.
    :param df: the dataframe of Craiglist data
    :return: dictionary of similar cars where the only car in the value lists are actually
        present in the original dataframe. And the names are formatted as `make,model`
    """
    manufacturer_list = df["manufacturer"].unique()
    model_list = df["model"].unique()
    similar_cars_clean = defaultdict(set)
    print(type(similar_cars))
    for key, similar_set in similar_cars.items():
        for car in similar_set:
            manufacturer = find_manufacturer(car, manufacturer_list)
            if manufacturer:
                model = find_model(car, model_list)
                if model:
                    similar_cars_clean[key.replace(",", "_")].add(
                        manufacturer + "_" + model
                    )
    return dict(similar_cars_clean)
