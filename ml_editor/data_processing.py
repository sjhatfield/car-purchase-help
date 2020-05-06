from sklearn.model_selection import train_test_split, GroupShuffleSplit
import pandas as pd
from ml_editor import constants


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
    df = df[(df["price"] > 0) & (df["price"] < 100000)]
    df = df[(df["year"] > 1980) & (df["year"] < 2020)]
    df = df[(df["odometer"] > 1000) & (df["odometer"] < 300000)]

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
