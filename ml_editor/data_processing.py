

def format_raw_df(df):
    """
    Cleanup data
    :param df: raw DataFrame
    :return: processed DataFrame
    """
    # Dropping the county column as it is completely missing
    df = df.drop(['county'])

    # Setting types and fill NAs
    df['id'] = df['id'].astype(int)
    df['year'].fillna(-1, inplace=True)
    df['year'] = df['year'].astype(int)
    df['odometer'].fillna(-1, inplace=True)
    df['odometer'] = df['odometer'].astype(int)

    # Use the provided ID as the index
    df.set_index('id', inplace=True, drop=False)
    
    # Limiting the range of the numerical columns to remove clear outliers
    df = df[0 < df['price'] < 100000]
    df = df[1980 < df['year'] < 2020]
    df = df[10000 < df['odometer'] < 300000]

    return df