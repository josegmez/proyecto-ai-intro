from sklearn.impute import SimpleImputer


def cleaning_data(df):
    """
    This function cleans the data by filling the missing values and removing the outliers.

    Args:
    df : DataFrame : The dataset to be cleaned.
    """

    fuel_type_imputer = SimpleImputer(strategy="most_frequent")
    df["fuel_type"] = fuel_type_imputer.fit_transform(df[["fuel_type"]]).ravel()

    missing_label_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    df[["accident", "clean_title"]] = missing_label_imputer.fit_transform(
        df[["accident", "clean_title"]]
    )

    return df
