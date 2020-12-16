import numpy as np
import pandas as pd


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


def processing_data_reservation_status():
    train_X = pd.read_csv("data/train.csv")

    exclude_columns = [
        "is_canceled",
        "ID",
        "adr",
        "reservation_status",
        "reservation_status_date",
    ]

    y_df = train_X["reservation_status"].astype("category")
    reservation_status_cats = y_df.cat.categories
    y_df = y_df.cat.codes  # convert categories data to numeric codes

    X_df = train_X.iloc[:, ~train_X.columns.isin(exclude_columns)]
    # X_df.loc[:, "children"] = X_df["children"].fillna(0)
    X_df.children = X_df.children.fillna(0)
    nan_cols = list(get_columns_with_nan(X_df))
    print(f"Columns that contain NaN: {nan_cols}")
    
    for col in nan_cols:
        X_df[col] = X_df[col].fillna("Null").astype(str)

    # for col in X_df.select_dtypes(include=["object"]).columns:
    #     X_df[col] = X_df[col].factorize()[0]

    X_df = pd.get_dummies(X_df)
    print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()
    return (X_np, y_np), reservation_status_cats


def processing_data_is_canceled():
    train_X = pd.read_csv("data/train.csv")

    exclude_columns = [
        "is_canceled",
        "ID",
        "adr",
        "reservation_status",
        "reservation_status_date",
    ]

    y_df = train_X["is_canceled"].astype("category")

    X_df = train_X.iloc[:, ~train_X.columns.isin(exclude_columns)]
    X_df.loc[:, "children"] = X_df["children"].fillna(0)
    nan_cols = list(get_columns_with_nan(X_df))
    print(f"Columns that contain NaN: {nan_cols}")
    
    for col in nan_cols:
        X_df[col] = X_df[col].fillna("Null").astype(str)

    X_df = pd.get_dummies(X_df)
    print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()
    return (X_np, y_np)
