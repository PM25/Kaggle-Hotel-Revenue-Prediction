import numpy as np
import pandas as pd


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


def processing_data(binary=False):
    train_df = pd.read_csv("data/train.csv")
    # train_df = pd.read_csv("data/hotel_bookings.csv")

    train_df["expected_room"] = 0
    train_df.loc[
        train_df["reserved_room_type"] == train_df["assigned_room_type"],
        "expected_room",
    ] = 1
    train_df["net_cancelled"] = 0
    train_df.loc[
        train_df["previous_cancellations"] > train_df["previous_bookings_not_canceled"],
        "net_cancelled",
    ] = 1

    exclude_columns = [
        "is_canceled",
        "ID",
        "adr",
        "reservation_status",
        "reservation_status_date",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "arrival_date_month",
        "assigned_room_type",
        "reserved_room_type",
        "previous_cancellations",
        "previous_bookings_not_canceled",
    ]

    if binary:
        y_df = train_df["is_canceled"].astype("category")
    else:
        y_df = train_df["reservation_status"].astype("category")
        reservation_status_cats = y_df.cat.categories
        y_df = y_df.cat.codes  # convert categories data to numeric codes

    X_df = train_df.drop(exclude_columns, axis=1)
    X_df.children = X_df.children.fillna(0)
    nan_cols = list(get_columns_with_nan(X_df))
    print(f"Columns that contain NaN: {nan_cols}")

    for col in nan_cols:
        X_df[col] = X_df[col].fillna("Null").astype(str)

    for col in X_df.select_dtypes(include=["object"]).columns:
        X_df[col] = X_df[col].factorize()[0]
    # X_df = pd.get_dummies(X_df)
    print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()
    if binary:
        return (X_np, y_np)
    else:
        return (X_np, y_np), reservation_status_cats
