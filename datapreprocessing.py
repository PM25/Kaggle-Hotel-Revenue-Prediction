#%%
from utils import *

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


# target is one of the ["is_canceled", "reservation_status", "adr"]
class Data:
    def __init__(self, fname="data/train.csv"):
        self.train_df = pd.read_csv(fname)
        self.scaler = None
        self.y_cats = None

    def __call__(self, fname):
        self.train_df = pd.read_csv(fname)

    def get_y_cats(self):
        return self.y_cats

    def processing(self, target="is_canceled", use_dummies=True, normalize=True):
        train_df = self.train_df

        exclude_columns = [
            target,
            "is_canceled",
            "ID",
            "adr",
            "reservation_status",
            "reservation_status_date",
        ]

        if target == "is_canceled" or target == "reservation_status":
            train_df["expected_room"] = 0
            train_df.loc[
                train_df["reserved_room_type"] == train_df["assigned_room_type"],
                "expected_room",
            ] = 1
            train_df["net_cancelled"] = 0
            train_df.loc[
                train_df["previous_cancellations"]
                > train_df["previous_bookings_not_canceled"],
                "net_cancelled",
            ] = 1

            exclude_columns += [
                "arrival_date_year",
                "arrival_date_week_number",
                "arrival_date_day_of_month",
                "arrival_date_month",
                "assigned_room_type",
                "reserved_room_type",
                "previous_cancellations",
                "previous_bookings_not_canceled",
            ]
        else:
            exclude_columns += [
                "arrival_date_year",
                "previous_cancellations",
                "previous_bookings_not_canceled",
                "days_in_waiting_list",
            ]

        # TrainDataVisualization(train_df, None,).correlation_matrix().show()

        if is_numeric_dtype(train_df[target]):
            y_df = train_df[target]
        else:
            y_df = train_df[target].astype("category")
            self.y_cats = y_df.cat.categories
            y_df = y_df.cat.codes  # convert categories data to numeric codes

        X_df = train_df.drop(exclude_columns, axis=1)
        X_df.children = X_df.children.fillna(0)
        nan_cols = list(get_columns_with_nan(X_df))
        print(f"Columns that contain NaN: {nan_cols}")

        for col in nan_cols:
            X_df[col] = X_df[col].fillna("Null").astype(str)

        if use_dummies:
            X_df = pd.get_dummies(X_df)
        else:
            for col in X_df.select_dtypes(include=["object"]).columns:
                X_df[col] = X_df[col].factorize()[0]
            # TrainDataVisualization(
            #     pd.concat([X_df, y_df], axis=1), None,
            # ).correlation_matrix().show()

        print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")
        print(f"Excluded columns: {exclude_columns}")

        X_np, y_np = X_df.to_numpy(), y_df.to_numpy()

        # TODO: make this function into class and store scaler for new data to use
        if normalize:
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(X_np)
            X_np = self.scaler.transform(X_np)

        return (X_np, y_np)


if __name__ == "__main__":
    data = Data()
    X, y = data.processing("adr", use_dummies=False)

# %%
