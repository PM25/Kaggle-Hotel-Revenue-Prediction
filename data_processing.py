#%%
from utils import *

from random import shuffle, seed
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


# target is one of the ["is_canceled", "reservation_status", "adr"]
class Data:
    def __init__(self, fname="data/train.csv", use_dummies=False, normalize=False):
        self.y_cats = None
        self.scalers = None
        self.label_encoders = None
        self.onehot_encoders = None
        self.normalize = normalize
        self.use_dummies = use_dummies
        self.fname = fname
        # dataframes
        self.train_df = pd.read_csv(fname, index_col="ID")
        self.label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
        self.clean_train_df = self.preprocessing(self.train_df, log=True)
        self.processed_df = self.processing(log=True)
        print(f"Shape of Read Data: {self.train_df.shape}")

    def __call__(self, fname):
        self.train_df = pd.read_csv(fname, index_col="ID")
        self.clean_train_df = self.preprocessing(self.train_df)

    def preprocessing(self, df, log=False):
        df = self.add_features(df, log=log)
        # self.processed_df = self.add_sklearn_prediction(self.processed_df)
        return df

    def processing_test_data(self, fname="data/test.csv", log=False):
        if log:
            print(f"-" * 15)
            print(f"Processing file: {fname}")
        test_df = pd.read_csv(fname, index_col="ID")
        X_test_df = self.preprocessing(test_df, log=log)
        X_test_df = self.postprocessing(X_test_df, log=log)
        return X_test_df

    def get_y_cats(self):
        return self.y_cats

    def add_features(self, df, log=False):
        added_columns = []

        if {"adr", "is_canceled"}.issubset(df.columns):
            df["revenue"] = (
                df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
            ) * df["adr"]
            df.loc[df["is_canceled"] == 1, "revenue"] = 0
            added_columns.append("revenue")

        df["net_canceled"] = 0
        df.loc[
            df["previous_cancellations"] > df["previous_bookings_not_canceled"],
            "net_canceled",
        ] = 1
        added_columns.append("net_canceled")

        if log:
            print(f"New added columns: {added_columns}")

        return df

    def postprocessing(self, df, log=False):
        exclude_columns = [
            "is_canceled",
            "adr",
            "reservation_status",
            "reservation_status_date",
            "revenue",
        ]
        df = df.drop(exclude_columns, axis=1, errors="ignore")

        df.arrival_date_month = df.arrival_date_month.map(MONTHS)
        df.children = df.children.fillna(0)
        nan_cols = get_columns_with_nan(df)
        if log:
            print(f"Columns that contain NaN (before):\n {nan_cols}")

        for col in nan_cols:
            df[col] = df[col].fillna("Null").astype(str)

        df = self.label_encoder(df)
        if self.normalize:
            df = self.use_scaler(df)
        if self.use_dummies:
            df = self.onehot_encoder(df)

        if log:
            print(f"Columns that contain NaN (after):\n {get_columns_with_nan(df)}")
            print(f"Excluded columns: {exclude_columns}")

        return df

    def processing(self, target="is_canceled", dropout=[], log=False):
        if log:
            print(f"-" * 15)
            print(f"Processing file: {self.fname}")
        clean_df = self.clean_train_df.copy()

        if is_numeric_dtype(clean_df[target]):
            y_df = clean_df[target]
        else:
            y_df = clean_df[target].astype("category")
            self.y_cats = y_df.cat.categories
            y_df = y_df.cat.codes  # convert categories data to numeric codes

        clean_df = clean_df.drop(dropout, axis=1, errors="ignore")
        X_df = self.postprocessing(clean_df, log=log)

        return (X_df, y_df)

    def add_sklearn_prediction(self, df, test=False):
        processed_df = df.copy()
        out_processed_df = processed_df.copy()

        # add predicted is_canceled column
        X_train, y_train = self.processing("is_canceled")

        for clf in [AdaBoostClassifier, RandomForestClassifier]:
            X, y = X_train.copy(), y_train.copy()

            eval_clf = clf()
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
            eval_clf.fit(train_X, train_y)
            print("clf score:", eval_clf.score(test_X, test_y))

            c = clf()
            c.fit(X, y)
            pred = c.predict(self.postprocessing(processed_df, test=test))
            out_processed_df = pd.concat([out_processed_df, pd.DataFrame(pred)], axis=1)

        # add predicted adr column
        X_train, y_train = self.processing("adr")

        for reg in [BaggingRegressor, RandomForestRegressor]:
            X, y = X_train.copy(), y_train.copy()

            eval_reg = reg()
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
            eval_reg.fit(train_X, train_y)
            print("reg score:", eval_reg.score(test_X, test_y))

            r = reg()
            r.fit(X, y)
            pred = r.predict(self.postprocessing(processed_df, test=test))
            out_processed_df = pd.concat([out_processed_df, pd.DataFrame(pred)], axis=1)

        return out_processed_df

    def onehot_encoder(self, df, refit=False):
        if self.onehot_encoders != None and refit == False:
            dummies_df = df.copy()
            for cname, encoder in self.onehot_encoders.items():
                transformed_col = encoder.transform(df[[cname]])
                transformed_cols_df = pd.DataFrame(
                    transformed_col, columns=encoder.get_feature_names()
                )
                dummies_df = pd.concat([dummies_df, transformed_cols_df], axis=1).drop(
                    [cname], axis=1
                )
            df = dummies_df
        else:
            encoders = {}
            for cname in df.columns:
                if is_string_dtype(df[cname]):
                    encoders[cname] = OneHotEncoder(
                        handle_unknown="ignore", sparse=False
                    )
                    transformed_col = encoders[cname].fit_transform(df[[cname]])
                    transformed_cols_df = pd.DataFrame(
                        transformed_col,
                        columns=encoders[cname].get_feature_names([cname]),
                    )
                    df = pd.concat([df, transformed_cols_df], axis=1).drop(
                        [cname], axis=1
                    )
            self.onehot_encoders = encoders
        return df

    def label_encoder(self, df, refit=False):
        if self.label_encoders != None and refit == False:
            for cname, encoder in self.label_encoders.items():
                encoder_dict = dict(
                    zip(encoder.classes_, encoder.transform(encoder.classes_))
                )
                # handling previous unseen label (assign -1)
                df[cname] = (
                    df[cname].apply(lambda x: encoder_dict.get(x, -1)).astype(int)
                )
        else:
            encoders = {}
            for cname in df.columns:
                if is_string_dtype(df[cname]):
                    encoders[cname] = LabelEncoder()
                    df[cname] = encoders[cname].fit_transform(df[cname])
                    df[cname] = df[cname].astype("category")
            self.label_encoders = encoders
        return df

    def use_scaler(self, df, refit=False):
        if self.scalers != None and refit == False:
            for cname, scaler in self.scalers.items():
                df[cname] = scaler.transform(df[[cname]])
        else:
            scalers = {}
            for cname in df.columns:
                if is_numeric_dtype(df[cname]):
                    scalers[cname] = MinMaxScaler(feature_range=(0, 1))
                    df[cname] = scalers[cname].fit_transform(df[[cname]])
            self.scalers = scalers

        return df

    def train_test_split_by_date(self, target="revenue", test_ratio=0.25, random=True):
        seed(1129)
        X_df, y_df = self.processing(target)
        processed_df = pd.concat([X_df, y_df], axis=1)

        date_df = processed_df.groupby(
            ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
        )

        keys = [key for key in date_df.groups.keys()]
        # shuffle(keys)
        train_amount = int(len(keys) * (1 - test_ratio))
        train_keys = keys[:train_amount]
        test_keys = keys[train_amount:]

        train_idxs = []
        for key in train_keys:
            train_idxs += date_df.groups[key].tolist()
        test_idxs = []
        for key in test_keys:
            test_idxs += date_df.groups[key].tolist()

        if random:
            shuffle(train_idxs)
            shuffle(test_idxs)

        train_df = processed_df.loc[train_idxs, :]
        y_train_df = train_df[target]
        X_train_df = train_df.drop([target], axis=1)
        test_df = processed_df.loc[test_idxs, :]
        y_test_df = test_df[target]
        X_test_df = test_df.drop([target], axis=1)

        return X_train_df, X_test_df, y_train_df, y_test_df

    # target = label or revenue
    def predict_clean(self, reg, df, target="label"):
        df = self.predict(reg, df)
        target_col = "pred_label" if target == "label" else "pred_revenue_per_day"
        predict_df = (
            df[["arrival_date", target_col]]
            .reset_index(drop=True)
            .set_index("arrival_date")
        )
        return predict_df

    def predict(self, reg, df):
        df["pred_revenue"] = reg.predict(df.to_numpy())

        df_per_day = (
            df.groupby(
                ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
            )
            .sum()
            .reset_index()
            .rename(columns={"pred_revenue": "pred_revenue_per_day"})
        )
        df_per_day["pred_label"] = df_per_day["pred_revenue_per_day"] // 10000
        df_per_day["arrival_date"] = df_per_day.apply(
            lambda x: f"{int(x['arrival_date_year'])}-{int(x['arrival_date_month']):02d}-{int(x['arrival_date_day_of_month']):02d}",
            axis=1,
        )
        return df_per_day


#%%
if __name__ == "__main__":
    data = Data(use_dummies=True, normalize=True)
    X_df = data.processing_test_data()
# %%
