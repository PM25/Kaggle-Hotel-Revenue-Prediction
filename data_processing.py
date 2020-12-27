#%%
from utils import *

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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
    def __init__(self, fname="data/train.csv", use_dummies=False):
        self.train_df = pd.read_csv(fname, index_col="ID")
        self.label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
        self.processed_df = self.add_features(self.train_df)
        # self.processed_df = self.add_sklearn_prediction(self.processed_df)
        print(f"Shape of Read Data: {self.train_df.shape}")
        self.scaler = None
        self.y_cats = None
        self.adr_regs = []
        self.is_canceled_clf = []
        self.column_transformer = None
        self.use_dummies = use_dummies
        self.label_encoders = None
        self.onehot_encoders = None

    def __call__(self, fname):
        self.train_df = pd.read_csv(fname, index_col="ID")
        self.processed_df = self.add_features(self.train_df)
        # self.processed_df = self.add_sklearn_prediction(self.processed_df)

    def processing_test_data(self, fname="data/test.csv"):
        test_df = pd.read_csv(fname, index_col="ID")
        processed_df = self.add_features(test_df, test=True)
        # processed_df = self.add_sklearn_prediction(processed_df, test=True)
        X = self.postprocessing(self.processed_df)
        return X

    def get_y_cats(self):
        return self.y_cats

    def add_features(self, train_df, test=False):
        if not test:
            train_df["revenue"] = (
                train_df["stays_in_weekend_nights"] + train_df["stays_in_week_nights"]
            ) * train_df["adr"]

            train_df.loc[train_df["is_canceled"] == 1, "revenue"] = 0

        train_df["net_cancelled"] = 0
        train_df.loc[
            train_df["previous_cancellations"]
            > train_df["previous_bookings_not_canceled"],
            "net_cancelled",
        ] = 1

        return train_df

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
        nan_cols = list(get_columns_with_nan(df))
        if log:
            print(f"Columns that contain NaN: {nan_cols}")

        for col in nan_cols:
            df[col] = df[col].fillna("Null").astype(str)

        df = self.label_encoder(df)
        if self.use_dummies:
            np = self.onehot_encoder(df)
        else:
            np = df.to_numpy()

        if log:
            print(f"Columns that contain NaN: {list(get_columns_with_nan(df))}")
            print(f"Excluded columns: {exclude_columns}")

        return np

    def processing(self, target="is_canceled", dropout=[], normalize=False):
        processed_df = self.processed_df.copy()

        if is_numeric_dtype(processed_df[target]):
            y_df = processed_df[target]
        else:
            y_df = processed_df[target].astype("category")
            self.y_cats = y_df.cat.categories
            y_df = y_df.cat.codes  # convert categories data to numeric codes

        processed_df = processed_df.drop(dropout, axis=1, errors="ignore")
        X_np = self.postprocessing(processed_df, log=True)

        y_np = y_df.to_numpy()

        # TODO: make this function into class and store scaler for new data to use
        if normalize:
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(X_np)
            X_np = self.scaler.transform(X_np)

        return (X_np, y_np)

    def processing_revenue(self, use_dummies=False, normalize=True):
        train_df = self.processed_df.copy()

        revenue_df = (
            train_df["stays_in_weekend_nights"] + train_df["stays_in_week_nights"]
        ) * train_df["adr"]

        X_df = train_df.drop(
            ["is_canceled", "adr", "reservation_status", "reservation_status_date",],
            axis=1,
        )

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

        print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

        X_np = X_df.to_numpy()
        y_np = revenue_df.to_numpy()

        # TODO: make this function into class and store scaler for new data to use
        if normalize:
            if self.scaler is None:
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(X_np)
            X_np = self.scaler.transform(X_np)

        return (X_np, y_np)

    def processing_cnn(self):
        train_df, _ = self.processing(None, use_dummies=False, test=True)
        train_df_label = pd.read_csv("data/train_label.csv", index_col="arrival_date")

        groupby_date = train_df.groupby(
            ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
        )

        max_booking_a_day = max(
            [len(value) for group, value in groupby_date.groups.items()]
        )

        X, y = [], []
        for group, data in groupby_date.groups.items():
            padding_sz = max_booking_a_day - len(data)
            nfeatures = groupby_date.get_group(group).shape[1]
            columns = groupby_date.get_group(group).columns
            padding = pd.DataFrame(np.zeros((padding_sz, nfeatures)), columns=columns)
            padding -= 1
            processed_data = pd.concat([groupby_date.get_group(group), padding], axis=0)
            date_str = f"{group[0]}-{group[1]:02d}-{group[2]:02d}"
            label = train_df_label["label"][date_str]
            X.append(processed_data.to_numpy())
            y.append(label)

        return (np.expand_dims(np.array(X), axis=1), np.array(y))

    def processing_1d_cnn(self, previous=5):
        train_df_X, train_df_label = self.processing_revenue(use_dummies=False)

        X_list, y_list = [], []
        for i in range(previous - 1, train_df_X.shape[0]):
            X = train_df_X[i - previous + 1 : i + 1]
            y = train_df_label[i]
            X_list.append(X)
            y_list.append(y)

        return (np.array(X_list), np.array(y_list))

    def add_sklearn_prediction(self, df, normalize=False, test=False):
        processed_df = df.copy()
        out_processed_df = processed_df.copy()

        # add predicted is_canceled column
        X_train, y_train = self.processing("is_canceled", normalize=normalize)

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
        X_train, y_train = self.processing("adr", normalize=normalize)

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
        cat_cols_idx = []
        for idx, cname in enumerate(df.columns):
            if is_string_dtype(df[cname]):
                cat_cols_idx.append(idx)
        print(f"string columns: {[df.columns[idx] for idx in cat_cols_idx]}")

        if refit == False and self.column_transformer != None:
            np = self.column_transformer.transform(df).toarray()
        else:
            self.column_transformer = ColumnTransformer(
                [("encoder", OneHotEncoder(handle_unknown="ignore"), cat_cols_idx)],
                remainder="passthrough",
            )
            np = self.column_transformer.fit_transform(df).toarray()
        return np

    def label_encoder(self, df, refit=False):
        if self.label_encoders != None and refit == False:
            for cname, encoder in self.label_encoders.items():
                df[cname] = encoder.transform(df[cname])
        else:
            encoders = {}
            for cname in df.columns:
                if is_string_dtype(df[cname]):
                    encoders[cname] = LabelEncoder()
                    df[cname] = encoders[cname].fit_transform(df[cname])
                    df[cname] = df[cname].astype("category")
            self.label_encoders = encoders

        return df


#%%
if __name__ == "__main__":
    data = Data(use_dummies=True)
    # X, y = data.processing("adr", use_dummies=False)
    # X, y = data.processing("revenue")
    X1, y1 = data.processing("adr")
    X2 = data.processing_test_data("data/train.csv")
    print(np.array_equal(X1, X2))

# %%
