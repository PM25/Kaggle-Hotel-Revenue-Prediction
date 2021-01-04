#%%
from utils import *

import datetime
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
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

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


def evaluate_by_label(pred_label_df, true_label_df, target="label"):
    true_preds = []
    for date, row in pred_label_df.iterrows():
        if target == "label":
            true = true_label_df.loc[date, "label"]
            pred = row["pred_label"]
        else:
            true = true_label_df.loc[date, "revenue"]
            pred = row["pred_revenue"]
        true_preds.append((true, pred))

    true = [true for true, pred in true_preds]
    pred = [pred for true, pred in true_preds]
    report = []
    report.append(f"MAE: {mean_absolute_error(true, pred)}")
    if target == "label":
        report.append(classification_report(true, pred))
        Visualization(true, pred).classification_report().confusion_matrix().show()
    return "\n".join(report)


#%% fill label
def fill_label(predict_df, fname="data/test_nolabel.csv"):
    label_df = pd.read_csv(fname, index_col="arrival_date")

    label_df["label"] = 0
    for idx, subdf in predict_df.iterrows():
        label_df.loc[idx, "label"] = subdf["pred_label"]

    label_df.to_csv("label_pred.csv")


# target is one of the ["is_canceled", "reservation_status", "adr"]
class Data:
    def __init__(self, fname="data/train.csv", use_dummies=False, normalize=False):
        self.y_cats = {}
        self.scalers = None
        self.label_encoders = None
        self.onehot_encoders = None
        self.normalize = normalize
        self.use_dummies = use_dummies
        self.fname = fname
        # dataframes
        self.train_df = pd.read_csv(fname, index_col="ID")
        print(f"Shape of Read Data: {self.train_df.shape}")
        self.label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
        self.clean_train_df = self.preprocessing(self.train_df, log=True)
        self.processed_df = self.processing(log=True)

    def __call__(self, fname):
        self.train_df = pd.read_csv(fname, index_col="ID")
        self.clean_train_df = self.preprocessing(self.train_df)

    def preprocessing(self, df, log=False):
        df = df.copy()
        df.arrival_date_month = df.arrival_date_month.map(MONTHS)
        df = self.add_features(df, log=log)
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

    def add_features(self, df, features=[], log=False):
        df = df.copy()
        added_columns = []

        if {"adr", "is_canceled"}.issubset(df.columns):
            # revenue
            df["revenue"] = (
                df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
            ) * df["adr"]
            df.loc[df["is_canceled"] == 1, "revenue"] = 0
            added_columns.append("revenue")

            # actual adr (is_canceled adr = 0)
            df["actual_adr"] = df["adr"]
            df.loc[df["is_canceled"] == 1, "actual_adr"] = 0
            added_columns.append("actual_adr")

        # net canceled
        df["net_canceled"] = 0
        df.loc[
            df["previous_cancellations"] > df["previous_bookings_not_canceled"],
            "net_canceled",
        ] = 1
        added_columns.append("net_canceled")

        if "orders_in_the_same_day" in features:
            df = self.add_orders_in_same_day(df)
            added_columns.append("orders_in_the_same_day")

        if log:
            print(f"New added columns: {added_columns}")

        return df

    def postprocessing(self, df, log=False):
        df = df.copy()
        exclude_columns = [
            "is_canceled",
            "adr",
            "reservation_status",
            "reservation_status_date",
            "revenue",
            "actual_adr",
        ]
        df = df.drop(exclude_columns, axis=1, errors="ignore")

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
            print(f"Processed DataFrame shape: {df.shape}")

        return df

    def processing(self, targets=["is_canceled"], dropout=[], log=False):
        if log:
            print(f"-" * 15)
            print(f"Processing file: {self.fname}")
        clean_df = self.clean_train_df.copy()

        y_dfs = []
        for target in targets:
            if is_numeric_dtype(clean_df[target]):
                y_df = clean_df[target]
            else:
                y_df = clean_df[target].astype("category")
                self.y_cats[target] = y_df.cat.categories
                y_df = y_df.cat.codes  # convert categories data to numeric codes
            y_dfs.append(y_df)
        y_df = pd.concat(y_dfs, axis=1)

        clean_df = clean_df.drop(dropout, axis=1, errors="ignore")
        X_df = self.postprocessing(clean_df, log=log)

        return (X_df, y_df)

    def add_sklearn_prediction(self, df, test=False):
        processed_df = df.copy()
        out_processed_df = processed_df.copy()

        # add predicted is_canceled column
        X_train, y_train = self.processing(["is_canceled"])

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
        X_train, y_train = self.processing(["adr"])

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

    def train_test_split_by_date(
        self, targets=["revenue"], test_ratio=0.25, random=True
    ):
        seed(1129)
        X_df, y_df = self.processing(targets)
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
        y_train_df = train_df[targets]
        X_train_df = train_df.drop(targets, axis=1)
        test_df = processed_df.loc[test_idxs, :]
        y_test_df = test_df[targets]
        X_test_df = test_df.drop(targets, axis=1)

        return X_train_df, X_test_df, y_train_df, y_test_df

    def predict_label(
        self, reg, df, reg_out="revenue", columns=["pred_label", "pred_revenue"]
    ):
        df = df.copy()
        if reg_out == "revenue":
            df["pred_revenue"] = reg.predict(df.to_numpy())
        elif reg_out == "adr":
            df["pred_adr"] = reg.predict(df.to_numpy())
            df["pred_revenue"] = (
                df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
            ) * df["pred_adr"]

        df = self.to_label(df, columns=columns)
        return df

    def to_label(self, df, columns=["pred_revenue", "pred_label"]):
        df = df.copy()
        df["orders"] = 1
        # df.loc[df["is_canceled"] == 1, "orders"] = 0
        df = (
            df.groupby(
                ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
            )
            .sum()
            .reset_index()
            .drop(["orders_in_the_same_day"], axis=1, errors="ignore")
            .rename(columns={"orders": "orders_in_the_same_day"})
        )
        if "pred_revenue" in df.columns:
            df["pred_label"] = df["pred_revenue"] // 10000
        if "revenue" in df.columns:
            df["label"] = df["revenue"] // 10000
            df["avg_revenue"] = df["revenue"] / df["orders_in_the_same_day"]
        if "adr" in df.columns:
            df["avg_adr"] = df["adr"] / df["orders_in_the_same_day"]

        df["arrival_date"] = df.apply(
            lambda x: f"{int(x['arrival_date_year'])}-{int(x['arrival_date_month']):02d}-{int(x['arrival_date_day_of_month']):02d}",
            axis=1,
        )

        df = (
            df[["arrival_date"] + columns]
            .reset_index(drop=True)
            .set_index("arrival_date")
        )
        return df

    # only work when setting normalize & use_dummies to False
    def get_true_label(self, columns=["adr", "label", "revenue"]):
        X_df, y_df = self.processing(["adr", "is_canceled", "revenue", "actual_adr"])
        df = pd.concat([X_df, y_df], axis=1)
        df = self.to_label(df, columns=columns)
        return df

    def add_orders_in_same_day(self, df):
        df = df.copy()
        group_by_date_df = df.groupby(
            ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
        )
        for date, orders in group_by_date_df:
            df.loc[orders.index, "orders_in_the_same_day",] = orders.shape[0]
        return df

    def create_data(
        self, start_date=(2017, 3, 14), end_date=(2017, 9, 1), ratio=0.5, offset=20
    ):
        X_df, y_df = self.processing(["adr", "is_canceled", "revenue", "actual_adr"])
        df = pd.concat([X_df, y_df], axis=1).copy()

        if is_string_dtype(df.arrival_date_month):
            df.arrival_date_month = df.arrival_date_month.map(MONTHS)
        start_date = datetime.date(start_date[0], start_date[1], start_date[2])
        end_date = datetime.date(end_date[0], end_date[1], end_date[2])
        day_delta = datetime.timedelta(days=1)

        new_dfs = []
        while start_date <= end_date:
            year = start_date.year
            passed_date = start_date.replace(year=year - 1)

            passed_data = df.loc[
                (df["arrival_date_year"] == passed_date.year)
                & (df["arrival_date_month"] == passed_date.month)
                & (df["arrival_date_day_of_month"] == passed_date.day)
            ]
            if passed_data.shape[0] > 1:
                passed_data.loc[:, "adr"] += offset
                passed_data.loc[:, "arrival_date_year"] = year
                new_dfs.append(passed_data)

            start_date += day_delta

        new_df = pd.concat(new_dfs, axis=0)
        # update revenue & actual_adr
        new_df = self.add_features(new_df)

        if ratio < 1:
            np.random.seed(1126)
            drop_amount = int(new_df.shape[0] * (1 - ratio))
            drop_indices = np.random.choice(new_df.index, drop_amount, replace=False)
            new_df = new_df.drop(drop_indices)

        return new_df


#%%
if __name__ == "__main__":
    data = Data(use_dummies=False, normalize=False)
    # x_df, y_df = data.processing(["adr"])
    # new_x_df = data.add_order_in_same_day(x_df)
    train_df = data.train_df
    new_df = data.create_data(train_df, ratio=0.3)
    new_data = pd.concat([train_df, new_df], axis=0)
#%%

