#%%
import numpy as np
import pandas as pd

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

    # for col in X_df.select_dtypes(include=["object"]).columns:
    #     X_df[col] = X_df[col].factorize()[0]
    X_df = pd.get_dummies(X_df)
    print(f"Columns that contain NaN: {list(get_columns_with_nan(X_df))}")

    return (X_df, y_df), None
    X_np = X_df.to_numpy()
    y_np = y_df.to_numpy()
    if binary:
        return (X_np, y_np)
    else:
        return (X_np, y_np), reservation_status_cats

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
        train_df_X, train_df_label = self.processing2(use_dummies=False)

        X_list, y_list = [], []
        for i in range(previous - 1, train_df_X.shape[0]):
            X = train_df_X[i - previous + 1 : i + 1]
            y = train_df_label[i]
            X_list.append(X)
            y_list.append(y)

        return (np.array(X_list), np.array(y_list))


#%%


if __name__ == "__main__":
    (x, y), _ = processing_data()
    x.to_csv("train_x.csv", index=False)
# %%
