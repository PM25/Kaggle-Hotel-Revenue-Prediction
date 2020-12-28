#%%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor


train_df = pd.read_csv("data/train.csv", index_col="ID")
test_df = pd.read_csv("data/test.csv", index_col="ID")

revenue_df = (
    (train_df["stays_in_weekend_nights"] + train_df["stays_in_week_nights"])
    * train_df["adr"]
    * (1 - train_df["is_canceled"])
)


train_df = train_df.drop(
    [
        "is_canceled",
        "adr",
        "reservation_status",
        "reservation_status_date",
        "arrival_date_year",
        "arrival_date_day_of_month",
        "arrival_date_week_number",
    ],
    axis=1,
)

test_df = test_df.drop(
    ["arrival_date_year", "arrival_date_day_of_month", "arrival_date_week_number"],
    axis=1,
)

print("train shape", train_df.shape)
print("test shape", test_df.shape)


#%%
def get_columns_with_nan(df):
    nan_values = df.isna()
    nan_columns = nan_values.any()
    columns_with_nan = df.columns[nan_columns].tolist()
    return columns_with_nan


train_df.children = train_df.children.fillna(0)

nan_cols = list(get_columns_with_nan(train_df))
for col in nan_cols:
    train_df[col] = train_df[col].fillna("Null").astype(str)

#%%
cat_cols_idx = []
for idx, cname in enumerate(train_df.columns):
    if is_string_dtype(train_df[cname]):
        cat_cols_idx.append(idx)

print([train_df.columns[idx] for idx in cat_cols_idx])
columnTransformer = ColumnTransformer(
    [("encoder", OneHotEncoder(handle_unknown="ignore"), cat_cols_idx)],
    remainder="passthrough",
)
train_np = columnTransformer.fit_transform(train_df).toarray()
train_df = pd.DataFrame(train_np)
print("train shape", train_df.shape)


x_train, x_test, y_train, y_test = train_test_split(
    train_df.to_numpy(), revenue_df.to_numpy(), test_size=0.25, random_state=1129
)

print("X train shape", x_train.shape)
print("y train shape", y_train.shape)

#%%
reg = DecisionTreeRegressor(criterion="mae")
# reg = DecisionTreeRegressor()
reg.fit(x_train, y_train)

#%%
import math
import sklearn.metrics as metrics

# report for regression result
def regression_report(y_true, y_pred, nfeatures=None):
    r2 = metrics.r2_score(y_true, y_pred)
    max_error = metrics.max_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    report = [
        f"R2: {r2:.3f}",
        f"MSE: {mse:.3f}",
        f"RMSE: {math.sqrt(mse):.3f}",
        f"Max Error: {max_error:.3f}",
    ]
    return "\n".join(report) + "\n"


pred = reg.predict(x_test)
print(regression_report(y_test, pred))

#%%
print("training with all data")
reg = DecisionTreeRegressor(criterion="mae")
reg.fit(train_df.to_numpy(), revenue_df.to_numpy())

#%%
test_np = columnTransformer.transform(test_df).toarray()
print("test shape", test_np.shape)
pred = reg.predict(test_np)
print(pred)

test_df = pd.read_csv("data/test.csv")
pred_df = pd.DataFrame(pred, columns=["pred"])
results = pd.concat([test_df, pred_df], axis=1)
results.to_csv("test_results.csv")
