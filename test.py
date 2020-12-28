#%%
from utils import *
from data_processing import Data
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingRegressor,
    BaggingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
from utils.metrics import regression_report

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


# def evaluation(pred, fname="data/test.csv"):
#     test_df = pd.read_csv(fname)
#     revenue = pd.DataFrame(pred, columns=["pred_revenue"])
#     results = pd.merge([test_df, revenue], axis=1)
#     results.to_csv("test_results.csv")

#     groupby_date = results.groupby(
#         ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
#     )

#     labels = []
#     day_labels = {}
#     for day, daily_revenue in groupby_date["pred_revenue"]:
#         label = daily_revenue.sum() // 10000
#         day_str = f"{int(day[0])}-{MONTHS[day[1]]:02d}-{int(day[2]):02d}"
#         labels.append(label)
#         day_labels[day_str] = label

#     print(mean_absolute_error(labels_pred, labels_true))

#     return labels


# test classifiers
data = Data(use_dummies=False)
X_df, y_df = data.processing("revenue", normalize=False)
X_df, y_df = data.processing("revenue", normalize=False)
X_np, y_np = X_df.to_numpy(), y_df.to_numpy()
print(X_np.shape)
print(y_np.shape)


eval_reg = RandomForestRegressor(verbose=True)
# eval_reg = DecisionTreeRegressor()
train_X, test_X, train_y, test_y = train_test_split(
    X_np, y_np, test_size=0.25, random_state=1129
)
eval_reg.fit(train_X, train_y)
print("reg r2", eval_reg.score(test_X, test_y))
print(regression_report(eval_reg.predict(test_X), test_y, train_X.shape[1]))


# # training with all data
# reg = DecisionTreeRegressor()
reg = RandomForestRegressor(verbose=True)
reg.fit(X_np, y_np)
print(X_np.shape)

#%%
def predict(reg, df):
    np = df.to_numpy()
    pred_revenue = reg.predict(np)
    df["pred_revenue"] = pred_revenue

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


# test_X_df = data.processing_test_data("data/train.csv")
test_X_df = data.processing_test_data("data/test.csv")
predict_df = predict(reg, test_X_df)

#%%
predict_df = predict_df[["arrival_date", "pred_label"]]
predict_df.reset_index(drop=True).set_index("arrival_date")

# label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
label_df = pd.read_csv("data/test_nolabel.csv", index_col="arrival_date")
# y_true = label_df["label"].to_numpy()
#%%
# result = pd.merge(label_df, predict_df, on="arrival_date")
# print(result)
label_df["label"] = 0
for _, subdf in predict_df.iterrows():
    idx = subdf["arrival_date"]
    label_df.loc[idx, "label"] = subdf["pred_label"]

label_df.to_csv("result.csv")
# y_pred = label_df["label"].to_numpy()
# print(mean_absolute_error(y_pred, y_true))
# %%
