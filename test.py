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


def get_predict(pred, fname="data/test.csv"):
    test_df = pd.read_csv(fname)
    revenue = pd.DataFrame(pred, columns=["pred_revenue"])
    results = pd.concat([test_df, revenue], axis=1)
    results.to_csv("test_results.csv")

    groupby_date = results.groupby(
        ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
    )

    labels = []
    day_labels = {}
    for day, daily_revenue in groupby_date["pred_revenue"]:
        label = daily_revenue.sum() // 10000
        day_str = f"{int(day[0])}-{MONTHS[day[1]]:02d}-{int(day[2]):02d}"
        labels.append(label)
        day_labels[day_str] = label

    return labels


# test classifiers
data = Data(use_dummies=False)
X_np, y_np = data.processing("revenue", normalize=False)
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

# training with all data
# reg = DecisionTreeRegressor()
reg = RandomForestRegressor(verbose=True)
reg.fit(X_np, y_np)
print(X_np.shape)

#%%
test_X_np = data.processing_test_data()
pred_revenue = reg.predict(test_X_np)

labels_pred = get_predict(pred_revenue, "data/train.csv")
labels_true = pd.read_csv("data/train_label.csv", index_col="arrival_date").to_numpy()
print(mean_absolute_error(labels_pred, labels_true))

#%%
# test_nolabel_df = pd.read_csv("data/test_nolabel.csv", index_col="arrival_date")
# test_nolabel_df["label"] = 0
# for day, daily_revenue in groupby_date["pred_revenue"]:
#     label = daily_revenue.sum() // 10000
#     day_str = f"{int(day[0])}-{MONTHS[day[1]]:02d}-{int(day[2]):02d}"
#     # test_nolabel_df.loc[day_str, "label"] = label
#     test_nolabel_df["label"][day_str] = label

# test_nolabel_df.to_csv("test_label.csv")

