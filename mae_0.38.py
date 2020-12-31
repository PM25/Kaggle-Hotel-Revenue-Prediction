#%%
from utils import *
from data_processing import Data
from utils.metrics import regression_report

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, classification_report

#%%
def evaluate_by_label(pred_label_df, true_label_df, target="label"):
    true_pred_labels = []
    for date, row in pred_label_df.iterrows():
        if target == "label":
            label_true = true_label_df.loc[date, "label"]
            label_pred = row["pred_label"]
        else:
            label_true = true_label_df.loc[date, "revenuePerDay"]
            label_pred = row["pred_revenue_per_day"]
        true_pred_labels.append((label_true, label_pred))

    label_true = [true for true, pred in true_pred_labels]
    label_pred = [pred for true, pred in true_pred_labels]
    report = []
    report.append(f"MAE: {mean_absolute_error(label_true, label_pred)}")
    if target == "label":
        report.append(classification_report(label_true, label_pred))
        Visualization(
            label_true, label_pred
        ).classification_report().confusion_matrix().show()
    return "\n".join(report)


#%% fill label
def fill_label(predict_df, fname="data/test_nolabel.csv"):
    label_df = pd.read_csv(fname, index_col="arrival_date")

    label_df["label"] = 0
    for idx, subdf in predict_df.iterrows():
        label_df.loc[idx, "label"] = subdf["pred_label"]

    label_df.to_csv("label_pred.csv")


# data
data = Data(use_dummies=False, normalize=False)
X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
    ["actual_adr"], test_ratio=0.3
)
X_train, X_test, y_train, y_test = (
    X_train_df.to_numpy(),
    X_test_df.to_numpy(),
    y_train_df["actual_adr"].to_numpy(),
    y_test_df["actual_adr"].to_numpy(),
)
print(f"X_train shape {X_train.shape}, y_train shape {y_train.shape}")
print(f"X_test shape {X_test.shape}, y_test shape {y_test.shape}")

#%% evaluate performance with training data
eval_reg = HistGradientBoostingRegressor(random_state=1129)
eval_reg.fit(X_train, y_train)

print("-" * 10, "regression report", "-" * 10)
report = regression_report(y_test, eval_reg.predict(X_test), X_test.shape[1])
print(report)

print("-" * 10, "evaluation of label", "-" * 10)
label_df = pd.read_csv("data/revenue_per_day.csv", index_col="arrival_date")
pred_label_df = data.predict_label(eval_reg, X_test_df, reg_out="adr")

#%%
print("[ label evaluation ]")
report_label = evaluate_by_label(pred_label_df, label_df, target="label")
print(report_label)
print("[ revenue_per_day evaluation ]")
report_revenue = evaluate_by_label(pred_label_df, label_df, target="revenue")
print(report_revenue)

#%% training with all data
X_df, y_df = data.processing(["actual_adr"])
reg = HistGradientBoostingRegressor(random_state=1129)
reg.fit(X_df.to_numpy(), y_df["actual_adr"].to_numpy())

#%% fill predict label to csv
test_X_df = data.processing_test_data("data/test.csv")
pred_label_df = data.predict_label(reg, test_X_df, reg_out="adr")
fill_label(pred_label_df, "data/test_nolabel.csv")