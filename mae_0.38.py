#%%
from utils import *
from data_processing import Data
from utils.metrics import regression_report

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, classification_report

#%%
def evaluate(reg, X_df, label_df):
    pred_df = data.predict(reg, X_df)

    label_true_pred = []
    for date, row in pred_df.iterrows():
        label_true = label_df.loc[date, "label"]
        label_pred = row["pred_label"]
        label_true_pred.append((label_true, label_pred))

    label_true = [true for true, pred in label_true_pred]
    label_pred = [pred for true, pred in label_true_pred]
    print(f"MAE: {mean_absolute_error(label_true, label_pred)}")
    print(classification_report(label_true, label_pred))
    # Visualization(
    #     label_true, label_pred
    # ).classification_report().confusion_matrix().show()


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
    ["revenue"], test_ratio=0.3
)
X_train, X_test, y_train, y_test = (
    X_train_df.to_numpy(),
    X_test_df.to_numpy(),
    y_train_df["revenue"].to_numpy(),
    y_test_df["revenue"].to_numpy(),
)
print(f"X_train shape {X_train.shape}, y_train shape {y_train.shape}")
print(f"X_test shape {X_test.shape}, y_test shape {y_test.shape}")

#%% evaluate performance with training data
eval_reg = HistGradientBoostingRegressor(random_state=1129)
eval_reg.fit(X_train, y_train)
report = regression_report(y_test, eval_reg.predict(X_test), X_test.shape[1])
print("-" * 10, "regression report", "-" * 10)
print(report)

label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
print("-" * 10, "evaluation", "-" * 10)
evaluate(eval_reg, X_test_df, label_df)

#%% training with all data
X_df, y_df = data.processing(["revenue"])
reg = HistGradientBoostingRegressor(random_state=1129)
reg.fit(X_df.to_numpy(), y_df["revenue"].to_numpy())


#%% fill predict label to csv
test_X_df = data.processing_test_data("data/test.csv")
predict_df = data.predict(reg, test_X_df)
fill_label(predict_df, "data/test_nolabel.csv")
