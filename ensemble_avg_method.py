#%%
from utils import *
from data_processing import Data
from utils.metrics import regression_report

import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
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


def evaluate_by_label2(pred_label_df, true_label_df, target="label"):
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
    print(f"MAE: {mean_absolute_error(label_true, label_pred)}")
    if target == "label":
        print(classification_report(label_true, label_pred))
        Visualization(
            label_true, label_pred
        ).classification_report().confusion_matrix().show()


#%% evaluate performance with training data
def split_train(estimator_class, X_df, y_df, nsplit=2):
    nrow = X_df.shape[0]
    part_nrow = int(nrow * (1 / nsplit))
    print(part_nrow)

    regs = []
    X_df = X_df.copy().reset_index().drop("ID", axis=1)
    y_df = y_df.copy().reset_index().drop("ID", axis=1)
    for i in range(nsplit):
        test_start = i * part_nrow
        test_end = (i + 1) * part_nrow

        X_test_df = X_df.loc[test_start : test_end - 1, :].copy()
        y_test_df = y_df.loc[test_start : test_end - 1, :].copy()
        X_train_df = pd.concat(
            [X_df.loc[test_end:, :], X_df.loc[: test_start - 1, :]], axis=0,
        )
        y_train_df = pd.concat(
            [y_df.loc[test_end:, :], y_df.loc[: test_start - 1, :]], axis=0,
        )
        print(f"X_train shape: {X_train_df.shape}, y_train shape: {y_train_df.shape}")
        print(f"X_test shape: {X_test_df.shape}, y_test shape: {y_test_df.shape}")

        X_train, X_test, y_train, y_test = (
            X_train_df.to_numpy(),
            X_test_df.to_numpy(),
            np.squeeze(y_train_df.to_numpy()),
            np.squeeze(y_test_df.to_numpy()),
        )
        # X_train, X_test = X_test, X_train
        # y_train, y_test = y_test, y_train
        reg = estimator_class()
        reg.fit(X_train, y_train)
        regs.append(reg)
        report = regression_report(y_test, reg.predict(X_test), X_test.shape[1])
        print("-" * 10, f"revenue report ({i})", "-" * 10)
        print(report)

    return regs


def main(regressor, X_train_df, X_test_df, y_train_df, y_test_df, nsplit=2):
    X_train, X_test, y_train, y_test = (
        X_train_df.to_numpy(),
        X_test_df.to_numpy(),
        y_train_df["revenue"].to_numpy(),
        y_test_df["revenue"].to_numpy(),
    )

    regs = split_train(regressor, X_train_df, y_train_df["revenue"], nsplit)
    preds = []
    for reg in regs:
        pred = reg.predict(X_test_df.to_numpy())
        preds.append(pred)
    pred = np.sum(preds, axis=0) / len(preds)

    print("-" * 5, "revenue evaluation", "-" * 5)
    y_test = y_test_df["revenue"].to_numpy()
    report = regression_report(y_test, pred, X_test_df.shape[1])
    print(report)

    pred_df = X_test_df.copy()
    pred_df["pred_revenue"] = pred
    pred_label_df = data.to_label(pred_df)
    true_label_df = pd.read_csv("data/revenue_per_day.csv", index_col="arrival_date")
    print("-" * 5, "label evaluation", "-" * 5)
    evaluate_by_label2(pred_label_df, true_label_df, "label")
    print("-" * 5, "revenue_per_day evaluation", "-" * 5)
    evaluate_by_label2(pred_label_df, true_label_df, "revenue")


# data
data = Data(use_dummies=False, normalize=False)
X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
    ["revenue"], test_ratio=0.3
)
print(f"X_train shape {X_train_df.shape}, y_train shape {y_train_df.shape}")
print(f"X_test shape {X_test_df.shape}, y_test shape {y_test_df.shape}")

for i in range(2, 20, 3):
    print(f"nsplit = {i}")
    main(
        HistGradientBoostingRegressor,
        X_train_df,
        X_test_df,
        y_train_df,
        y_test_df,
        nsplit=i,
    )
#%%
# label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
# print("-" * 10, "evaluation", "-" * 10)
# evaluate(reg, X_test_df, label_df)
