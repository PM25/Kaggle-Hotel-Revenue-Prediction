#%%
from utils import *
from utils.metrics import regression_report

import threading
import numpy as np
import pandas as pd
from datetime import datetime
from data_processing import Data
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingRegressor,
    BaggingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Lars, HuberRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, classification_report


def get_models():
    is_canceled_clfs = [
        # (
        #     "RandomForestClassifier",
        #     RandomForestClassifier(n_estimators=100, max_depth=40, random_state=1129),
        # ),
        #     ("BaggingClassifier", BaggingClassifier()),
        # ("KNeighborsClassifier", KNeighborsClassifier()),
        # ("MLPClassifier", MLPClassifier(max_iter=1000, early_stopping=True)),
        # (
        #     "CalibratedClassifierCV",
        #     CalibratedClassifierCV(base_estimator=LinearSVC(max_iter=10000)),
        # ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=100, max_depth=40, random_state=1129),
        ),
        # ("MLPRegressor", MLPRegressor(max_iter=1000, early_stopping=True)),
    ]

    adr_regs = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=100, max_depth=40, random_state=1129),
        ),
        #     ("BaggingRegressor", BaggingRegressor()),
        # (
        #     "HistGradientBoostingRegressor",
        #     HistGradientBoostingRegressor(random_state=1129),
        # ),
        ("MLPRegressor", MLPRegressor(max_iter=1000, early_stopping=True)),
        ("LinearRegression", LinearRegression()),
        # ("BayesianRidge", BayesianRidge()),
        # ("KNeighborsRegressor", KNeighborsRegressor()),
        # ("Lars", Lars()),
        # ("HuberRegressor", HuberRegressor(max_iter=10000)),
    ]
    return is_canceled_clfs, adr_regs


#%%
def get_true_pred_labels(reg, X_df, label_df, target):
    pred_df = data.predict(reg, X_df)

    true_pred_labels = []
    for date, row in pred_df.iterrows():
        if target == "label":
            label_true = label_df.loc[date, "label"]
            label_pred = row["pred_label"]
        else:
            label_true = label_df.loc[date, "revenuePerDay"]
            label_pred = row["pred_revenue_per_day"]
        true_pred_labels.append((label_true, label_pred))

    return true_pred_labels


#%%
# target = label or revenue
def evaluate_by_label(reg, X_df, label_df, target="label"):
    true_pred_labels = get_true_pred_labels(reg, X_df, label_df, target)
    label_true = [true for true, pred in true_pred_labels]
    label_pred = [pred for true, pred in true_pred_labels]
    print(f"MAE: {mean_absolute_error(label_true, label_pred)}")
    if target == "label":
        print(classification_report(label_true, label_pred))
        Visualization(
            label_true, label_pred
        ).classification_report().confusion_matrix().show()


#%%
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


#%%
def save_output(reg):
    # training with all data
    X_df, y_df = data.processing(["revenue"])
    reg.fit(X_df.to_numpy(), y_df.to_numpy())

    test_X_df = data.processing_test_data("data/test.csv")
    predict_df = data.predict(reg, test_X_df)
    fill_label(predict_df, "data/test_nolabel.csv")
    print("*Saved prediction Ouput")


#%%
def append_pred(models, df):
    out_df = df.copy()
    for name, model in models:
        pred = model.predict(df.to_numpy())
        out_df[f"{name}"] = pred
    return out_df


def train_w_reg_clf(X1_df, X2_df, y1_df, y2_df):
    clfs, regs = get_models()

    print(f"[ training classifier ]")
    train_clfs = []
    for idx, (name, clf) in enumerate(clfs):
        clf.fit(X1_df.to_numpy(), y1_df["is_canceled"].to_numpy())
        score = clf.score(X2_df.to_numpy(), y2_df["is_canceled"].to_numpy())
        print(f"[{idx}/{len(clfs)}] {name} score: {score}")
        train_clfs.append((name, clf))

    print(f"[ training regressor ]")
    train_regs = []
    for idx, (name, reg) in enumerate(regs):
        reg.fit(X1_df.to_numpy(), y1_df["adr"].to_numpy())
        score = reg.score(X2_df.to_numpy(), y2_df["adr"].to_numpy())
        print(f"[{idx}/{len(regs)}] {name} score: {score}")
        train_regs.append((name, reg))

    X2_w_pred_df = append_pred(train_clfs + train_regs, X2_df)
    return X2_w_pred_df, train_clfs + train_regs


#%%
def cross_train(estimator_class, X1_df, X2_df, y1_df, y2_df):
    X2_w_pred_df, models = train_w_reg_clf(
        X1_df.copy(), X2_df.copy(), y1_df.copy(), y2_df.copy()
    )

    reg = estimator_class()
    reg.fit(X2_w_pred_df.to_numpy(), y2_df["revenue"].to_numpy())

    return reg, models


#%%
def split_train(estimator_class, X_df, y_df, nsplit=2):
    nrow = X_df.shape[0]
    part_nrow = int(nrow * (1 / nsplit))

    regs = []
    X_df = X_df.copy().reset_index().drop("ID", axis=1)
    y_df = y_df.copy().reset_index().drop("ID", axis=1)
    for i in range(nsplit):
        print("-" * 5, f"[{i}/{nsplit}] split training", "-" * 5)
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
        # print(f"X_train shape: {X_train_df.shape}, y_train shape: {y_train_df.shape}")
        # print(f"X_test shape: {X_test_df.shape}, y_test shape: {y_test_df.shape}")
        X_train_df, X_test_df = X_test_df, X_train_df
        y_train_df, y_test_df = y_test_df, y_train_df
        reg, models = cross_train(
            estimator_class, X_train_df, X_test_df, y_train_df, y_test_df
        )
        regs.append((reg, models))

    return regs


def main(regressor, X_train_df, X_test_df, y_train_df, y_test_df, nsplit=2):
    # data
    X_train, X_test, y_train, y_test = (
        X_train_df.to_numpy(),
        X_test_df.to_numpy(),
        y_train_df["revenue"].to_numpy(),
        y_test_df["revenue"].to_numpy(),
    )

    # training
    regs = split_train(regressor, X_train_df, y_train_df, nsplit)

    # evaluation on validation data
    revenue_preds = []
    for reg, models in regs:
        X_df = append_pred(models, X_test_df.copy())
        revenue_pred = reg.predict(X_df)
        revenue_preds.append(revenue_pred)
    revenue_pred = np.sum(revenue_preds, axis=0) / len(revenue_preds)

    # print report
    report = []
    report.append("[ revenue_per_order evaluation ]")
    y_test = y_test_df["revenue"].to_numpy()
    reg_report = regression_report(y_test, revenue_pred, X_test_df.shape[1])
    report.append(reg_report)

    pred_df = X_test_df.copy()
    pred_df["pred_revenue"] = revenue_pred
    pred_label_df = data.to_label(pred_df)
    true_label_df = pd.read_csv("data/revenue_per_day.csv", index_col="arrival_date")

    report.append("[ label evaluation ]")
    report.append(evaluate_by_label2(pred_label_df, true_label_df, "label"))
    report.append("[ revenue_per_day evaluation ]")
    report.append(evaluate_by_label2(pred_label_df, true_label_df, "revenue"))
    return "\n".join(report) + "\n"


#%% data
data = Data(use_dummies=False, normalize=False)
X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
    ["revenue", "is_canceled", "adr"], test_ratio=0.3
)
print(f"X_train shape {X_train_df.shape}, y_train shape {y_train_df.shape}")
print(f"X_test shape {X_test_df.shape}, y_test shape {y_test_df.shape}")


report = main(
    HistGradientBoostingRegressor,
    X_train_df,
    X_test_df,
    y_train_df,
    y_test_df,
    nsplit=2,
)
clfs, regs = get_models()
print(report)
print(f"*Save result to Ensemble_w_Pred_Report.txt")
with open(f"Ensemble_w_Pred_Report2.txt", "a") as ofile:
    ofile.write(f"nsplit={3}\n")
    ofile.write(f"is_canceled classifier: {[name for name, _ in clfs]}\n")
    ofile.write(f"adr regressor: {[name for name, _ in regs]}]\n")
    ofile.write(f"finished time: {datetime.now()}\n\n")
    ofile.write(report)
    ofile.write("-" * 10 + " End " + "-" * 10 + "\n")

