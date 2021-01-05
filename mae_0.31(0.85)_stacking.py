#%%
from utils import *
from utils.metrics import regression_report
from data_processing import Data, fill_label, evaluate_by_label

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, classification_report


def get_models():
    is_canceled_clfs = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=100, max_depth=40, random_state=1126),
        ),
    ]

    adr_regs = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(n_estimators=100, max_depth=40, random_state=1126),
        ),
        ("MLPRegressor", MLPRegressor(max_iter=1000, early_stopping=True)),
        ("LinearRegression", LinearRegression()),
    ]
    return is_canceled_clfs, adr_regs


#%%
def save_output(regs, X_df):
    revenue_preds = []
    for reg, models in regs:
        X_w_pred_df = append_pred(models, X_df.copy())
        revenue_pred = reg.predict(X_w_pred_df)
        revenue_preds.append(revenue_pred)
    revenue_pred = np.sum(revenue_preds, axis=0) / len(revenue_preds)

    pred_df = X_df.copy()
    pred_df["pred_revenue"] = revenue_pred
    predict_df = data.to_label(pred_df)
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


nsplit = 3
regressor = HistGradientBoostingRegressor
if __name__ == "__main__":
    data = Data(use_dummies=False, normalize=False)
    X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
        ["revenue", "is_canceled", "adr"], test_ratio=0.3
    )
    print(f"X_train shape {X_train_df.shape}, y_train shape {y_train_df.shape}")
    print(f"X_test shape {X_test_df.shape}, y_test shape {y_test_df.shape}")

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
    true_label_df = data.get_true_label(
        columns=["adr", "revenue", "is_canceled", "label"]
    )

    report.append("[ label evaluation ]")
    report.append(evaluate_by_label(pred_label_df, true_label_df, "label"))
    report.append("[ revenue_per_day evaluation ]")
    report.append(evaluate_by_label(pred_label_df, true_label_df, "revenue"))
    report = "\n".join(report) + "\n"
    print(report)

    # training with all data
    X_df, y_df = data.processing(["revenue", "is_canceled", "adr"])
    regs = split_train(regressor, X_df, y_df, nsplit)

    test_X_df = data.processing_test_data()
    save_output(regs, test_X_df)

