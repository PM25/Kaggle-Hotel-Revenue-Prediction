#%%
from utils import *
from data_processing import Data, evaluate_by_label, fill_label
from utils.metrics import regression_report

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import classification_report


def predict(clf, reg, X_df):
    adr_pred = reg.predict(X_df.to_numpy())
    canceled_pred = clf.predict(X_df.to_numpy())

    pred_df = X_df.copy()
    pred_df["pred_actual_adr"] = adr_pred * (1 - canceled_pred)
    pred_df["pred_revenue"] = (
        pred_df["stays_in_weekend_nights"] + pred_df["stays_in_week_nights"]
    ) * pred_df["pred_actual_adr"]

    return pred_df


if __name__ == "__main__":
    # data
    data = Data(use_dummies=False, normalize=False)
    X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
        ["adr", "is_canceled"], test_ratio=0.3
    )
    X_train, X_test, y_train_adr, y_test_adr, y_train_canceled, y_test_canceled = (
        X_train_df.to_numpy(),
        X_test_df.to_numpy(),
        y_train_df["adr"].to_numpy(),
        y_test_df["adr"].to_numpy(),
        y_train_df["is_canceled"].to_numpy(),
        y_test_df["is_canceled"].to_numpy(),
    )
    print(f"X_train shape {X_train.shape}, y_train shape {y_train_adr.shape}")
    print(f"X_test shape {X_test.shape}, y_test shape {y_test_adr.shape}")

    #%% evaluate performance with training data
    eval_reg = HistGradientBoostingRegressor(random_state=1129)
    eval_reg.fit(X_train.copy(), y_train_adr.copy())
    print("-" * 10, "regression report", "-" * 10)
    report = regression_report(
        y_test_adr.copy(), eval_reg.predict(X_test.copy()), X_test.shape[1]
    )
    print(report)

    # eval_clf = RandomForestClassifier(random_state=1129)
    eval_clf = HistGradientBoostingClassifier(random_state=1129)
    eval_clf.fit(X_train.copy(), y_train_canceled.copy())
    print("-" * 10, "classification report", "-" * 10)
    report = classification_report(
        y_test_canceled.copy(), eval_clf.predict(X_test.copy())
    )
    print(report)

    #%%
    pred_df = predict(eval_clf, eval_reg, X_test_df)
    pred_label_df = data.to_label(pred_df)
    label_df = data.get_true_label(columns=["adr", "revenue", "is_canceled", "label"])

    print("[ label evaluation ]")
    report_label = evaluate_by_label(pred_label_df, label_df, target="label")
    print(report_label)
    print("[ revenue_per_day evaluation ]")
    report_revenue = evaluate_by_label(pred_label_df, label_df, target="revenue")
    print(report_revenue)

    #%% training with all data
    X_df, y_df = data.processing(["adr", "is_canceled"])
    reg = HistGradientBoostingRegressor(random_state=1126)
    reg.fit(X_df.to_numpy(), y_df["adr"].to_numpy())
    clf = HistGradientBoostingClassifier(random_state=1126)
    clf.fit(X_df.to_numpy(), y_df["is_canceled"].to_numpy())

    #%% fill predict label to csv
    test_X_df = data.processing_test_data("data/test.csv")
    pred_df = predict(clf, reg, test_X_df)
    pred_label_df = data.to_label(pred_df)
    fill_label(pred_label_df, "data/test_nolabel.csv")
