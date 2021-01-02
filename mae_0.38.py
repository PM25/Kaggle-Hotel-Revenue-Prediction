#%%
from utils import *
from utils.metrics import regression_report
from data_processing import Data, evaluate_by_label, fill_label

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor


if __name__ == "__main__":
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
    label_df = data.get_true_label(columns=["adr", "revenue", "is_canceled", "label"])
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
