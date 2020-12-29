#%%
from utils import *
from utils.metrics import regression_report

import pandas as pd
from data_processing import Data
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingRegressor,
    BaggingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, classification_report

#%%
def evaluate(reg, X_df, label_df):
    pred_df = data.predict_label(reg, X_df)

    label_true_pred = []
    for idx, row in pred_df.iterrows():
        date = row["arrival_date"]
        label_true = label_df.loc[date, "label"]
        label_pred = row["pred_label"]
        label_true_pred.append((label_true, label_pred))

    label_true = [true for true, pred in label_true_pred]
    label_pred = [pred for true, pred in label_true_pred]
    print(f"MAE: {mean_absolute_error(label_true, label_pred)}")
    print(classification_report(label_true, label_pred))
    Visualization(label_true, label_pred).classification_report().confusion_matrix()


#%% fill label
def fill_label(predict_df, fname="data/test_nolabel.csv"):
    label_df = pd.read_csv(fname, index_col="arrival_date")

    label_df["label"] = 0
    for idx, subdf in predict_df.iterrows():
        label_df.loc[idx, "label"] = subdf["pred_label"]

    label_df.to_csv("result.csv")


# test classifiers
data = Data(use_dummies=False, normalize=False)
X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
    "revenue", test_ratio=0.2
)
X_train, X_test, y_train, y_test = (
    X_train_df.to_numpy(),
    X_test_df.to_numpy(),
    y_train_df.to_numpy(),
    y_test_df.to_numpy(),
)
print(f"X_train shape {X_train.shape}")
print(f"X_test shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"y_test shape {y_test.shape}")

#%%
# eval_reg = RandomForestRegressor()
eval_reg = DecisionTreeRegressor()
eval_reg.fit(X_train, y_train)
report = regression_report(eval_reg.predict(X_test), y_test, X_test.shape[1])
print("-" * 10, "regression report", "-" * 10)
print(report)

label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
print("-" * 10, "evaluation", "-" * 10)
evaluate(eval_reg, X_test_df, label_df)


#%%
# # training with all data
X_df, y_df = data.processing("revenue")
reg = DecisionTreeRegressor()
# reg = RandomForestRegressor()
reg.fit(X_df.to_numpy(), y_df.to_numpy())


#%%
# test_X_df = data.processing_test_data("data/train.csv")
test_X_df = data.processing_test_data("data/test.csv")
predict_df = data.predict_label(reg, test_X_df)

#%%
predict_df = (
    predict_df[["arrival_date", "pred_label"]]
    .reset_index(drop=True)
    .set_index("arrival_date")
)
#%%
fill_label(predict_df, "data/test_nolabel.csv")

# %%
