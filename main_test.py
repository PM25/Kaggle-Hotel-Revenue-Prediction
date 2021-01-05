#%%
from utils import *
from data_processing import Data, evaluate_by_label, fill_label
from utils.metrics import regression_report

import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import (
    BaggingRegressor,
    BaggingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, classification_report

import threading
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.utils import all_estimators

#%%
RANDOM_SEED = 1126


def evaluate(reg, X_df, label_df):
    pred_df = data.predict_label(reg, X_df)

    label_true_pred = []
    for date, row in pred_df.iterrows():
        label_true = label_df.loc[date, "label"]
        label_pred = row["pred_label"]
        label_true_pred.append((label_true, label_pred))

    label_true = [true for true, pred in label_true_pred]
    label_pred = [pred for true, pred in label_true_pred]
    # Visualization(label_true, label_pred).classification_report().confusion_matrix()
    report = [
        f"MAE: {mean_absolute_error(label_true, label_pred)}",
        classification_report(label_true, label_pred),
    ]
    return "\n".join(report)


#%%
class MLModelWrapper2:
    def __init__(self, X_df, X_train, X_test, y_train, y_test):
        self.X_df = X_df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # TODO: finished implement cluster and transformer
    # filter_type can be one of the type ['classifier', 'regressor', 'cluster', 'transformer']
    def quick_test(self, filter_type="classifier", max_threads=5, save=True):
        label_df = pd.read_csv("data/train_label.csv", index_col="arrival_date")
        print("*Quick test for multiple classification models!")
        threads = []
        for name, estimator_class in all_estimators(filter_type):
            print(f"*start training: {name} model.")
            model = estimator_class()
            try:
                model = estimator_class()
                thread = TrainModelThread2(
                    self.X_df.copy(),
                    self.X_train.copy(),
                    self.y_train.copy(),
                    self.X_test.copy(),
                    self.y_test.copy(),
                    label_df.copy(),
                    model,
                    filter_type,
                    name,
                    save,
                )
                threads.append(thread)
                thread.start()
                if len(threads) > 5:
                    break
            except:
                print(f"*Failed to initialize model: {name}.")

        for thread in threads:
            thread.join()
        print("*Training of all classification models are finished!")


class TrainModelThread2(threading.Thread):
    def __init__(
        self,
        X_df,
        X_train,
        y_train,
        X_test,
        y_test,
        label_df,
        model,
        model_type="classifier",
        name=None,
        save=True,
    ):
        threading.Thread.__init__(self)
        self.X_df = X_df
        self.train_X = X_train
        self.train_y = y_train
        self.test_X = X_test
        self.test_y = y_test
        self.model = model
        self.model_type = model_type
        self.name = name
        self.save = save
        self.label_df = label_df

    def run(self):
        self.model.fit(self.train_X, self.train_y)
        y_pred = self.model.predict(self.test_X)
        if self.model_type == "classifier":
            report = classification_report(self.test_y, y_pred)
        elif self.model_type == "regressor":
            report = regression_report(self.test_y, y_pred, self.test_X.shape[1])
        if self.name != None:
            print(f"Method: {self.name}")

        print("-" * 10, "evaluation", "-" * 10)
        report += evaluate(self.model, self.X_df, self.label_df)

        print(report)
        if self.save:
            print(f"*Append result to SKLearn_{self.model_type}s_Report.txt")
            with open(f"SKLearn_{self.model_type}s_Report.txt", "a") as ofile:
                if self.name != None:
                    ofile.write(f"Method: {self.name}\n")
                ofile.write(f"finished time: {datetime.now()}\n")
                ofile.write(report)
                ofile.write("-" * 20 + "\n")
        print("-" * 20)


# test classifiers
data = Data(use_dummies=False, normalize=False)
X_train_df, X_test_df, y_train_df, y_test_df = data.train_test_split_by_date(
    "revenue", test_ratio=0.3
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


print(X_test_df.shape)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
models = MLModelWrapper2(X_test_df, X_train, X_test, y_train, y_test)
models.quick_test("regressor")

#%%
