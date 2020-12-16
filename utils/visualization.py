import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# set of functions to help visualize ML/DL results.
class Visualization:
    def __init__(self, y_true, y_pred, target_names=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_names = target_names
        self.counter = 0

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        df_cm = pd.DataFrame(cm)
        if self.target_names is not None:
            df_cm.columns = [name for name in self.target_names]
            df_cm.index = [name for name in self.target_names]
        plt.figure(f"{self.counter}. Confusion Matrix")
        ax = sns.heatmap(df_cm, annot=True)
        ax.set(xlabel="Predicted label", ylabel="True label", title="Confustion Matrix")
        plt.tight_layout()
        self.counter += 1

    def classification_report(self):
        report = classification_report(
            self.y_true, self.y_pred, target_names=self.target_names, output_dict=True
        )
        for key in ["accuracy", "macro avg", "weighted avg"]:
            report.pop(key, None)
        for key in report:
            report[key].pop("support", None)
        plt.figure(f"{self.counter}. Classification Report")
        ax = sns.heatmap(pd.DataFrame(report).T, annot=True)
        ax.set(title="Classification Report")
        plt.tight_layout()
        self.counter += 1

    def show(self):
        plt.show()
