import math
import sklearn.metrics as metrics

# report for regression result
def regression_report(y_true, y_pred, nfeatures=None):
    r2 = metrics.r2_score(y_true, y_pred)
    max_error = metrics.max_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    if nfeatures is not None:
        adjusted_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - nfeatures - 1)
    else:
        adjusted_r2 = None
    report = [
        f"R2: {r2:.3f}",
        # FIXME: case when Adj. R2 is None
        f"Adj. R2: {adjusted_r2:.3f}",
        f"MSE: {mse:.3f}",
        f"RMSE: {math.sqrt(mse):.3f}",
        f"Max Error: {max_error:.3f}",
    ]
    return "\n".join(report) + "\n"
