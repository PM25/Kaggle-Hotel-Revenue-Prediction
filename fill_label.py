#%%
import pandas as pd

test_df = pd.read_csv("test_results.csv")
groupby_date = test_df.groupby(
    ["arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"]
)


MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

test_nolabel_df = pd.read_csv("data/test_nolabel.csv", index_col="arrival_date")
test_nolabel_df["label"] = 0
for day, daily_revenue in groupby_date["pred_revnue"]:
    label = daily_revenue.sum() // 10000
    day_str = f"{int(day[0])}-{MONTHS[day[1]]:02d}-{int(day[2]):02d}"
    # test_nolabel_df.loc[day_str, "label"] = label
    test_nolabel_df["label"][day_str] = label

test_nolabel_df.to_csv("test_label.csv")


# %%
