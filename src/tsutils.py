# src/tsutils.py
import pandas as pd

def train_valid_split_by_date(df, date_col="date", valid_days=28):
    df = df.sort_values(date_col)
    cutoff = df[date_col].max() - pd.Timedelta(days=valid_days)
    train = df[df[date_col] <= cutoff].copy()
    valid = df[df[date_col] > cutoff].copy()
    return train, valid
