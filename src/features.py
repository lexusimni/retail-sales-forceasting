# src/features.py
import pandas as pd
import numpy as np

def load_sales(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # enforce schema
    df.columns = [c.strip().lower() for c in df.columns]
    assert set(["date","store","item","sales"]).issubset(set(df.columns)), "CSV must have date, store, item, sales"
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store","item","date"])
    return df

def filter_series(df: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    return df[(df["store"] == store) & (df["item"] == item)].copy()

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["year"] = out["date"].dt.year
    return out

def add_lags(df: pd.DataFrame, target_col: str = "sales", lags=(1,7,28)):
    out = df.copy()
    for l in lags:
        out[f"lag_{l}"] = out[target_col].shift(l)
    return out

def add_rollings(df: pd.DataFrame, target_col: str = "sales", windows=(7,28)):
    out = df.copy()
    for w in windows:
        out[f"rmean_{w}"] = out[target_col].shift(1).rolling(w).mean()
        out[f"rstd_{w}"]  = out[target_col].shift(1).rolling(w).std()
    return out

def build_feature_frame(df: pd.DataFrame, target_col: str = "sales"):
    # Order matters for lags/rollings to be computed on prior values
    out = add_time_features(df)
    out = add_lags(out, target_col=target_col, lags=(1,7,28))
    out = add_rollings(out, target_col=target_col, windows=(7,28))
    # Drop rows with NA created by lags/rollings
    out = out.dropna().reset_index(drop=True)
    return out
