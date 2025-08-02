# src/models.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def smape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def train_xgb_reg(X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> XGBRegressor:
    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
    )
    params.update(kwargs)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_predictions(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }

def recursive_forecast(
    model: XGBRegressor,
    df_full: pd.DataFrame,
    feature_maker,   # function(df)->df with lags/rollings/time feats
    horizon: int,
    target_col: str = "sales",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    df_full must include all columns needed for feature_maker and known history up to T.
    Returns (future_df_with_features, preds)
    """
    df = df_full.copy().sort_values("date")
    preds = []
    for step in range(1, horizon + 1):
        # Build features with current history
        feats = feature_maker(df)
        # Find the last date in feats (current 'T')
        last_date = feats["date"].max()
        # Select the row to predict = next date (we'll synthesize it)
        next_date = last_date + pd.Timedelta(days=1)

        # Build a one-row frame for next_date using known calendar features + lags/rollings from feats
        # We take the last row of feats as template, then adjust date-based columns by recomputing
        row_template = feats.iloc[-1:].copy()
        row_template["date"] = next_date
        # Recompute calendar features
        row_template["dow"] = next_date.dayofweek
        row_template["week"] = next_date.isocalendar().week
        row_template["month"] = next_date.month
        row_template["year"] = next_date.year

        # For lags/rollings: re-run feature_maker on df (already up to date),
        # then grab the last row's lag/rolling columns (they are functions of history in df)
        feats_updated = feature_maker(df)
        one_row = feats_updated.iloc[-1:].copy()
        one_row["date"] = next_date
        # Ensure calendar feats correspond to next_date
        one_row["dow"] = next_date.dayofweek
        one_row["week"] = next_date.isocalendar().week
        one_row["month"] = next_date.month
        one_row["year"] = next_date.year

        # Predict with model: drop non-feature cols
        feature_cols = [c for c in one_row.columns if c not in ["date", target_col]]
        yhat = float(model.predict(one_row[feature_cols])[0])
        preds.append(yhat)

        # Append synthetic prediction to history to roll lags forward
        new_row = {col: None for col in df.columns}
        for col in df.columns:
            if col in one_row.columns:
                new_row[col] = one_row[col].values[0]
        new_row[target_col] = yhat
        new_row["date"] = next_date
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    future_dates = pd.date_range(start=df_full["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({"date": future_dates, "pred": preds})
    return future_df, pd.Series(preds)
