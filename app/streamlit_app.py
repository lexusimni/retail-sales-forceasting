import os, json, joblib
import pandas as pd
import numpy as np
import streamlit as st

from datetime import timedelta
from src.features import load_sales, filter_series, build_feature_frame
from src.models import train_xgb_reg, evaluate_predictions, recursive_forecast

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "store_item_demand.csv")

st.set_page_config(page_title="Retail Sales Forecasting", layout="centered")
st.title("🛒 Retail Sales Forecasting")

@st.cache_resource
def load_data():
    return load_sales(DATA_PATH)

df_all = load_data()
stores = sorted(df_all["store"].unique())
items  = sorted(df_all["item"].unique())

col1, col2 = st.columns(2)
with col1:
    store_id = st.selectbox("Select Store", stores, index=0)
with col2:
    item_id = st.selectbox("Select Item", items, index=0)

horizon = st.slider("Forecast Horizon (days)", 7, 60, 28)

series = filter_series(df_all, store=store_id, item=item_id)

# Build features and split last 28 days for quick validation
df_feat = build_feature_frame(series)
if len(df_feat) < 60:
    st.warning("Not enough history for robust lags/rolling features.")
feature_cols = [c for c in df_feat.columns if c not in ["date","sales"]]

cutoff = df_feat["date"].max() - pd.Timedelta(days=28)
train_df = df_feat[df_feat["date"] <= cutoff].copy()
valid_df = df_feat[df_feat["date"] >  cutoff].copy()

if len(train_df) < 30 or len(valid_df) < 7:
    st.warning("Limited data after lag/rolling feature generation.")

model = train_xgb_reg(train_df[feature_cols], train_df["sales"])
y_pred = model.predict(valid_df[feature_cols])

metrics = evaluate_predictions(valid_df["sales"], y_pred)
st.subheader("Validation Metrics (last 28 days)")
st.write(metrics)

# Plot validation
st.subheader("Validation — Actual vs Predicted")
vplot = pd.DataFrame({
    "date": valid_df["date"],
    "actual": valid_df["sales"].values,
    "pred": y_pred
})
st.line_chart(vplot.set_index("date")[["actual","pred"]])

# Forecast forward
def feature_maker(df_raw):
    return build_feature_frame(df_raw)

future_df, preds = recursive_forecast(model, series, feature_maker, horizon=horizon, target_col="sales")

st.subheader(f"Forecast (next {horizon} days)")
st.line_chart(future_df.set_index("date")[["pred"]])

st.caption("Model: XGBoost with time features, lags, and rolling statistics.")
