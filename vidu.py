# app.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gold Price Forecast Dashboard", layout="wide")
st.title("📈 Gold Price Forecast Dashboard")
st.caption("Demo web tổng quan + dự đoán (baseline ML). Không phải lời khuyên đầu tư.")

# =========================
# LOAD & CLEAN
# =========================
@st.cache_data
def load_data(path: str):
    df0 = pd.read_csv(path)

    # drop index column if exists
    if "Unnamed: 0" in df0.columns:
        df0 = df0.drop(columns=["Unnamed: 0"])

    # parse date
    df0["Date"] = pd.to_datetime(df0["Date"])
    df0 = df0.sort_values("Date").reset_index(drop=True)

    # basic cleaning
    df = df0.copy()
    df = df.drop_duplicates()

    # validation rules (nếu cột tồn tại)
    required = {"Open", "High", "Low", "Close/Last"}
    if required.issubset(df.columns):
        df = df[
            (df["High"] >= df["Open"]) &
            (df["High"] >= df["Close/Last"]) &
            (df["High"] >= df["Low"]) &
            (df["Low"] <= df["Open"]) &
            (df["Low"] <= df["Close/Last"])
        ]

    # Volume non-negative if exists
    if "Volume" in df.columns:
        df = df[df["Volume"] >= 0]

    df = df.reset_index(drop=True)
    return df0, df

DATA_PATH = st.sidebar.text_input(
    "CSV path",
    value="D:/Documents/GitHub/GiaVang/goldstock v2.csv"
)

try:
    original_df, df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Không load được file. Lỗi: {e}")
    st.stop()

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.markdown("## Điều khiển")
horizon = st.sidebar.slider("Số ngày dự báo (N ngày tới)", 3, 60, 14, 1)
test_ratio = st.sidebar.slider("Tỷ lệ test (time split)", 0.1, 0.4, 0.2, 0.05)

# =========================
# FEATURE ENGINEERING
# =========================
def make_supervised_timeseries(df: pd.DataFrame):
    """Tạo bộ feature lag + rolling để dự đoán Close/Last(t+1)"""
    d = df.copy()

    # đảm bảo cột mục tiêu
    if "Close/Last" not in d.columns:
        raise ValueError("Dataset thiếu cột 'Close/Last'.")

    # target: ngày mai
    d["y"] = d["Close/Last"].shift(-1)

    # lag features
    for lag in [1, 2, 3, 5, 7, 14]:
        d[f"close_lag_{lag}"] = d["Close/Last"].shift(lag)

    # rolling features
    d["ma_7"] = d["Close/Last"].rolling(7).mean()
    d["ma_14"] = d["Close/Last"].rolling(14).mean()
    d["std_7"] = d["Close/Last"].rolling(7).std()

    if "Volume" in d.columns:
        d["vol_lag_1"] = d["Volume"].shift(1)
        d["vol_ma_7"] = d["Volume"].rolling(7).mean()

    d = d.dropna().reset_index(drop=True)
    feature_cols = [c for c in d.columns if c.startswith(("close_lag_", "ma_", "std_", "vol_"))]
    return d, feature_cols

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "🧾 Data", "🔎 EDA", "🤖 Model", "🔮 Forecast"])

# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.subheader("Tổng quan nhanh")

    latest_close = df["Close/Last"].iloc[-1]
    prev_close = df["Close/Last"].iloc[-2] if len(df) > 1 else latest_close
    delta = latest_close - prev_close
    delta_pct = (delta / prev_close) * 100 if prev_close != 0 else 0.0

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Close ล่าสุด", f"${latest_close:,.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
    colB.metric("Số dòng", f"{len(df):,}")
    colC.metric("Khoảng thời gian", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    if "Volume" in df.columns:
        colD.metric("Volume TB", f"{df['Volume'].tail(30).mean():,.0f}")
    else:
        colD.metric("Volume", "N/A")

    # Trend chart
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["Date"], df["Close/Last"], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close/Last")
    ax.set_title("Gold Close Price Trend")
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

# =========================
# TAB 2: DATA
# =========================
with tab2:
    st.subheader("Dữ liệu & làm sạch")

    c1, c2 = st.columns(2)
    with c1:
        st.write("### Dữ liệu gốc (sample)")
        st.dataframe(original_df.head(10), use_container_width=True)
    with c2:
        st.write("### Dữ liệu sau làm sạch (sample)")
        st.dataframe(df.head(10), use_container_width=True)

    st.write("### Thống kê missing")
    missing = original_df.isnull().sum()
    miss_df = pd.DataFrame({"col": missing.index, "missing": missing.values})
    st.dataframe(miss_df[miss_df["missing"] > 0] if miss_df["missing"].sum() > 0
                 else pd.DataFrame({"Status": ["✅ Không có missing values"]}),
                 use_container_width=True)

    st.write("### So sánh trước/sau")
    st.write(f"- Trước: **{len(original_df):,}** dòng")
    st.write(f"- Sau: **{len(df):,}** dòng")
    st.write(f"- Loại bỏ: **{(len(original_df) - len(df)):,}** dòng")

# =========================
# TAB 3: EDA
# =========================
with tab3:
    st.subheader("Exploratory Data Analysis")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Date" in num_cols:
        num_cols.remove("Date")

    col_pick = st.selectbox("Chọn cột để xem phân phối", options=num_cols, index=num_cols.index("Close/Last") if "Close/Last" in num_cols else 0)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[col_pick].dropna(), bins=30)
        ax.set_title(f"Histogram: {col_pick}")
        ax.set_xlabel(col_pick)
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    with c2:
        corr_cols = [c for c in ["Open", "High", "Low", "Close/Last", "Volume"] if c in df.columns]
        if len(corr_cols) >= 2:
            corr = df[corr_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr.values)
            ax.set_xticks(range(len(corr_cols)))
            ax.set_yticks(range(len(corr_cols)))
            ax.set_xticklabels(corr_cols, rotation=45, ha="right")
            ax.set_yticklabels(corr_cols)
            ax.set_title("Correlation (subset)")
            # annotate
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Không đủ cột số để vẽ correlation.")

# =========================
# TAB 4: MODEL
# =========================
with tab4:
    st.subheader("Mô hình dự đoán (Ví dụ: Linear Regression với lag/rolling features)")

    try:
        sup_df, feature_cols = make_supervised_timeseries(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # time split
    n = len(sup_df)
    test_n = int(n * test_ratio)
    train_df = sup_df.iloc[:-test_n].copy()
    test_df = sup_df.iloc[-test_n:].copy()

    X_train, y_train = train_df[feature_cols], train_df["y"]
    X_test, y_test = test_df[feature_cols], test_df["y"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = mean_squared_error(y_test, pred_test, squared=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("Test size", f"{len(test_df):,} ngày")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_df["Date"], y_test.values, label="Actual", linewidth=2)
    ax.plot(test_df["Date"], pred_test, label="Predicted", linewidth=2)
    ax.set_title("Actual vs Predicted (Test)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close/Last (next day)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

    st.write("### Feature coefficients (tham khảo)")
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": model.coef_}).sort_values("coef", ascending=False)
    st.dataframe(coef_df, use_container_width=True)

    st.info(
        "Ghi chú: Linear Regression ở đây là **baseline** để demo website. "
        "Nếu muốn mạnh hơn, bạn có thể thay bằng RandomForest/XGBoost/ARIMA/Prophet/LSTM."
    )

# =========================
# TAB 5: FORECAST
# =========================
with tab5:
    st.subheader("Dự báo N ngày tới (Forecast)")

    # Train on full supervised data
    sup_df, feature_cols = make_supervised_timeseries(df)
    model = LinearRegression()
    model.fit(sup_df[feature_cols], sup_df["y"])

    # recursive forecasting
    hist = df[["Date", "Close/Last"]].copy().reset_index(drop=True)

    # Nếu có volume, giữ để tạo vol features (đơn giản: dùng volume gần nhất lặp lại)
    has_vol = "Volume" in df.columns
    last_volume = df["Volume"].iloc[-1] if has_vol else None

    future_dates = pd.date_range(start=df["Date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")

    # tạo series close để tính rolling/lag
    close_series = df["Close/Last"].copy().reset_index(drop=True)
    vol_series = df["Volume"].copy().reset_index(drop=True) if has_vol else None

    preds = []

    for i in range(horizon):
        # build one-row feature vector using current close_series
        temp = pd.DataFrame({"Close/Last": close_series})
        # create same features
        row = {}
        for lag in [1, 2, 3, 5, 7, 14]:
            row[f"close_lag_{lag}"] = float(temp["Close/Last"].iloc[-lag]) if len(temp) >= lag else np.nan
        row["ma_7"] = float(temp["Close/Last"].tail(7).mean())
        row["ma_14"] = float(temp["Close/Last"].tail(14).mean())
        row["std_7"] = float(temp["Close/Last"].tail(7).std(ddof=0))

        if has_vol:
            # dùng volume gần nhất (giữ đơn giản)
            row["vol_lag_1"] = float(vol_series.iloc[-1])
            row["vol_ma_7"] = float(vol_series.tail(7).mean())

        x = pd.DataFrame([row])[feature_cols].fillna(method="ffill", axis=1).fillna(0)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)

        # append predicted close into close_series for next step
        close_series = pd.concat([close_series, pd.Series([yhat])], ignore_index=True)
        if has_vol:
            vol_series = pd.concat([vol_series, pd.Series([last_volume])], ignore_index=True)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast_Close": preds})

    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Date"].tail(200), df["Close/Last"].tail(200), label="History (last 200)", linewidth=2)
        ax.plot(forecast_df["Date"], forecast_df["Forecast_Close"], label="Forecast", linewidth=2)
        ax.set_title(f"Forecast next {horizon} days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close/Last")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.write("### Bảng dự báo")
        st.dataframe(forecast_df, use_container_width=True)

    st.warning("Disclaimer: Dự báo chỉ để học tập/demo, không phải tư vấn đầu tư.")
