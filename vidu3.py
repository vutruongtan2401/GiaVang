# ==========================================================
# DATA MINING PROJECT - GOLD PRICE DATA
# Dataset: goldstock v2.csv
# B1 → B5 (EDA-focused, model minh họa, có GUI)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pltt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats


plt.style.use('ggplot')

# ==========================================================
# LOAD DATA - B0: DATASET OVERVIEW
# ==========================================================
original_df = pd.read_csv("goldstock v2.csv")

# Xóa cột index không cần thiết
if "Unnamed: 0" in original_df.columns:
    original_df.drop(columns=["Unnamed: 0"], inplace=True)

# Chuyển Date sang datetime
original_df["Date"] = pd.to_datetime(original_df["Date"])

# Sắp xếp theo thời gian
original_df.sort_values(by="Date", inplace=True)
original_df.reset_index(drop=True, inplace=True)
# ===== Convert các cột số về numeric (phòng trường hợp có '$' và ',' ) =====
def to_numeric_clean(s: pd.Series):
    if s.dtype == "O":  # object/string
        s = (s.astype(str)
               .str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(s, errors="coerce")

for col in ["Open", "High", "Low", "Close/Last", "Volume"]:
    if col in original_df.columns:
        original_df[col] = to_numeric_clean(original_df[col])

# ===== CLEANING BỔ SUNG SAU CONVERT NUMERIC =====

# Drop Date lỗi
original_df = original_df.dropna(subset=["Date"]).copy()

# Drop NaN phát sinh sau khi convert numeric
num_cols = ["Open", "High", "Low", "Close/Last", "Volume"]
num_cols = [c for c in num_cols if c in original_df.columns]
original_df = original_df.dropna(subset=num_cols).copy()

# Volume phải không âm
if "Volume" in original_df.columns:
    original_df = original_df[original_df["Volume"] >= 0].copy()


# ==========================================================
# B2 – DATA CLEANING (TIỀN XỬ LÝ DỮ LIỆU)
# ==========================================================
# Clone một bản sao để giữ nguyên dữ liệu gốc
df = original_df.copy()

# Kiểm tra dữ liệu thiếu
missing_data = df.isnull().sum() * 100 / df.shape[0]

# Xóa duplicate
# Remove duplicate theo Date (chuẩn time-series)
df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


# Kiểm tra logic giá (High >= Open, Close, Low; Low <= Open, Close)
df = df[
    (df["High"] >= df["Open"]) &
    (df["High"] >= df["Close/Last"]) &
    (df["High"] >= df["Low"]) &
    (df["Low"] <= df["Open"]) &
    (df["Low"] <= df["Close/Last"])
]
# ===== VALIDATION BỔ SUNG =====

# Giá phải dương
for c in ["Open", "High", "Low", "Close/Last"]:
    df = df[df[c] > 0]

# Close và Open phải nằm trong [Low, High]
df = df[
    (df["Close/Last"] >= df["Low"]) &
    (df["Close/Last"] <= df["High"]) &
    (df["Open"] >= df["Low"]) &
    (df["Open"] <= df["High"])
]

df.reset_index(drop=True, inplace=True)

df.reset_index(drop=True, inplace=True)

# ==========================================================
# B1 – MÔ TẢ DỮ LIỆU (DATA OVERVIEW)
# ==========================================================
quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
qualitative_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# ==========================================================
# B5+ (SUPERVISED) – LINEAR REGRESSION FORECAST (NÂNG CẤP)
# ==========================================================
def make_lr_dataset(df_in: pd.DataFrame, horizon: int = 1):
    """
    horizon=1: dự đoán Close(t+1)
    horizon=7: dự đoán Close(t+7)
    
    ✅ CẢI THIỆN: Thêm nhiều feature chất lượng cao (momentum, ROC, RSI, Bollinger Bands)
    """
    d = df_in.copy()

    # Target: future return (thay vì absolute price - dễ scale hơn)
    d["y"] = d["Close/Last"].shift(-horizon) / d["Close/Last"] - 1

    # ===== LAG FEATURES (giữ nguyên) =====
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        d[f"close_lag_{lag}"] = d["Close/Last"].shift(lag)

    # ===== ROLLING FEATURES (NÂNG CẤP) =====
    d["ma_7"] = d["Close/Last"].rolling(7).mean()
    d["ma_14"] = d["Close/Last"].rolling(14).mean()
    d["ma_20"] = d["Close/Last"].rolling(20).mean()
    d["ma_30"] = d["Close/Last"].rolling(30).mean()
    d["ma_60"] = d["Close/Last"].rolling(60).mean()
    d["ma_200"] = d["Close/Last"].rolling(200).mean()
    
    d["std_7"] = d["Close/Last"].rolling(7).std()
    d["std_14"] = d["Close/Last"].rolling(14).std()
    d["std_20"] = d["Close/Last"].rolling(20).std()
    
    # ===== MOMENTUM & TREND FEATURES (TỐI QUAN TRỌNG - CẢI THIỆN) =====
    d["momentum_7"] = d["Close/Last"] - d["Close/Last"].shift(7)
    d["momentum_14"] = d["Close/Last"] - d["Close/Last"].shift(14)
    d["momentum_30"] = d["Close/Last"] - d["Close/Last"].shift(30)
    
    # Rate of change (tốc độ thay đổi %)
    d["roc_7"] = (d["Close/Last"] - d["Close/Last"].shift(7)) / d["Close/Last"].shift(7)
    d["roc_14"] = (d["Close/Last"] - d["Close/Last"].shift(14)) / d["Close/Last"].shift(14)
    d["roc_30"] = (d["Close/Last"] - d["Close/Last"].shift(30)) / d["Close/Last"].shift(30)
    
    # RSI-like oscillator (0-100)
    delta = d["Close/Last"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (độ lệch so với MA)
    d["bb_upper"] = d["ma_20"] + 2 * d["std_20"]
    d["bb_lower"] = d["ma_20"] - 2 * d["std_20"]
    d["bb_position"] = (d["Close/Last"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    
    # ===== VOLUME FEATURES (CẬP NHẬT) =====
    if "Volume" in d.columns:
        d["vol_lag_1"] = d["Volume"].shift(1)
        d["vol_ma_7"] = d["Volume"].rolling(7).mean()
        d["vol_ma_30"] = d["Volume"].rolling(30).mean()
        d["vol_std"] = d["Volume"].rolling(7).std()
        
        # Volume-Price Trend
        d["price_vol_trend"] = (d["Close/Last"] - d["Close/Last"].shift(1)) / d["Close/Last"].shift(1) * d["Volume"]

    # ===== LOẠI BỎ NaN & CHỌN FEATURES =====
    d = d.dropna().reset_index(drop=True)

    feature_cols = [c for c in d.columns if c.startswith(("close_lag_", "ma_", "std_", "vol_", "momentum_", "roc_", "rsi_", "bb_", "price_"))]
    return d, feature_cols


def time_split(d: pd.DataFrame, test_ratio: float = 0.2):
    n = len(d)
    test_n = max(1, int(n * test_ratio))
    train = d.iloc[:-test_n].copy()
    test = d.iloc[-test_n:].copy()
    return train, test


# ...existing code...

# ==========================================================
# STREAMLIT GUI CONFIGURATION
# ==========================================================
st.set_page_config(page_title="Gold Price Data Mining", layout="wide")
st.title("📊 Gold Price Data Mining Project")
st.markdown("---")

# Create tabs for each phase
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "B1 - Data Overview",
    "B2 - Data Cleaning",
    "B3 - Exploratory Analysis",
    "B4 - Correlation & Dimensionality",
    "B5 - Model & Visualization"
])

# ==========================================================
# TAB 1: B1 – DATA OVERVIEW (MÔ TẢ DỮ LIỆU)
# ==========================================================
with tab1:
    st.header("B1 – Mô tả dữ liệu (Dataset Overview)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    
    st.write("### 📋 Danh sách các cột (Dataset Columns)")
    st.write(df.columns.tolist())
    
    st.write("### 📊 Phân loại dữ liệu (Data Types Classification)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Định lượng (Quantitative):**")
        st.write(f"- Số cột: **{len(quantitative_cols)}**")
        for col in quantitative_cols:
            st.write(f"  - `{col}` ({df[col].dtype})")
    
    with col2:
        st.write("**Định tính (Qualitative):**")
        st.write(f"- Số cột: **{len(qualitative_cols)}**")
        for col in qualitative_cols:
            st.write(f"  - `{col}` ({df[col].dtype})")
    
    st.write("### 📈 Thống kê mô tả chi tiết (Descriptive Statistics)")
    st.dataframe(df[quantitative_cols].describe(), use_container_width=True)
    
    st.write("### 🔍 Thông tin chi tiết các cột (Detailed Column Info)")
    info_data = {
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Min": [df[col].min() if col in quantitative_cols else "N/A" for col in df.columns],
        "Max": [df[col].max() if col in quantitative_cols else "N/A" for col in df.columns],
    }
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    st.write("### 📝 Dữ liệu mẫu (Sample Data)")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# TAB 2: B2 – DATA CLEANING (TIỀN XỬ LÝ)
# ==========================================================
with tab2:
    st.header("B2 – Tiền xử lý dữ liệu (Data Cleaning)")
    
    st.write("### 🔍 1. Kiểm tra dữ liệu thiếu (Missing Data)")
    missing_count = original_df.isnull().sum()
    missing_percent = (missing_count / len(original_df)) * 100
    missing_df = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %": missing_percent.values
    })
    st.dataframe(missing_df[missing_df["Missing Count"] > 0] if missing_df["Missing Count"].sum() > 0 
                 else pd.DataFrame({"Status": ["✅ No missing data found"]}), use_container_width=True)
    
    st.write("### 🔄 2. Kiểm tra dữ liệu trùng lặp (Duplicate Data)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"❌ **Trước xử lý:** {original_df.duplicated().sum()} dòng trùng lặp")
    with col2:
        st.write(f"✅ **Sau xử lý:** {df.duplicated().sum()} dòng trùng lặp")
    
    st.write("### 🚨 3. Phát hiện Noise & Outliers")
    
    st.write("### 📌 Outlier theo Return (khuyến nghị cho dữ liệu tài chính)")

    # 1) Check column
    if "Close/Last" not in df.columns:
        st.error("Không tìm thấy cột 'Close/Last' trong df.")
    else:
        # 2) Ép kiểu numeric an toàn
        close = pd.to_numeric(df["Close/Last"], errors="coerce")

        # 3) Tính return + làm sạch NaN/Inf
        ret = close.pct_change()
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

        # 4) Nếu dữ liệu quá ít thì cảnh báo
        if len(ret) < 30:
            st.warning(f"Return hợp lệ quá ít ({len(ret)} điểm) nên thống kê outlier không đáng tin.")
        else:
            q1, q3 = ret.quantile(0.25), ret.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            out_ret = ((ret < lb) | (ret > ub)).sum()
            st.write(f"- Số outlier theo return (IQR): **{out_ret}**")
            st.write(f"- Ngưỡng IQR: **[{lb:.4f}, {ub:.4f}]**")

            st.info("Outlier giá KHÔNG nên xóa vì phản ánh biến động thị trường. Nếu cần ổn định mô hình, cân nhắc xử lý outlier trên return/feature.")
        
        # Kiểm tra outliers bằng IQR
        st.write("**Phương pháp IQR (Interquartile Range):**")
        outlier_summary = []
        
        for col in quantitative_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_summary.append({
                "Column": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Outlier Count": outlier_count
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True)
        
        # Visualize outliers
        fig, axes = plt.subplots(len(quantitative_cols), 1, figsize=(10, 3*len(quantitative_cols)))
        if len(quantitative_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(quantitative_cols):
            sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
            axes[idx].set_title(f"Outlier Detection: {col}")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### ✅ 4. Các quy tắc xác thực dữ liệu (Data Validation Rules)")
        st.write("""
        ✔️ **High >= Open, Close/Last, Low** - Giá cao nhất >= tất cả các mức giá khác
        ✔️ **Low <= Open, Close/Last** - Giá thấp nhất <= tất cả các mức giá khác
        ✔️ **Volume >= 0** - Khối lượng không âm
        ✔️ **Date sorted in ascending order** - Dữ liệu sắp xếp theo thời gian
        ✔️ **Duplicates removed** - Loại bỏ dòng trùng lặp
        """)
        
        st.write("### 📊 5. So sánh dữ liệu trước & sau làm sạch")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows (Before)", original_df.shape[0])
        with col2:
            st.metric("Rows (After)", df.shape[0])
        with col3:
            st.metric("Rows Removed", original_df.shape[0] - df.shape[0])
        
        st.write("### 📝 Dữ liệu mẫu sau làm sạch (Sample Cleaned Data)")
        st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# TAB 3: B3 – EXPLORATORY DATA ANALYSIS (KHAI PHÁ DỮ LIỆU)
# ==========================================================
with tab3:
    st.header("B3 – Khai phá dữ liệu (Exploratory Data Analysis)")
    
    # ===== UNIVARIATE ANALYSIS =====
    st.subheader("📊 Phân tích đơn biến (Univariate Analysis)")
    
    col_univariate = st.selectbox(
        "Chọn cột định lượng để phân tích",
        quantitative_cols,
        key="univariate"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col_univariate], kde=True, ax=ax, color='steelblue', bins=30)
        ax.set_title(f"Distribution of {col_univariate}")
        ax.set_xlabel(col_univariate)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, y=col_univariate, ax=ax, color='lightcoral')
        ax.set_title(f"Box Plot of {col_univariate}")
        st.pyplot(fig)
    
    # Statistical summary
    st.write(f"### 📈 Thống kê mô tả: {col_univariate}")
    stats_data = {
        "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max", "Range", "Skewness"],
        "Value": [
            df[col_univariate].count(),
            round(df[col_univariate].mean(), 2),
            round(df[col_univariate].std(), 2),
            round(df[col_univariate].min(), 2),
            round(df[col_univariate].quantile(0.25), 2),
            round(df[col_univariate].median(), 2),
            round(df[col_univariate].quantile(0.75), 2),
            round(df[col_univariate].max(), 2),
            round(df[col_univariate].max() - df[col_univariate].min(), 2),
            round(df[col_univariate].skew(), 2),
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # ===== BIVARIATE ANALYSIS =====
    st.subheader("📈 Phân tích đa biến (Bivariate Analysis)")
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Chọn trục X", quantitative_cols, index=0, key="x_axis")
    with col2:
        y_col = st.selectbox("Chọn trục Y", quantitative_cols, index=1 if len(quantitative_cols) > 1 else 0, key="y_axis")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, alpha=0.6, color='steelblue', s=50)
    # Thêm trendline
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), "r--", linewidth=2, label='Trend')
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    ax.legend()
    st.pyplot(fig)
    
    # Correlation coefficient
    corr_coef = df[x_col].corr(df[y_col])
    st.write(f"**Hệ số tương quan Pearson:** {corr_coef:.4f}")
    
    # ===== TIME SERIES ANALYSIS =====
    st.subheader("⏰ Phân tích chuỗi thời gian (Time Series Analysis)")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close/Last"], linewidth=2, color='steelblue', label='Close Price')
    ax.fill_between(df["Date"], df["Low"], df["High"], alpha=0.2, color='lightblue', label='High-Low Range')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/oz)")
    ax.set_title("Xu hướng giá vàng (Gold Price Trend)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Thống kê chuỗi thời gian
    st.write("### 📊 Thống kê chuỗi thời gian:")
    time_stats = {
        "Metric": ["Avg Close Price", "Max Close Price", "Min Close Price", "Price Range", "Volatility (Std)"],
        "Value": [
            f"${df['Close/Last'].mean():.2f}",
            f"${df['Close/Last'].max():.2f}",
            f"${df['Close/Last'].min():.2f}",
            f"${df['Close/Last'].max() - df['Close/Last'].min():.2f}",
            f"${df['Close/Last'].std():.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(time_stats), use_container_width=True)
    
    # ===== VOLUME ANALYSIS =====
    st.subheader("📊 Phân tích khối lượng giao dịch (Volume Analysis)")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["Date"], df["Volume"], color='steelblue', alpha=0.7, width=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.set_title("Khối lượng giao dịch (Trading Volume Over Time)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Volume statistics
    st.write("### 📊 Thống kê khối lượng:")
    volume_stats = {
        "Metric": ["Avg Volume", "Max Volume", "Min Volume", "Total Volume"],
        "Value": [
            f"{df['Volume'].mean():.0f}",
            f"{df['Volume'].max():.0f}",
            f"{df['Volume'].min():.0f}",
            f"{df['Volume'].sum():.0f}"
        ]
    }
    st.dataframe(pd.DataFrame(volume_stats), use_container_width=True)

# ==========================================================
# TAB 4: B4 – CORRELATION & DIMENSIONALITY REDUCTION
# ==========================================================
with tab4:
    st.header("B4 – Ma trận tương quan & Giảm chiều (Correlation & PCA)")
    
    # ===== CORRELATION MATRIX =====
    st.subheader("🔗 Ma trận tương quan (Correlation Matrix)")
    
    corr_matrix = df[quantitative_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                square=True, ax=ax, fmt=".2f", cbar_kws={'label': 'Correlation'})
    ax.set_title("Correlation Matrix of Quantitative Variables")
    st.pyplot(fig)
    
    st.write("### 🧠 Phân tích & Lập luận Giữ/Bỏ cột (Analysis & Column Selection Reasoning)")
    
    # Tính tương quan cao
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append({
                    "Column 1": corr_matrix.columns[i],
                    "Column 2": corr_matrix.columns[j],
                    "Correlation": round(corr_matrix.iloc[i, j], 4)
                })
    
    if high_corr_pairs:
        st.write("**Các cột có tương quan rất cao (> 0.95):**")
        st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
    
    st.write("""
    **Kết luận & Quyết định:**
    
    1. **Open, High, Low, Close/Last** có tương quan rất cao (> 0.95)
       - ❌ Giữ tất cả 4 cột là dư thừa
       - ✅ **Giữ:** `Close/Last` (giá đóng cửa - chỉ báo chính)
       - ❌ **Bỏ:** `Open`, `High`, `Low` (có thể suy ra từ Close/Last)
    
    2. **Volume** tương quan yếu với các cột giá
       - ✅ **Giữ:** `Volume` (thông tin độc lập, hữu ích)
       - Khối lượng giao dịch phản ánh mức độ quan tâm của thị trường
    
    **Kết quả cuối cùng:**
    - ✅ Giữ: `Close/Last`, `Volume`
    - ❌ Bỏ: `Open`, `High`, `Low` (dư thừa, tương quan cao)
    """)
    
    # ===== DIMENSIONALITY REDUCTION WITH PCA =====
    st.subheader("🎯 Giảm chiều dữ liệu (PCA - Principal Component Analysis)")
    
    # Prepare data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[quantitative_cols])
    
    # Apply PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Show cumulative variance explained
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance by PCA Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(df)), 
                        cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA Projection of Gold Stock Data (2D)")
    plt.colorbar(scatter, ax=ax, label='Time Index')
    st.pyplot(fig)
    
    st.write("### 📊 PCA Chi tiết (PCA Detailed Results)")
    pca_stats = {
        "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained Variance %": [f"{v:.2%}" for v in pca.explained_variance_ratio_],
        "Cumulative Variance %": [f"{c:.2%}" for c in np.cumsum(pca.explained_variance_ratio_)]
    }
    st.dataframe(pd.DataFrame(pca_stats), use_container_width=True)
    
    st.write(f"""
    **Nhận xét:**
    - PC1 explains: **{pca.explained_variance_ratio_[0]:.2%}** của phương sai
    - PC2 explains: **{pca.explained_variance_ratio_[1]:.2%}** của phương sai
    - Tổng cộng: **{sum(pca.explained_variance_ratio_):.2%}** phương sai được giải thích
    
    **Kết luận:** 2 thành phần chính giải thích **{sum(pca.explained_variance_ratio_):.2%}** phương sai,
    tức là giảm được từ {len(quantitative_cols)} chiều xuống 2 chiều mà vẫn giữ lại hầu hết thông tin.
    """)
    
    # Feature loadings
    st.write("### 🔍 Đóng góp của từng biến vào PC (Feature Loadings)")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(pca.components_))],
        index=quantitative_cols
    )
    st.dataframe(loadings_df, use_container_width=True)

# ==========================================================
# TAB 5: B5 – MACHINE LEARNING MODEL & INTERACTIVE VISUALIZATION
# ==========================================================
with tab5:
    st.header("B5 – Mô hình ML & Trực quan hóa tương tác (Model & Visualization)")
    
    st.info("⚠️ **Lưu ý:** Mô hình K-Means dưới đây chỉ mang tính minh họa để phân cụm dữ liệu. Mục đích là làm rõ cấu trúc dữ liệu, không đánh giá cao hiệu suất dự báo.")
    
    # ===== KMEANS CLUSTERING =====
    st.subheader("🎯 Phân cụm dữ liệu (K-Means Clustering)")
    
    k = st.slider(
        "Chọn số cụm (Select number of clusters)",
        min_value=2,
        max_value=6,
        value=3,
        step=1
    )
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[quantitative_cols])
    
    # ===== VISUALIZATION 1: SCATTER PLOTS =====
    st.write("### 📊 Biểu đồ phân cụm (Cluster Scatter Plots)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["Open"], df["Close/Last"], 
                           c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                  label='Centroids', zorder=5)
        ax.set_xlabel("Open Price ($)", fontsize=11)
        ax.set_ylabel("Close Price ($)", fontsize=11)
        ax.set_title(f"K-Means Clustering (Open vs Close) - K={k}")
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["Low"], df["High"],
                           c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
                  c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                  label='Centroids', zorder=5)
        ax.set_xlabel("Low Price ($)", fontsize=11)
        ax.set_ylabel("High Price ($)", fontsize=11)
        ax.set_title(f"K-Means Clustering (Low vs High) - K={k}")
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
           

    
    # ===== VISUALIZATION 2: TIME SERIES WITH CLUSTERS =====
    st.write("### ⏰ Phân bố cụm theo thời gian (Cluster Distribution Over Time)")
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, k))
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(cluster_data["Date"], cluster_data["Close/Last"],
                  label=f"Cluster {cluster}", alpha=0.6, s=40, color=colors[cluster])
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price ($)", fontsize=11)
    ax.set_title(f"Gold Price with K-Means Clusters (K={k})")
    ax.legend(loc='best')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ===== CLUSTER STATISTICS =====
    st.write("### 📈 Thống kê chi tiết từng cụm (Cluster Statistics)")
    cluster_stats = df.groupby("Cluster")[quantitative_cols].agg(['mean', 'min', 'max', 'std'])
    st.dataframe(cluster_stats, use_container_width=True)
    
    # ===== CLUSTER SIZES =====
    st.write("### 📊 Kích thước cụm (Cluster Sizes)")
    cluster_sizes = df["Cluster"].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Number of Data Points", fontsize=11)
        ax.set_title(f"Cluster Size Distribution (K={k})")
        ax.set_xticks(range(k))
        for i, v in enumerate(cluster_sizes.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(cluster_sizes.values, labels=[f"Cluster {i}\n({v} points)" for i, v in enumerate(cluster_sizes.values)],
               colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.05]*k)
        ax.set_title(f"Cluster Distribution Percentage (K={k})")
        st.pyplot(fig)
    
    # ===== CLUSTER CHARACTERISTICS =====
    st.write("### 🔍 Đặc điểm của từng cụm (Cluster Characteristics)")
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        st.write(f"**Cluster {cluster}:** {len(cluster_data)} điểm dữ liệu")
        
        char_text = f"- **Giá Close trung bình:** ${cluster_data['Close/Last'].mean():.2f} (Range: ${cluster_data['Close/Last'].min():.2f} - ${cluster_data['Close/Last'].max():.2f})\n"
        char_text += f"- **Khối lượng trung bình:** {cluster_data['Volume'].mean():.0f}\n"
        char_text += f"- **Giai đoạn thời gian:** {cluster_data['Date'].min().date()} → {cluster_data['Date'].max().date()}"
        st.write(char_text)
    
    # ===== SAMPLE DATA WITH CLUSTERS =====
    st.write("### 📝 Dữ liệu mẫu sau phân cụm (Sample Data with Clusters)")
    display_cols = ["Date", "Open", "High", "Low", "Close/Last", "Volume", "Cluster"]
    st.dataframe(df[display_cols].head(20), use_container_width=True)
    
    # ===== MODEL EVALUATION =====
    st.write("### 📊 Đánh giá mô hình K-Means (Model Evaluation)")
    
    # Calculate inertia and silhouette score
    inertia = kmeans.inertia_
    silhouette = silhouette_score(df[quantitative_cols], df["Cluster"])
    davies_bouldin = davies_bouldin_score(df[quantitative_cols], df["Cluster"])
    calinski = calinski_harabasz_score(df[quantitative_cols], df["Cluster"])
    
    eval_metrics = {
        "Metric": ["Inertia", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
        "Value": [
            f"{inertia:.2f}",
            f"{silhouette:.4f}",
            f"{davies_bouldin:.4f}",
            f"{calinski:.2f}"
        ],
        "Interpretation": [
            "Sum of squared distances (Lower is better)",
            "Clustering quality (-1 to 1, Higher is better)",
            "Cluster separation (Lower is better)",
            "Cluster density (Higher is better)"
        ]
    }
    st.dataframe(pd.DataFrame(eval_metrics), use_container_width=True)
    
    st.write(f"""
    **Nhận xét:**
    - **Silhouette Score = {silhouette:.4f}**: {'Tốt ✓' if silhouette > 0.5 else 'Trung bình ⚠' if silhouette > 0.3 else 'Cần cải thiện ✗'}
    - Mô hình phân cụm giúp hiểu rõ cấu trúc dữ liệu giá vàng
    - Các cụm có thể đại diện cho các giai đoạn hoặc xu hướng giá khác nhau
    """)
    
    # ===========================
    # LINEAR REGRESSION - FORECAST (NÂNG CẤP ✅)
    # ===========================
    st.markdown("---")
    st.subheader("📈 Dự đoán giá vàng (Linear Regression + Regularization)")
    
    # 1) Chọn horizon + tỷ lệ test
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox("Chọn bước dự đoán (t+h, theo NGÀY)", [1, 7, 14, 30, 60, 90, 180, 252], index=3)
    with col2:
        test_ratio = st.slider("Tỷ lệ test (time split)", 0.15, 0.35, 0.25, 0.05)
    with col3:
        reg_type = st.radio("Regularization:", ["Ridge (L2)", "Lasso (L1)", "ElasticNet"], index=0)
    
    # 2) Tạo dataset supervised
    lr_data, feat_cols = make_lr_dataset(df, horizon=horizon)
    
    # ===== LỌC FEATURES NHIỄU =====
    feat_corr = lr_data[feat_cols + ["y"]].corr()["y"].drop("y")
    strong_features = feat_corr[feat_corr.abs() > 0.01].index.tolist()
    strong_features = [f for f in strong_features if f in feat_cols]
    
    if len(strong_features) < 3:
        st.warning("Quá ít feature sau lọc, dùng tất cả features")
        feat_cols_selected = feat_cols
    else:
        feat_cols_selected = strong_features
    
    st.write(f"**Features sau lọc:** {len(feat_cols_selected)}/{len(feat_cols)} (loại bỏ noise)")
    
    # Time split
    train_df, test_df = time_split(lr_data, test_ratio=test_ratio)
    X_train, y_train = train_df[feat_cols_selected], train_df["y"]
    X_test, y_test = test_df[feat_cols_selected], test_df["y"]
    
    # ===== CHUẨN HÓA DỮ LIỆU =====
    scaler_robust = RobustScaler()
    X_train_scaled = scaler_robust.fit_transform(X_train)
    X_test_scaled = scaler_robust.transform(X_test)
    
    # ===== REGULARIZATION + TRAINING =====
    alpha = st.slider("Độ mạnh regularization (α)", 0.001, 1.0, 0.1, 0.01)
    
    if reg_type == "Ridge (L2)":
        lr = Ridge(alpha=alpha)
    elif reg_type == "Lasso (L1)":
        lr = Lasso(alpha=alpha, max_iter=10000)
    else:
        lr = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    
    lr.fit(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    y_pred = lr.predict(X_test_scaled)
    
    # ===== METRICS =====
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = lr.score(X_test_scaled, y_test)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("R² Score", f"{r2:.4f}")
    c4.metric("Test size", f"{len(test_df)}")
    c5.metric("Horizon", f"t+{horizon}d")
    
    # ===== ACTUAL vs PREDICTED =====
    st.write("### 📊 Actual vs Predicted Return (trên tập test)")
    
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_df["Date"].values, y_test.values, label="Actual Return", linewidth=2, color='blue')
    ax.plot(test_df["Date"].values, y_pred, label="Predicted Return", linewidth=2, color='red', alpha=0.7)
    ax.fill_between(test_df["Date"].values, y_test.values, y_pred, alpha=0.2, color='gray')
    ax.set_title(f"Actual vs Predicted Return (t+{horizon} ngày)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (% change)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ===== RESIDUAL ANALYSIS =====
    st.write("### 🔍 Phân tích sai số (Residuals)")
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    
    axes[0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Distribution of Residuals")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Residual Plot")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Residual")
    
    st.pyplot(fig)
    
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    st.write(f"**Residual Mean:** {residual_mean:.6f} (nên ≈ 0)")
    st.write(f"**Residual Std:** {residual_std:.6f}")
    
    # ===== FEATURE COEFFICIENTS =====
    st.write("### 🧠 Đóng góp của feature (Top 15 Coefficients)")
    coef_df = pd.DataFrame({"Feature": feat_cols_selected, "Coef": lr.coef_})
    coef_df["AbsCoef"] = coef_df["Coef"].abs()
    coef_df = coef_df.sort_values("AbsCoef", ascending=False)
    
    st.dataframe(coef_df.drop(columns=["AbsCoef"]).head(15), use_container_width=True)
    
    # ===========================
    # FORECAST TƯƠNG LAI (NÂNG CẤP) ✅✅✅
    # ===========================
    st.markdown("---")
    st.subheader("🔮 Dự báo tương lai (252 ngày, với khoảng tin cậy)")
    
    do_forecast = st.checkbox("Bật dự báo tương lai (multi-step forecast)", value=True)
    
    if do_forecast:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_steps = st.slider("Số ngày dự báo", 30, 252, 90, 5)
        with col2:
            confidence_level = st.select_slider("Khoảng tin cậy", [0.68, 0.85, 0.95], value=0.95)
        with col3:
            use_business_days = st.checkbox("Dùng Business days", value=True)
        
        # ===== HÀM CHÍNH: TẠYO FEATURES TỪ LỊCH SỬ =====
        def build_features_from_history_scaled(hist_df: pd.DataFrame, scaler_obj, feat_cols_list) -> np.ndarray:
            """Xây dựng 1 dòng feature từ lịch sử, trả về array đã chuẩn hóa"""
            row = {}
            
            # lags
            for lag in [1, 2, 3, 5, 7, 14, 30]:
                if len(hist_df) >= lag:
                    row[f"close_lag_{lag}"] = hist_df["Close/Last"].iloc[-lag]
                else:
                    row[f"close_lag_{lag}"] = np.nan
            
            # rolling
            row["ma_7"] = hist_df["Close/Last"].rolling(7).mean().iloc[-1] if len(hist_df) >= 7 else np.nan
            row["ma_14"] = hist_df["Close/Last"].rolling(14).mean().iloc[-1] if len(hist_df) >= 14 else np.nan
            row["ma_20"] = hist_df["Close/Last"].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else np.nan
            row["ma_30"] = hist_df["Close/Last"].rolling(30).mean().iloc[-1] if len(hist_df) >= 30 else np.nan
            row["ma_60"] = hist_df["Close/Last"].rolling(60).mean().iloc[-1] if len(hist_df) >= 60 else np.nan
            row["ma_200"] = hist_df["Close/Last"].rolling(200).mean().iloc[-1] if len(hist_df) >= 200 else np.nan
            
            row["std_7"] = hist_df["Close/Last"].rolling(7).std().iloc[-1] if len(hist_df) >= 7 else np.nan
            row["std_14"] = hist_df["Close/Last"].rolling(14).std().iloc[-1] if len(hist_df) >= 14 else np.nan
            row["std_20"] = hist_df["Close/Last"].rolling(20).std().iloc[-1] if len(hist_df) >= 20 else np.nan
            
            # momentum
            row["momentum_7"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-8] if len(hist_df) >= 8 else np.nan
            row["momentum_14"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-15] if len(hist_df) >= 15 else np.nan
            row["momentum_30"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-31] if len(hist_df) >= 31 else np.nan
            
            # ROC
            row["roc_7"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-8]) / hist_df["Close/Last"].iloc[-8] if len(hist_df) >= 8 else np.nan
            row["roc_14"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-15]) / hist_df["Close/Last"].iloc[-15] if len(hist_df) >= 15 else np.nan
            row["roc_30"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-31]) / hist_df["Close/Last"].iloc[-31] if len(hist_df) >= 31 else np.nan
            
            # RSI
            delta = hist_df["Close/Last"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            row["rsi_14"] = rsi.iloc[-1] if len(rsi) > 0 else np.nan
            
            # Bollinger Bands
            ma_20 = hist_df["Close/Last"].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else np.nan
            std_20 = hist_df["Close/Last"].rolling(20).std().iloc[-1] if len(hist_df) >= 20 else np.nan
            if pd.notna(ma_20) and pd.notna(std_20):
                bb_upper = ma_20 + 2 * std_20
                bb_lower = ma_20 - 2 * std_20
                row["bb_position"] = (hist_df["Close/Last"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
            else:
                row["bb_position"] = np.nan
            row["bb_upper"] = bb_upper if pd.notna(bb_upper) else np.nan
            row["bb_lower"] = bb_lower if pd.notna(bb_lower) else np.nan
            
            # Volume
            if "Volume" in hist_df.columns:
                row["vol_lag_1"] = hist_df["Volume"].iloc[-1]
                row["vol_ma_7"] = hist_df["Volume"].rolling(7).mean().iloc[-1] if len(hist_df) >= 7 else np.nan
                row["vol_ma_30"] = hist_df["Volume"].rolling(30).mean().iloc[-1] if len(hist_df) >= 30 else np.nan
                row["vol_std"] = hist_df["Volume"].rolling(7).std().iloc[-1] if len(hist_df) >= 7 else np.nan
                row["price_vol_trend"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-2]) / hist_df["Close/Last"].iloc[-2] * hist_df["Volume"].iloc[-1] if len(hist_df) >= 2 else np.nan
            
            # Chuyển thành DataFrame, chọn cột cần thiết, chuẩn hóa
            df_row = pd.DataFrame([row])
            df_row = df_row[feat_cols_list]
            
            x_scaled = scaler_obj.transform(df_row)
            return x_scaled.flatten()
        
        # ===== RECURSIVE FORECAST =====
        hist = df[["Date", "Close/Last"] + (["Volume"] if "Volume" in df.columns else [])].copy()
        hist = hist.sort_values("Date").reset_index(drop=True)
        
        last_date = hist["Date"].iloc[-1]
        last_close = hist["Close/Last"].iloc[-1]
        
        if use_business_days:
            future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_steps)
        else:
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_steps, freq="D")
        
        preds = []
        preds_std = []
        temp_hist = hist.copy()
        
        sigma = residual_std
        
        for i, dt in enumerate(future_dates):
            try:
                x_next = build_features_from_history_scaled(temp_hist, scaler_robust, feat_cols_selected)
                
                if np.isnan(x_next).any():
                    preds.append(np.nan)
                    preds_std.append(np.nan)
                else:
                    y_next = lr.predict(x_next.reshape(1, -1))[0]
                    preds.append(y_next)
                    preds_std.append(sigma)
                    
                    # Cập nhật lịch sử
                    new_close = temp_hist["Close/Last"].iloc[-1] * (1 + y_next)
                    new_row = {"Date": pd.Timestamp(dt), "Close/Last": new_close}
                    if "Volume" in temp_hist.columns:
                        new_row["Volume"] = temp_hist["Volume"].iloc[-1]
                    temp_hist = pd.concat([temp_hist, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                st.warning(f"Lỗi tại bước {i}: {e}")
                preds.append(np.nan)
                preds_std.append(np.nan)
        
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast_Return": preds,
            "Forecast_Std": preds_std
        })
        forecast_df = forecast_df.dropna().reset_index(drop=True)
        
        # Tính Confidence Interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

# ==========================================================
# DATA MINING PROJECT - GOLD PRICE DATA
# Dataset: goldstock v2.csv
# B1 → B5 (EDA-focused, model minh họa, có GUI)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats


plt.style.use('ggplot')

# ==========================================================
# LOAD DATA - B0: DATASET OVERVIEW
# ==========================================================
original_df = pd.read_csv("goldstock v2.csv")

# Xóa cột index không cần thiết
if "Unnamed: 0" in original_df.columns:
    original_df.drop(columns=["Unnamed: 0"], inplace=True)

# Chuyển Date sang datetime
original_df["Date"] = pd.to_datetime(original_df["Date"])

# Sắp xếp theo thời gian
original_df.sort_values(by="Date", inplace=True)
original_df.reset_index(drop=True, inplace=True)
# ===== Convert các cột số về numeric (phòng trường hợp có '$' và ',' ) =====
def to_numeric_clean(s: pd.Series):
    if s.dtype == "O":  # object/string
        s = (s.astype(str)
               .str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(s, errors="coerce")

for col in ["Open", "High", "Low", "Close/Last", "Volume"]:
    if col in original_df.columns:
        original_df[col] = to_numeric_clean(original_df[col])

# ===== CLEANING BỔ SUNG SAU CONVERT NUMERIC =====

# Drop Date lỗi
original_df = original_df.dropna(subset=["Date"]).copy()

# Drop NaN phát sinh sau khi convert numeric
num_cols = ["Open", "High", "Low", "Close/Last", "Volume"]
num_cols = [c for c in num_cols if c in original_df.columns]
original_df = original_df.dropna(subset=num_cols).copy()

# Volume phải không âm
if "Volume" in original_df.columns:
    original_df = original_df[original_df["Volume"] >= 0].copy()


# ==========================================================
# B2 – DATA CLEANING (TIỀN XỬ LÝ DỮ LIỆU)
# ==========================================================
# Clone một bản sao để giữ nguyên dữ liệu gốc
df = original_df.copy()

# Kiểm tra dữ liệu thiếu
missing_data = df.isnull().sum() * 100 / df.shape[0]

# Xóa duplicate
# Remove duplicate theo Date (chuẩn time-series)
df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)


# Kiểm tra logic giá (High >= Open, Close, Low; Low <= Open, Close)
df = df[
    (df["High"] >= df["Open"]) &
    (df["High"] >= df["Close/Last"]) &
    (df["High"] >= df["Low"]) &
    (df["Low"] <= df["Open"]) &
    (df["Low"] <= df["Close/Last"])
]
# ===== VALIDATION BỔ SUNG =====

# Giá phải dương
for c in ["Open", "High", "Low", "Close/Last"]:
    df = df[df[c] > 0]

# Close và Open phải nằm trong [Low, High]
df = df[
    (df["Close/Last"] >= df["Low"]) &
    (df["Close/Last"] <= df["High"]) &
    (df["Open"] >= df["Low"]) &
    (df["Open"] <= df["High"])
]

df.reset_index(drop=True, inplace=True)

df.reset_index(drop=True, inplace=True)

# ==========================================================
# B1 – MÔ TẢ DỮ LIỆU (DATA OVERVIEW)
# ==========================================================
quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
qualitative_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# ==========================================================
# B5+ (SUPERVISED) – LINEAR REGRESSION FORECAST (NÂNG CẤP)
# ==========================================================
def make_lr_dataset(df_in: pd.DataFrame, horizon: int = 1):
    """
    horizon=1: dự đoán Close(t+1)
    horizon=7: dự đoán Close(t+7)
    
    ✅ CẢI THIỆN: Thêm nhiều feature chất lượng cao (momentum, ROC, RSI, Bollinger Bands)
    """
    d = df_in.copy()

    # Target: future return (thay vì absolute price - dễ scale hơn)
    d["y"] = d["Close/Last"].shift(-horizon) / d["Close/Last"] - 1

    # ===== LAG FEATURES (giữ nguyên) =====
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        d[f"close_lag_{lag}"] = d["Close/Last"].shift(lag)

    # ===== ROLLING FEATURES (NÂNG CẤP) =====
    d["ma_7"] = d["Close/Last"].rolling(7).mean()
    d["ma_14"] = d["Close/Last"].rolling(14).mean()
    d["ma_20"] = d["Close/Last"].rolling(20).mean()
    d["ma_30"] = d["Close/Last"].rolling(30).mean()
    d["ma_60"] = d["Close/Last"].rolling(60).mean()
    d["ma_200"] = d["Close/Last"].rolling(200).mean()
    
    d["std_7"] = d["Close/Last"].rolling(7).std()
    d["std_14"] = d["Close/Last"].rolling(14).std()
    d["std_20"] = d["Close/Last"].rolling(20).std()
    
    # ===== MOMENTUM & TREND FEATURES (TỐI QUAN TRỌNG - CẢI THIỆN) =====
    d["momentum_7"] = d["Close/Last"] - d["Close/Last"].shift(7)
    d["momentum_14"] = d["Close/Last"] - d["Close/Last"].shift(14)
    d["momentum_30"] = d["Close/Last"] - d["Close/Last"].shift(30)
    
    # Rate of change (tốc độ thay đổi %)
    d["roc_7"] = (d["Close/Last"] - d["Close/Last"].shift(7)) / d["Close/Last"].shift(7)
    d["roc_14"] = (d["Close/Last"] - d["Close/Last"].shift(14)) / d["Close/Last"].shift(14)
    d["roc_30"] = (d["Close/Last"] - d["Close/Last"].shift(30)) / d["Close/Last"].shift(30)
    
    # RSI-like oscillator (0-100)
    delta = d["Close/Last"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (độ lệch so với MA)
    d["bb_upper"] = d["ma_20"] + 2 * d["std_20"]
    d["bb_lower"] = d["ma_20"] - 2 * d["std_20"]
    d["bb_position"] = (d["Close/Last"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])
    
    # ===== VOLUME FEATURES (CẬP NHẬT) =====
    if "Volume" in d.columns:
        d["vol_lag_1"] = d["Volume"].shift(1)
        d["vol_ma_7"] = d["Volume"].rolling(7).mean()
        d["vol_ma_30"] = d["Volume"].rolling(30).mean()
        d["vol_std"] = d["Volume"].rolling(7).std()
        
        # Volume-Price Trend
        d["price_vol_trend"] = (d["Close/Last"] - d["Close/Last"].shift(1)) / d["Close/Last"].shift(1) * d["Volume"]

    # ===== LOẠI BỎ NaN & CHỌN FEATURES =====
    d = d.dropna().reset_index(drop=True)

    feature_cols = [c for c in d.columns if c.startswith(("close_lag_", "ma_", "std_", "vol_", "momentum_", "roc_", "rsi_", "bb_", "price_"))]
    return d, feature_cols


def time_split(d: pd.DataFrame, test_ratio: float = 0.2):
    n = len(d)
    test_n = max(1, int(n * test_ratio))
    train = d.iloc[:-test_n].copy()
    test = d.iloc[-test_n:].copy()
    return train, test


# ...existing code...

# ==========================================================
# STREAMLIT GUI CONFIGURATION
# ==========================================================
st.set_page_config(page_title="Gold Price Data Mining", layout="wide")
st.title("📊 Gold Price Data Mining Project")
st.markdown("---")

# Create tabs for each phase
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "B1 - Data Overview",
    "B2 - Data Cleaning",
    "B3 - Exploratory Analysis",
    "B4 - Correlation & Dimensionality",
    "B5 - Model & Visualization"
])

# ==========================================================
# TAB 1: B1 – DATA OVERVIEW (MÔ TẢ DỮ LIỆU)
# ==========================================================
with tab1:
    st.header("B1 – Mô tả dữ liệu (Dataset Overview)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    
    st.write("### 📋 Danh sách các cột (Dataset Columns)")
    st.write(df.columns.tolist())
    
    st.write("### 📊 Phân loại dữ liệu (Data Types Classification)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Định lượng (Quantitative):**")
        st.write(f"- Số cột: **{len(quantitative_cols)}**")
        for col in quantitative_cols:
            st.write(f"  - `{col}` ({df[col].dtype})")
    
    with col2:
        st.write("**Định tính (Qualitative):**")
        st.write(f"- Số cột: **{len(qualitative_cols)}**")
        for col in qualitative_cols:
            st.write(f"  - `{col}` ({df[col].dtype})")
    
    st.write("### 📈 Thống kê mô tả chi tiết (Descriptive Statistics)")
    st.dataframe(df[quantitative_cols].describe(), use_container_width=True)
    
    st.write("### 🔍 Thông tin chi tiết các cột (Detailed Column Info)")
    info_data = {
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.count(),
        "Null Count": df.isnull().sum(),
        "Min": [df[col].min() if col in quantitative_cols else "N/A" for col in df.columns],
        "Max": [df[col].max() if col in quantitative_cols else "N/A" for col in df.columns],
    }
    st.dataframe(pd.DataFrame(info_data), use_container_width=True)
    
    st.write("### 📝 Dữ liệu mẫu (Sample Data)")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# TAB 2: B2 – DATA CLEANING (TIỀN XỬ LÝ)
# ==========================================================
with tab2:
    st.header("B2 – Tiền xử lý dữ liệu (Data Cleaning)")
    
    st.write("### 🔍 1. Kiểm tra dữ liệu thiếu (Missing Data)")
    missing_count = original_df.isnull().sum()
    missing_percent = (missing_count / len(original_df)) * 100
    missing_df = pd.DataFrame({
        "Column": missing_count.index,
        "Missing Count": missing_count.values,
        "Missing %": missing_percent.values
    })
    st.dataframe(missing_df[missing_df["Missing Count"] > 0] if missing_df["Missing Count"].sum() > 0 
                 else pd.DataFrame({"Status": ["✅ No missing data found"]}), use_container_width=True)
    
    st.write("### 🔄 2. Kiểm tra dữ liệu trùng lặp (Duplicate Data)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"❌ **Trước xử lý:** {original_df.duplicated().sum()} dòng trùng lặp")
    with col2:
        st.write(f"✅ **Sau xử lý:** {df.duplicated().sum()} dòng trùng lặp")
    
    st.write("### 🚨 3. Phát hiện Noise & Outliers")
    
    st.write("### 📌 Outlier theo Return (khuyến nghị cho dữ liệu tài chính)")

    # 1) Check column
    if "Close/Last" not in df.columns:
        st.error("Không tìm thấy cột 'Close/Last' trong df.")
    else:
        # 2) Ép kiểu numeric an toàn
        close = pd.to_numeric(df["Close/Last"], errors="coerce")

        # 3) Tính return + làm sạch NaN/Inf
        ret = close.pct_change()
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()

        # 4) Nếu dữ liệu quá ít thì cảnh báo
        if len(ret) < 30:
            st.warning(f"Return hợp lệ quá ít ({len(ret)} điểm) nên thống kê outlier không đáng tin.")
        else:
            q1, q3 = ret.quantile(0.25), ret.quantile(0.75)
            iqr = q3 - q1
            lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            out_ret = ((ret < lb) | (ret > ub)).sum()
            st.write(f"- Số outlier theo return (IQR): **{out_ret}**")
            st.write(f"- Ngưỡng IQR: **[{lb:.4f}, {ub:.4f}]**")

            st.info("Outlier giá KHÔNG nên xóa vì phản ánh biến động thị trường. Nếu cần ổn định mô hình, cân nhắc xử lý outlier trên return/feature.")
        
        # Kiểm tra outliers bằng IQR
        st.write("**Phương pháp IQR (Interquartile Range):**")
        outlier_summary = []
        
        for col in quantitative_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_summary.append({
                "Column": col,
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "Lower Bound": round(lower_bound, 2),
                "Upper Bound": round(upper_bound, 2),
                "Outlier Count": outlier_count
            })
        
        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True)
        
        # Visualize outliers
        fig, axes = plt.subplots(len(quantitative_cols), 1, figsize=(10, 3*len(quantitative_cols)))
        if len(quantitative_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(quantitative_cols):
            sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
            axes[idx].set_title(f"Outlier Detection: {col}")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### ✅ 4. Các quy tắc xác thực dữ liệu (Data Validation Rules)")
        st.write("""
        ✔️ **High >= Open, Close/Last, Low** - Giá cao nhất >= tất cả các mức giá khác
        ✔️ **Low <= Open, Close/Last** - Giá thấp nhất <= tất cả các mức giá khác
        ✔️ **Volume >= 0** - Khối lượng không âm
        ✔️ **Date sorted in ascending order** - Dữ liệu sắp xếp theo thời gian
        ✔️ **Duplicates removed** - Loại bỏ dòng trùng lặp
        """)
        
        st.write("### 📊 5. So sánh dữ liệu trước & sau làm sạch")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows (Before)", original_df.shape[0])
        with col2:
            st.metric("Rows (After)", df.shape[0])
        with col3:
            st.metric("Rows Removed", original_df.shape[0] - df.shape[0])
        
        st.write("### 📝 Dữ liệu mẫu sau làm sạch (Sample Cleaned Data)")
        st.dataframe(df.head(10), use_container_width=True)

# ==========================================================
# TAB 3: B3 – EXPLORATORY DATA ANALYSIS (KHAI PHÁ DỮ LIỆU)
# ==========================================================
with tab3:
    st.header("B3 – Khai phá dữ liệu (Exploratory Data Analysis)")
    
    # ===== UNIVARIATE ANALYSIS =====
    st.subheader("📊 Phân tích đơn biến (Univariate Analysis)")
    
    col_univariate = st.selectbox(
        "Chọn cột định lượng để phân tích",
        quantitative_cols,
        key="univariate"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram with KDE
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[col_univariate], kde=True, ax=ax, color='steelblue', bins=30)
        ax.set_title(f"Distribution of {col_univariate}")
        ax.set_xlabel(col_univariate)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with col2:
        # Box plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, y=col_univariate, ax=ax, color='lightcoral')
        ax.set_title(f"Box Plot of {col_univariate}")
        st.pyplot(fig)
    
    # Statistical summary
    st.write(f"### 📈 Thống kê mô tả: {col_univariate}")
    stats_data = {
        "Metric": ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max", "Range", "Skewness"],
        "Value": [
            df[col_univariate].count(),
            round(df[col_univariate].mean(), 2),
            round(df[col_univariate].std(), 2),
            round(df[col_univariate].min(), 2),
            round(df[col_univariate].quantile(0.25), 2),
            round(df[col_univariate].median(), 2),
            round(df[col_univariate].quantile(0.75), 2),
            round(df[col_univariate].max(), 2),
            round(df[col_univariate].max() - df[col_univariate].min(), 2),
            round(df[col_univariate].skew(), 2),
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # ===== BIVARIATE ANALYSIS =====
    st.subheader("📈 Phân tích đa biến (Bivariate Analysis)")
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Chọn trục X", quantitative_cols, index=0, key="x_axis")
    with col2:
        y_col = st.selectbox("Chọn trục Y", quantitative_cols, index=1 if len(quantitative_cols) > 1 else 0, key="y_axis")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, alpha=0.6, color='steelblue', s=50)
    # Thêm trendline
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col].sort_values(), p(df[x_col].sort_values()), "r--", linewidth=2, label='Trend')
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
    ax.legend()
    st.pyplot(fig)
    
    # Correlation coefficient
    corr_coef = df[x_col].corr(df[y_col])
    st.write(f"**Hệ số tương quan Pearson:** {corr_coef:.4f}")
    
    # ===== TIME SERIES ANALYSIS =====
    st.subheader("⏰ Phân tích chuỗi thời gian (Time Series Analysis)")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close/Last"], linewidth=2, color='steelblue', label='Close Price')
    ax.fill_between(df["Date"], df["Low"], df["High"], alpha=0.2, color='lightblue', label='High-Low Range')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD/oz)")
    ax.set_title("Xu hướng giá vàng (Gold Price Trend)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Thống kê chuỗi thời gian
    st.write("### 📊 Thống kê chuỗi thời gian:")
    time_stats = {
        "Metric": ["Avg Close Price", "Max Close Price", "Min Close Price", "Price Range", "Volatility (Std)"],
        "Value": [
            f"${df['Close/Last'].mean():.2f}",
            f"${df['Close/Last'].max():.2f}",
            f"${df['Close/Last'].min():.2f}",
            f"${df['Close/Last'].max() - df['Close/Last'].min():.2f}",
            f"${df['Close/Last'].std():.2f}"
        ]
    }
    st.dataframe(pd.DataFrame(time_stats), use_container_width=True)
    
    # ===== VOLUME ANALYSIS =====
    st.subheader("📊 Phân tích khối lượng giao dịch (Volume Analysis)")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["Date"], df["Volume"], color='steelblue', alpha=0.7, width=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    ax.set_title("Khối lượng giao dịch (Trading Volume Over Time)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Volume statistics
    st.write("### 📊 Thống kê khối lượng:")
    volume_stats = {
        "Metric": ["Avg Volume", "Max Volume", "Min Volume", "Total Volume"],
        "Value": [
            f"{df['Volume'].mean():.0f}",
            f"{df['Volume'].max():.0f}",
            f"{df['Volume'].min():.0f}",
            f"{df['Volume'].sum():.0f}"
        ]
    }
    st.dataframe(pd.DataFrame(volume_stats), use_container_width=True)

# ==========================================================
# TAB 4: B4 – CORRELATION & DIMENSIONALITY REDUCTION
# ==========================================================
with tab4:
    st.header("B4 – Ma trận tương quan & Giảm chiều (Correlation & PCA)")
    
    # ===== CORRELATION MATRIX =====
    st.subheader("🔗 Ma trận tương quan (Correlation Matrix)")
    
    corr_matrix = df[quantitative_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                square=True, ax=ax, fmt=".2f", cbar_kws={'label': 'Correlation'})
    ax.set_title("Correlation Matrix of Quantitative Variables")
    st.pyplot(fig)
    
    st.write("### 🧠 Phân tích & Lập luận Giữ/Bỏ cột (Analysis & Column Selection Reasoning)")
    
    # Tính tương quan cao
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append({
                    "Column 1": corr_matrix.columns[i],
                    "Column 2": corr_matrix.columns[j],
                    "Correlation": round(corr_matrix.iloc[i, j], 4)
                })
    
    if high_corr_pairs:
        st.write("**Các cột có tương quan rất cao (> 0.95):**")
        st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
    
    st.write("""
    **Kết luận & Quyết định:**
    
    1. **Open, High, Low, Close/Last** có tương quan rất cao (> 0.95)
       - ❌ Giữ tất cả 4 cột là dư thừa
       - ✅ **Giữ:** `Close/Last` (giá đóng cửa - chỉ báo chính)
       - ❌ **Bỏ:** `Open`, `High`, `Low` (có thể suy ra từ Close/Last)
    
    2. **Volume** tương quan yếu với các cột giá
       - ✅ **Giữ:** `Volume` (thông tin độc lập, hữu ích)
       - Khối lượng giao dịch phản ánh mức độ quan tâm của thị trường
    
    **Kết quả cuối cùng:**
    - ✅ Giữ: `Close/Last`, `Volume`
    - ❌ Bỏ: `Open`, `High`, `Low` (dư thừa, tương quan cao)
    """)
    
    # ===== DIMENSIONALITY REDUCTION WITH PCA =====
    st.subheader("🎯 Giảm chiều dữ liệu (PCA - Principal Component Analysis)")
    
    # Prepare data for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[quantitative_cols])
    
    # Apply PCA with all components
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # Show cumulative variance explained
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance by PCA Components")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Apply PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=range(len(df)), 
                        cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("PCA Projection of Gold Stock Data (2D)")
    plt.colorbar(scatter, ax=ax, label='Time Index')
    st.pyplot(fig)
    
    st.write("### 📊 PCA Chi tiết (PCA Detailed Results)")
    pca_stats = {
        "Component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained Variance %": [f"{v:.2%}" for v in pca.explained_variance_ratio_],
        "Cumulative Variance %": [f"{c:.2%}" for c in np.cumsum(pca.explained_variance_ratio_)]
    }
    st.dataframe(pd.DataFrame(pca_stats), use_container_width=True)
    
    st.write(f"""
    **Nhận xét:**
    - PC1 explains: **{pca.explained_variance_ratio_[0]:.2%}** của phương sai
    - PC2 explains: **{pca.explained_variance_ratio_[1]:.2%}** của phương sai
    - Tổng cộng: **{sum(pca.explained_variance_ratio_):.2%}** phương sai được giải thích
    
    **Kết luận:** 2 thành phần chính giải thích **{sum(pca.explained_variance_ratio_):.2%}** phương sai,
    tức là giảm được từ {len(quantitative_cols)} chiều xuống 2 chiều mà vẫn giữ lại hầu hết thông tin.
    """)
    
    # Feature loadings
    st.write("### 🔍 Đóng góp của từng biến vào PC (Feature Loadings)")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(pca.components_))],
        index=quantitative_cols
    )
    st.dataframe(loadings_df, use_container_width=True)

# ==========================================================
# TAB 5: B5 – MACHINE LEARNING MODEL & INTERACTIVE VISUALIZATION
# ==========================================================
with tab5:
    st.header("B5 – Mô hình ML & Trực quan hóa tương tác (Model & Visualization)")
    
    st.info("⚠️ **Lưu ý:** Mô hình K-Means dưới đây chỉ mang tính minh họa để phân cụm dữ liệu. Mục đích là làm rõ cấu trúc dữ liệu, không đánh giá cao hiệu suất dự báo.")
    
    # ===== KMEANS CLUSTERING =====
    st.subheader("🎯 Phân cụm dữ liệu (K-Means Clustering)")
    
    k = st.slider(
        "Chọn số cụm (Select number of clusters)",
        min_value=2,
        max_value=6,
        value=3,
        step=1
    )
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df[quantitative_cols])
    
    # ===== VISUALIZATION 1: SCATTER PLOTS =====
    st.write("### 📊 Biểu đồ phân cụm (Cluster Scatter Plots)")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["Open"], df["Close/Last"], 
                           c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                  c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                  label='Centroids', zorder=5)
        ax.set_xlabel("Open Price ($)", fontsize=11)
        ax.set_ylabel("Close Price ($)", fontsize=11)
        ax.set_title(f"K-Means Clustering (Open vs Close) - K={k}")
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["Low"], df["High"],
                           c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
                  c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                  label='Centroids', zorder=5)
        ax.set_xlabel("Low Price ($)", fontsize=11)
        ax.set_ylabel("High Price ($)", fontsize=11)
        ax.set_title(f"K-Means Clustering (Low vs High) - K={k}")
        ax.legend()
        cbar = plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
           

    
    # ===== VISUALIZATION 2: TIME SERIES WITH CLUSTERS =====
    st.write("### ⏰ Phân bố cụm theo thời gian (Cluster Distribution Over Time)")
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, k))
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        ax.scatter(cluster_data["Date"], cluster_data["Close/Last"],
                  label=f"Cluster {cluster}", alpha=0.6, s=40, color=colors[cluster])
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price ($)", fontsize=11)
    ax.set_title(f"Gold Price with K-Means Clusters (K={k})")
    ax.legend(loc='best')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ===== CLUSTER STATISTICS =====
    st.write("### 📈 Thống kê chi tiết từng cụm (Cluster Statistics)")
    cluster_stats = df.groupby("Cluster")[quantitative_cols].agg(['mean', 'min', 'max', 'std'])
    st.dataframe(cluster_stats, use_container_width=True)
    
    # ===== CLUSTER SIZES =====
    st.write("### 📊 Kích thước cụm (Cluster Sizes)")
    cluster_sizes = df["Cluster"].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xlabel("Cluster", fontsize=11)
        ax.set_ylabel("Number of Data Points", fontsize=11)
        ax.set_title(f"Cluster Size Distribution (K={k})")
        ax.set_xticks(range(k))
        for i, v in enumerate(cluster_sizes.values):
            ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie(cluster_sizes.values, labels=[f"Cluster {i}\n({v} points)" for i, v in enumerate(cluster_sizes.values)],
               colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.05]*k)
        ax.set_title(f"Cluster Distribution Percentage (K={k})")
        st.pyplot(fig)
    
    # ===== CLUSTER CHARACTERISTICS =====
    st.write("### 🔍 Đặc điểm của từng cụm (Cluster Characteristics)")
    for cluster in range(k):
        cluster_data = df[df["Cluster"] == cluster]
        st.write(f"**Cluster {cluster}:** {len(cluster_data)} điểm dữ liệu")
        
        char_text = f"- **Giá Close trung bình:** ${cluster_data['Close/Last'].mean():.2f} (Range: ${cluster_data['Close/Last'].min():.2f} - ${cluster_data['Close/Last'].max():.2f})\n"
        char_text += f"- **Khối lượng trung bình:** {cluster_data['Volume'].mean():.0f}\n"
        char_text += f"- **Giai đoạn thời gian:** {cluster_data['Date'].min().date()} → {cluster_data['Date'].max().date()}"
        st.write(char_text)
    
    # ===== SAMPLE DATA WITH CLUSTERS =====
    st.write("### 📝 Dữ liệu mẫu sau phân cụm (Sample Data with Clusters)")
    display_cols = ["Date", "Open", "High", "Low", "Close/Last", "Volume", "Cluster"]
    st.dataframe(df[display_cols].head(20), use_container_width=True)
    
    # ===== MODEL EVALUATION =====
    st.write("### 📊 Đánh giá mô hình K-Means (Model Evaluation)")
    
    # Calculate inertia and silhouette score
    inertia = kmeans.inertia_
    silhouette = silhouette_score(df[quantitative_cols], df["Cluster"])
    davies_bouldin = davies_bouldin_score(df[quantitative_cols], df["Cluster"])
    calinski = calinski_harabasz_score(df[quantitative_cols], df["Cluster"])
    
    eval_metrics = {
        "Metric": ["Inertia", "Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
        "Value": [
            f"{inertia:.2f}",
            f"{silhouette:.4f}",
            f"{davies_bouldin:.4f}",
            f"{calinski:.2f}"
        ],
        "Interpretation": [
            "Sum of squared distances (Lower is better)",
            "Clustering quality (-1 to 1, Higher is better)",
            "Cluster separation (Lower is better)",
            "Cluster density (Higher is better)"
        ]
    }
    st.dataframe(pd.DataFrame(eval_metrics), use_container_width=True)
    
    st.write(f"""
    **Nhận xét:**
    - **Silhouette Score = {silhouette:.4f}**: {'Tốt ✓' if silhouette > 0.5 else 'Trung bình ⚠' if silhouette > 0.3 else 'Cần cải thiện ✗'}
    - Mô hình phân cụm giúp hiểu rõ cấu trúc dữ liệu giá vàng
    - Các cụm có thể đại diện cho các giai đoạn hoặc xu hướng giá khác nhau
    """)
    
    # ===========================
    # LINEAR REGRESSION - FORECAST (NÂNG CẤP ✅)
    # ===========================
    st.markdown("---")
    st.subheader("📈 Dự đoán giá vàng (Linear Regression + Regularization)")
    
    # 1) Chọn horizon + tỷ lệ test
    col1, col2, col3 = st.columns(3)
    with col1:
        horizon = st.selectbox("Chọn bước dự đoán (t+h, theo NGÀY)", [1, 7, 14, 30, 60, 90, 180, 252], index=3)
    with col2:
        test_ratio = st.slider("Tỷ lệ test (time split)", 0.15, 0.35, 0.25, 0.05)
    with col3:
        reg_type = st.radio("Regularization:", ["Ridge (L2)", "Lasso (L1)", "ElasticNet"], index=0)
    
    # 2) Tạo dataset supervised
    lr_data, feat_cols = make_lr_dataset(df, horizon=horizon)
    
    # ===== LỌC FEATURES NHIỄU =====
    feat_corr = lr_data[feat_cols + ["y"]].corr()["y"].drop("y")
    strong_features = feat_corr[feat_corr.abs() > 0.01].index.tolist()
    strong_features = [f for f in strong_features if f in feat_cols]
    
    if len(strong_features) < 3:
        st.warning("Quá ít feature sau lọc, dùng tất cả features")
        feat_cols_selected = feat_cols
    else:
        feat_cols_selected = strong_features
    
    st.write(f"**Features sau lọc:** {len(feat_cols_selected)}/{len(feat_cols)} (loại bỏ noise)")
    
    # Time split
    train_df, test_df = time_split(lr_data, test_ratio=test_ratio)
    X_train, y_train = train_df[feat_cols_selected], train_df["y"]
    X_test, y_test = test_df[feat_cols_selected], test_df["y"]
    
    # ===== CHUẨN HÓA DỮ LIỆU =====
    scaler_robust = RobustScaler()
    X_train_scaled = scaler_robust.fit_transform(X_train)
    X_test_scaled = scaler_robust.transform(X_test)
    
    # ===== REGULARIZATION + TRAINING =====
    alpha = st.slider("Độ mạnh regularization (α)", 0.001, 1.0, 0.1, 0.01)
    
    if reg_type == "Ridge (L2)":
        lr = Ridge(alpha=alpha)
    elif reg_type == "Lasso (L1)":
        lr = Lasso(alpha=alpha, max_iter=10000)
    else:
        lr = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    
    lr.fit(X_train_scaled, y_train)
    y_pred_train = lr.predict(X_train_scaled)
    y_pred = lr.predict(X_test_scaled)
    
    # ===== METRICS =====
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = lr.score(X_test_scaled, y_test)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("R² Score", f"{r2:.4f}")
    c4.metric("Test size", f"{len(test_df)}")
    c5.metric("Horizon", f"t+{horizon}d")
    
    # ===== ACTUAL vs PREDICTED =====
    st.write("### 📊 Actual vs Predicted Return (trên tập test)")
    
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_df["Date"].values, y_test.values, label="Actual Return", linewidth=2, color='blue')
    ax.plot(test_df["Date"].values, y_pred, label="Predicted Return", linewidth=2, color='red', alpha=0.7)
    ax.fill_between(test_df["Date"].values, y_test.values, y_pred, alpha=0.2, color='gray')
    ax.set_title(f"Actual vs Predicted Return (t+{horizon} ngày)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (% change)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # ===== RESIDUAL ANALYSIS =====
    st.write("### 🔍 Phân tích sai số (Residuals)")
    
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    
    axes[0].hist(residuals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Distribution of Residuals")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
    
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=30)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Residual Plot")
    axes[1].set_xlabel("Predicted Return")
    axes[1].set_ylabel("Residual")
    
    st.pyplot(fig)
    
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    st.write(f"**Residual Mean:** {residual_mean:.6f} (nên ≈ 0)")
    st.write(f"**Residual Std:** {residual_std:.6f}")
    
    # ===== FEATURE COEFFICIENTS =====
    st.write("### 🧠 Đóng góp của feature (Top 15 Coefficients)")
    coef_df = pd.DataFrame({"Feature": feat_cols_selected, "Coef": lr.coef_})
    coef_df["AbsCoef"] = coef_df["Coef"].abs()
    coef_df = coef_df.sort_values("AbsCoef", ascending=False)
    
    st.dataframe(coef_df.drop(columns=["AbsCoef"]).head(15), use_container_width=True)
    
    # ===========================
    # FORECAST TƯƠNG LAI (NÂNG CẤP) ✅✅✅
    # ===========================
    st.markdown("---")
    st.subheader("🔮 Dự báo tương lai (252 ngày, với khoảng tin cậy)")
    
    do_forecast = st.checkbox("Bật dự báo tương lai (multi-step forecast)", value=True)
    
    if do_forecast:
        col1, col2, col3 = st.columns(3)
        with col1:
            n_steps = st.slider("Số ngày dự báo", 30, 252, 90, 5)
        with col2:
            confidence_level = st.select_slider("Khoảng tin cậy", [0.68, 0.85, 0.95], value=0.95)
        with col3:
            use_business_days = st.checkbox("Dùng Business days", value=True)
        
        # ===== HÀM CHÍNH: TẠYO FEATURES TỪ LỊCH SỬ =====
        def build_features_from_history_scaled(hist_df: pd.DataFrame, scaler_obj, feat_cols_list) -> np.ndarray:
            """Xây dựng 1 dòng feature từ lịch sử, trả về array đã chuẩn hóa"""
            row = {}
            
            # lags
            for lag in [1, 2, 3, 5, 7, 14, 30]:
                if len(hist_df) >= lag:
                    row[f"close_lag_{lag}"] = hist_df["Close/Last"].iloc[-lag]
                else:
                    row[f"close_lag_{lag}"] = np.nan
            
            # rolling
            row["ma_7"] = hist_df["Close/Last"].rolling(7).mean().iloc[-1] if len(hist_df) >= 7 else np.nan
            row["ma_14"] = hist_df["Close/Last"].rolling(14).mean().iloc[-1] if len(hist_df) >= 14 else np.nan
            row["ma_20"] = hist_df["Close/Last"].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else np.nan
            row["ma_30"] = hist_df["Close/Last"].rolling(30).mean().iloc[-1] if len(hist_df) >= 30 else np.nan
            row["ma_60"] = hist_df["Close/Last"].rolling(60).mean().iloc[-1] if len(hist_df) >= 60 else np.nan
            row["ma_200"] = hist_df["Close/Last"].rolling(200).mean().iloc[-1] if len(hist_df) >= 200 else np.nan
            
            row["std_7"] = hist_df["Close/Last"].rolling(7).std().iloc[-1] if len(hist_df) >= 7 else np.nan
            row["std_14"] = hist_df["Close/Last"].rolling(14).std().iloc[-1] if len(hist_df) >= 14 else np.nan
            row["std_20"] = hist_df["Close/Last"].rolling(20).std().iloc[-1] if len(hist_df) >= 20 else np.nan
            
            # momentum
            row["momentum_7"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-8] if len(hist_df) >= 8 else np.nan
            row["momentum_14"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-15] if len(hist_df) >= 15 else np.nan
            row["momentum_30"] = hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-31] if len(hist_df) >= 31 else np.nan
            
            # ROC
            row["roc_7"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-8]) / hist_df["Close/Last"].iloc[-8] if len(hist_df) >= 8 else np.nan
            row["roc_14"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-15]) / hist_df["Close/Last"].iloc[-15] if len(hist_df) >= 15 else np.nan
            row["roc_30"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-31]) / hist_df["Close/Last"].iloc[-31] if len(hist_df) >= 31 else np.nan
            
            # RSI
            delta = hist_df["Close/Last"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            row["rsi_14"] = rsi.iloc[-1] if len(rsi) > 0 else np.nan
            
            # Bollinger Bands
            ma_20 = hist_df["Close/Last"].rolling(20).mean().iloc[-1] if len(hist_df) >= 20 else np.nan
            std_20 = hist_df["Close/Last"].rolling(20).std().iloc[-1] if len(hist_df) >= 20 else np.nan
            if pd.notna(ma_20) and pd.notna(std_20):
                bb_upper = ma_20 + 2 * std_20
                bb_lower = ma_20 - 2 * std_20
                row["bb_position"] = (hist_df["Close/Last"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
            else:
                row["bb_position"] = np.nan
            row["bb_upper"] = bb_upper if pd.notna(bb_upper) else np.nan
            row["bb_lower"] = bb_lower if pd.notna(bb_lower) else np.nan
            
            # Volume
            if "Volume" in hist_df.columns:
                row["vol_lag_1"] = hist_df["Volume"].iloc[-1]
                row["vol_ma_7"] = hist_df["Volume"].rolling(7).mean().iloc[-1] if len(hist_df) >= 7 else np.nan
                row["vol_ma_30"] = hist_df["Volume"].rolling(30).mean().iloc[-1] if len(hist_df) >= 30 else np.nan
                row["vol_std"] = hist_df["Volume"].rolling(7).std().iloc[-1] if len(hist_df) >= 7 else np.nan
                row["price_vol_trend"] = (hist_df["Close/Last"].iloc[-1] - hist_df["Close/Last"].iloc[-2]) / hist_df["Close/Last"].iloc[-2] * hist_df["Volume"].iloc[-1] if len(hist_df) >= 2 else np.nan
            
            # Chuyển thành DataFrame, chọn cột cần thiết, chuẩn hóa
            df_row = pd.DataFrame([row])
            df_row = df_row[feat_cols_list]
            
            x_scaled = scaler_obj.transform(df_row)
            return x_scaled.flatten()
        
        # ===== RECURSIVE FORECAST =====
        hist = df[["Date", "Close/Last"] + (["Volume"] if "Volume" in df.columns else [])].copy()
        hist = hist.sort_values("Date").reset_index(drop=True)
        
        last_date = hist["Date"].iloc[-1]
        last_close = hist["Close/Last"].iloc[-1]
        
        if use_business_days:
            future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=n_steps)
        else:
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_steps, freq="D")
        
        preds = []
        preds_std = []
        temp_hist = hist.copy()
        
        sigma = residual_std
        
        for i, dt in enumerate(future_dates):
            try:
                x_next = build_features_from_history_scaled(temp_hist, scaler_robust, feat_cols_selected)
                
                if np.isnan(x_next).any():
                    preds.append(np.nan)
                    preds_std.append(np.nan)
                else:
                    y_next = lr.predict(x_next.reshape(1, -1))[0]
                    preds.append(y_next)
                    preds_std.append(sigma)
                    
                    # Cập nhật lịch sử
                    new_close = temp_hist["Close/Last"].iloc[-1] * (1 + y_next)
                    new_row = {"Date": pd.Timestamp(dt), "Close/Last": new_close}
                    if "Volume" in temp_hist.columns:
                        new_row["Volume"] = temp_hist["Volume"].iloc[-1]
                    temp_hist = pd.concat([temp_hist, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                st.warning(f"Lỗi tại bước {i}: {e}")
                preds.append(np.nan)
                preds_std.append(np.nan)
        
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast_Return": preds,
            "Forecast_Std": preds_std
        })
        forecast_df = forecast_df.dropna().reset_index(drop=True)
        
        # Tính Confidence Interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        forecast_df["CI_Lower"] = forecast_df["Forecast_Return"] - z_score * forecast_df["Forecast_Std"]
        forecast_df["CI_Upper"] = forecast_df["Forecast_Return"] + z_score * forecast_df["Forecast_Std"]
        
        # Tính giá dự báo
        forecast_df["Forecast_Price"] = last_close
        for idx in range(len(forecast_df)):
            forecast_df.loc[idx, "Forecast_Price"] = forecast_df.loc[idx-1, "Forecast_Price"] * (1 + forecast_df.loc[idx, "Forecast_Return"]) if idx > 0 else last_close * (1 + forecast_df.loc[idx, "Forecast_Return"])
        
        st.write("### 📊 Kết quả dự báo (Forecast Results)")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(hist["Date"], hist["Close/Last"], label="Historical Price", linewidth=2, color='blue')
        ax.plot(forecast_df["Date"], forecast_df["Forecast_Price"], label="Forecast Price", linewidth=2, color='red', linestyle='--')
        ax.fill_between(forecast_df["Date"], forecast_df["Forecast_Price"] * (1 + forecast_df["CI_Lower"]), 
                        forecast_df["Forecast_Price"] * (1 + forecast_df["CI_Upper"]), 
                        alpha=0.2, color='red', label=f'{int(confidence_level*100)}% Confidence Interval')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD/oz)")
        ax.set_title(f"Gold Price Forecast ({n_steps} days)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)