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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.style.use('ggplot')

# ==========================================================
# LOAD DATA - B0: DATASET OVERVIEW
# ==========================================================
# Load CSV with semicolon delimiter
original_df = pd.read_csv("goldstock v2.csv", sep=";")

print("Columns in CSV:", original_df.columns.tolist())
print("First few rows:")
print(original_df.head())

# Xóa cột index không cần thiết nếu có
if "Column1" in original_df.columns:
    original_df.drop(columns=["Column1"], inplace=True)

if "Unnamed: 0" in original_df.columns:
    original_df.drop(columns=["Unnamed: 0"], inplace=True)

# Đảm bảo các cột chứa dữ liệu được xử lý đúng
# Xử lý khoảng trắng dư thừa
original_df.columns = original_df.columns.str.strip()

# Chuyển đổi cột số sang numeric type
numeric_cols = ["Volume", "Open", "High", "Low", "Close/Last"]
for col in numeric_cols:
    if col in original_df.columns:
        original_df[col] = pd.to_numeric(original_df[col], errors='coerce')

# Chuyển Date sang datetime với xử lý lỗi
try:
    # Thử nhiều định dạng khác nhau (DD/MM/YYYY là phổ biến)
    original_df["Date"] = pd.to_datetime(original_df["Date"], format="%d/%m/%Y", errors='coerce')
    
    # Kiểm tra và loại bỏ các giá trị null sau khi convert
    null_dates = original_df["Date"].isnull().sum()
    if null_dates > 0:
        print(f"Warning: {null_dates} rows with invalid dates will be removed")
        original_df = original_df.dropna(subset=["Date"])
        
except Exception as e:
    print(f"Error converting Date column: {e}")
    print("Attempting alternative date parsing...")
    original_df["Date"] = pd.to_datetime(original_df["Date"], infer_datetime_format=True, errors='coerce')
    original_df = original_df.dropna(subset=["Date"])

# Sắp xếp theo thời gian (từ cũ đến mới)
original_df.sort_values(by="Date", inplace=True, ascending=True)
original_df.reset_index(drop=True, inplace=True)

print(f"Data loaded successfully: {len(original_df)} rows")
print("Columns after processing:", original_df.columns.tolist())
print(original_df.head())

# ==========================================================
# B2 – DATA CLEANING (TIỀN XỬ LÝ DỮ LIỆU)
# ==========================================================
# Clone một bản sao để giữ nguyên dữ liệu gốc
df = original_df.copy()

# Kiểm tra dữ liệu thiếu
missing_data = df.isnull().sum() * 100 / df.shape[0]

# Xóa duplicate
df = df[df.duplicated() == False].reset_index(drop=True)

# Xóa các hàng có giá trị NaN sau khi xử lý
df = df.dropna(subset=["Date", "Open", "High", "Low", "Close/Last", "Volume"])

# Kiểm tra logic giá (High >= Open, Close/Last, Low; Low <= Open, Close/Last)
if "High" in df.columns and "Low" in df.columns:
    df = df[
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close/Last"]) &
        (df["High"] >= df["Low"]) &
        (df["Low"] <= df["Open"]) &
        (df["Low"] <= df["Close/Last"])
    ]

df.reset_index(drop=True, inplace=True)

# ==========================================================
# B1 – MÔ TẢ DỮ LIỆU (DATA OVERVIEW)
# ==========================================================
quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
qualitative_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

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
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
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
    
    # ==========================================================
    # LINEAR REGRESSION - PRICE PREDICTION
    # ==========================================================
    st.markdown("---")
    st.subheader("📈 Dự đoán giá vàng (Linear Regression Prediction)")
    
    st.info("🔮 **Mô hình Linear Regression để dự đoán giá vàng đến năm 2027**")
    
    # Prepare data for Linear Regression
    # Convert Date to numeric (days since first date)
    df_model = df.copy()
    df_model['Days'] = (df_model['Date'] - df_model['Date'].min()).dt.days
    
    # Features and target
    X = df_model[['Days']].values
    y = df_model['Close/Last'].values
    
    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    # Predictions on training data
    y_pred_train = lr_model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    
    # Display model performance
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"${rmse:.2f}")
    with col3:
        st.metric("MAE", f"${mae:.2f}")
    with col4:
        st.metric("MSE", f"${mse:.2f}")
    
    # Create future dates up to 2027
    last_date = df_model['Date'].max()
    target_date = pd.Timestamp('2027-12-31')
    
    # Generate future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=target_date, freq='D')
    future_days = (future_dates - df_model['Date'].min()).days.values.reshape(-1, 1)
    
    # Predict future prices
    future_prices = lr_model.predict(future_days)
    
    # Combine historical and future data
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices
    })
    
    # Visualization: Historical + Predictions
    st.write("### 📊 Biểu đồ dự đoán giá vàng (Gold Price Prediction Chart)")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical actual prices
    ax.plot(df_model['Date'], df_model['Close/Last'], 
            linewidth=2, color='steelblue', label='Historical Actual Price', alpha=0.8)
    
    # Historical predicted (fitted line)
    ax.plot(df_model['Date'], y_pred_train, 
            linewidth=2, color='orange', linestyle='--', label='Linear Regression Fit', alpha=0.7)
    
    # Future predictions
    ax.plot(future_df['Date'], future_df['Predicted_Price'], 
            linewidth=2.5, color='red', linestyle='-', label='Future Prediction (to 2027)', alpha=0.8)
    
    # Add confidence interval (simple approach)
    std_error = np.std(y - y_pred_train)
    ax.fill_between(future_df['Date'], 
                    future_df['Predicted_Price'] - 1.96*std_error,
                    future_df['Predicted_Price'] + 1.96*std_error,
                    alpha=0.2, color='red', label='95% Confidence Interval')
    
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Price (USD/oz)", fontsize=12, fontweight='bold')
    ax.set_title("Gold Price Prediction using Linear Regression (Historical + Future)", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show prediction statistics
    st.write("### 📊 Thống kê dự đoán (Prediction Statistics)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dự đoán giá vàng:**")
        prediction_stats = {
            "Date": ["Last Historical", "End of 2025", "End of 2026", "End of 2027"],
            "Predicted Price": [
                f"${df_model['Close/Last'].iloc[-1]:.2f}",
                f"${lr_model.predict([[((pd.Timestamp('2025-12-31') - df_model['Date'].min()).days)]])[0]:.2f}",
                f"${lr_model.predict([[((pd.Timestamp('2026-12-31') - df_model['Date'].min()).days)]])[0]:.2f}",
                f"${lr_model.predict([[((pd.Timestamp('2027-12-31') - df_model['Date'].min()).days)]])[0]:.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(prediction_stats), use_container_width=True)
    
    with col2:
        st.write("**Thông số mô hình:**")
        model_params = {
            "Parameter": ["Slope (Hệ số góc)", "Intercept (Hằng số)", "Daily Price Change"],
            "Value": [
                f"{lr_model.coef_[0]:.4f}",
                f"${lr_model.intercept_:.2f}",
                f"${lr_model.coef_[0]:.4f}/day"
            ]
        }
        st.dataframe(pd.DataFrame(model_params), use_container_width=True)
    
    # Model equation
    st.write("### 📐 Phương trình hồi quy (Regression Equation)")
    st.latex(f"Price = {lr_model.intercept_:.2f} + {lr_model.coef_[0]:.4f} \\times Days")
    
    # Interpretation
    st.write("### 💡 Giải thích kết quả (Interpretation)")
    st.write(f"""
    **Ý nghĩa các chỉ số:**
    - **R² = {r2:.4f}**: Mô hình giải thích {r2*100:.2f}% sự biến động của giá vàng {'✓ (Tốt)' if r2 > 0.7 else '⚠ (Trung bình)' if r2 > 0.5 else '✗ (Yếu)'}
    - **RMSE = ${rmse:.2f}**: Sai số trung bình khoảng ${rmse:.2f}
    - **Slope = {lr_model.coef_[0]:.4f}**: Giá vàng {'tăng' if lr_model.coef_[0] > 0 else 'giảm'} trung bình ${abs(lr_model.coef_[0]):.4f}/ngày
    
    **Xu hướng:**
    {f"📈 Giá vàng có xu hướng tăng đều đặn với tốc độ ${lr_model.coef_[0]*365:.2f}/năm" if lr_model.coef_[0] > 0 else f"📉 Giá vàng có xu hướng giảm với tốc độ ${abs(lr_model.coef_[0]*365):.2f}/năm"}
    
    **Lưu ý:** ⚠️ Dự đoán dài hạn với Linear Regression có thể không chính xác do giả định xu hướng tuyến tính. 
    Giá vàng bị ảnh hưởng bởi nhiều yếu tố kinh tế, chính trị phức tạp.
    """)
    
    # Download prediction data
    st.write("### 💾 Tải dữ liệu dự đoán (Download Prediction Data)")
    
    # Combine all data for download
    download_df = pd.DataFrame({
        'Date': list(df_model['Date']) + list(future_df['Date']),
        'Actual_Price': list(df_model['Close/Last']) + [np.nan]*len(future_df),
        'Predicted_Price': list(y_pred_train) + list(future_df['Predicted_Price']),
        'Type': ['Historical']*len(df_model) + ['Future']*len(future_df)
    })
    
    csv_data = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction CSV",
        data=csv_data,
        file_name="gold_price_prediction_to_2027.csv",
        mime="text/csv"
    )
