# ==========================================================
# QUYTRÌNH PHÂN TÍCH DỰ VIỄN GIÁ VÀNG - STREAMLIT APP
# Gold Price Forecasting Dashboard
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score

# Configuration
st.set_page_config(
    page_title="Gold Price Analysis",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="expanded"
)

plt.style.use('ggplot')
sns.set_palette("husl")

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING & CACHING
# ============================================================
@st.cache_data
def load_default_data():
    """Load dữ liệu mặc định"""
    try:
        df = pd.read_csv("goldstock v2.csv", sep=";")
        df.columns = df.columns.str.strip()
        
        if "Column1" in df.columns:
            df.drop(columns=["Column1"], inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        
        numeric_cols = ["Volume", "Open", "High", "Low", "Close/Last"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Lỗi load dữ liệu: {e}")
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Sidebar Navigation
    st.sidebar.markdown("# 📊 QUYTRÌNH PHÂN TÍCH DỰ VIỄN GIÁ VÀNG")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "🔍 Chọn bước phân tích:",
        ["🏠 Trang chủ", "📤 B1: Mô tả dữ liệu", "🧹 B2: Làm sạch", 
         "📈 B3: Khai phá dữ liệu", "🔗 B4: Tương quan & PCA", "🤖 B5: Mô hình & Dự báo"]
    )
    
    # Load dữ liệu mặc định
    df = load_default_data()
    
    if df is None:
        st.error("❌ Không thể load dữ liệu. Vui lòng kiểm tra file goldstock v2.csv")
        return
    
    # ============================================================
    # TRANG CHỦ
    # ============================================================
    if page == "🏠 Trang chủ":
        st.title("🏆 Phân Tích Dự Viễn Giá Vàng")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Số hàng", f"{len(df):,}")
        with col2:
            st.metric("📋 Số cột", df.shape[1])
        with col3:
            st.metric("📅 Khoảng thời gian", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        
        st.markdown("""
        ### 📌 Quy trình phân tích:
        
        1. **B1: Mô tả dữ liệu** - Tổng quan, phân loại biến, thống kê
        2. **B2: Làm sạch dữ liệu** - Xử lý missing, outliers, validation
        3. **B3: Khai phá dữ liệu** - Phân tích đơn biến & đa biến
        4. **B4: Tương quan & PCA** - Ma trận tương quan, lựa chọn features
        5. **B5: Mô hình & Dự báo** - K-Means Clustering, Linear Regression
        
        ### 🎯 Mục tiêu:
        Phân tích toàn diện dữ liệu giá vàng từ khám phá đến xây dựng mô hình dự báo
        """)
        
        st.info("👈 Chọn một bước từ menu bên trái để bắt đầu!")
    
    # ============================================================
    # B1: MÔ TẢ DỮ LIỆU
    # ============================================================
    elif page == "📤 B1: Mô tả dữ liệu":
        st.title("B1 — Mô Tả Dữ Liệu (Data Description)")
        st.markdown("---")
        
        # B1.1 - SHAPE
        st.header("📊 B1.1 - Kích Thước Dữ Liệu")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔢 SHAPE", f"{df.shape}", delta=None)
        with col2:
            st.metric("📈 Số dòng", f"{df.shape[0]:,}")
        with col3:
            st.metric("📋 Số cột", df.shape[1])
        
        # B1.2 - PHÂN LOẠI DỮ LIỆU
        st.header("📋 B1.2 - Phân Loại Dữ Liệu")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        
        classification_data = []
        for i, col in enumerate(df.columns, 1):
            col_type = "🔢 Numerical (Định lượng)" if col in numeric_cols else "📝 Categorical (Định tính)"
            classification_data.append({
                "STT": i,
                "Column Name": col,
                "Data Type": col_type,
                "Type Detail": str(df[col].dtype)
            })
        
        classification_df = pd.DataFrame(classification_data)
        st.dataframe(classification_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🔢 Cột Định Lượng: {len(numeric_cols)}")
            for col in numeric_cols:
                st.write(f"   ✓ {col}")
        
        with col2:
            st.subheader(f"📝 Cột Định Tính: {len(categorical_cols)}")
            for col in categorical_cols:
                st.write(f"   ✓ {col}")
        
        # B1.3 - THỐNG KÊ MÔ TẢ
        st.header("📊 B1.3 - Thống Kê Mô Tả")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        
        # B1.4 - DỮ LIỆU MẪU
        st.header("🔹 B1.4 - Dữ Liệu Mẫu")
        st.dataframe(df.head(10), use_container_width=True)
    
    # ============================================================
    # B2: LÀM SẠCH DỮ LIỆU
    # ============================================================
    elif page == "🧹 B2: Làm sạch":
        st.title("B2 — Làm Sạch Dữ Liệu (Data Cleaning)")
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # B2.1 - MISSING VALUES
        st.header("📊 B2.1 - Kiểm Tra Missing Values")
        
        missing_df = pd.DataFrame({
            'Cột': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            fig = px.bar(missing_df, x='Cột', y='Missing %', title="Phần trăm Missing Values")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Không có missing values!")
        
        # B2.2 - OUTLIERS
        st.header("📊 B2.2 - Phát Hiện Outliers (IQR Method)")
        
        if numeric_cols:
            outlier_data = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                outlier_data.append({
                    'Cột': col,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound,
                    'Outlier Count': outlier_count
                })
            
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
            
            st.subheader("📊 Boxplot Visualization")
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 4*len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols):
                sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
                axes[idx].set_title(f"Outlier Detection: {col}", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # B2.3 - DUPLICATES
        st.header("📊 B2.3 - Kiểm Tra Dữ Liệu Trùng Lặp")
        
        duplicate_count = df.duplicated().sum()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Số dòng trùng lặp", duplicate_count)
        with col2:
            if duplicate_count == 0:
                st.success("✅ Không có dòng trùng lặp!")
            else:
                st.warning(f"⚠️ Phát hiện {duplicate_count} dòng trùng lặp")
    
    # ============================================================
    # B3: KHAI PHÁ DỮ LIỆU (EDA)
    # ============================================================
    elif page == "📈 B3: Khai phá dữ liệu":
        st.title("B3 — Khai Phá Dữ Liệu (Exploratory Data Analysis)")
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        # Thêm cột Year và Price_Range
        df['Year'] = df['Date'].dt.year
        df['Price_Range'] = df['High'] - df['Low']
        
        # B3.1 - LINE CHART
        st.header("📊 B3.1 - Biểu Đồ Đường: Xu Hướng Giá")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['Date'], df['Close/Last'], linewidth=2.5, color='steelblue', label='Close Price')
        ax.fill_between(df['Date'], df['Low'], df['High'], alpha=0.2, color='lightblue', label='High-Low Range')
        ax.axhline(y=df['Close/Last'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Ngày (Date)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Giá (USD/oz)', fontsize=11, fontweight='bold')
        ax.set_title('Xu Hướng Giá Vàng Theo Thời Gian', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # B3.2 - HISTOGRAM
        st.header("📊 B3.2 - Biểu Đồ Cột: Phân Phối Volume")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.hist(df['Volume'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(df['Volume'].mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {df["Volume"].mean():,.0f}')
        ax.axvline(df['Volume'].median(), color='green', linestyle='--', linewidth=2.5, label=f'Median: {df["Volume"].median():,.0f}')
        ax.set_xlabel('Khối Lượng (Volume)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tần Suất (Frequency)', fontsize=11, fontweight='bold')
        ax.set_title('Phân Phối Khối Lượng Giao Dịch', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        
        # B3.3 - SCATTER PLOT
        st.header("📊 B3.3 - Biểu Đồ Phân Tán: Volume vs Close/Last")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        scatter = ax.scatter(df['Volume'], df['Close/Last'], c=range(len(df)), cmap='viridis', alpha=0.6, s=80, edgecolors='black', linewidth=0.8)
        z = np.polyfit(df['Volume'], df['Close/Last'], 1)
        p = np.poly1d(z)
        volume_sorted = df['Volume'].sort_values()
        correlation = df['Volume'].corr(df['Close/Last'])
        ax.plot(volume_sorted, p(volume_sorted), "r--", linewidth=2.5, label=f'Trend (r={correlation:.3f})')
        ax.set_xlabel('Khối Lượng (Volume)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Giá Đóng Cửa (USD/oz)', fontsize=11, fontweight='bold')
        ax.set_title('Mối Quan Hệ Giữa Khối Lượng và Giá', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax, label='Thứ tự thời gian')
        st.pyplot(fig)
        
        # B3.4 - BOXPLOT
        st.header("📊 B3.4 - Boxplot: Biến Động Giá Theo Năm")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        years = sorted(df['Year'].unique())
        data_by_year = [df[df['Year'] == year]['Price_Range'].values for year in years]
        
        bp = ax.boxplot(data_by_year, labels=years, patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor='lightblue', color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2.5),
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
        
        ax.set_xlabel('Năm (Year)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Biến Động Giá (USD/oz)', fontsize=11, fontweight='bold')
        ax.set_title('So Sánh Biến Động Giá Theo Năm', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
    
    # ============================================================
    # B4: TƯƠNG QUAN & PCA
    # ============================================================
    elif page == "🔗 B4: Tương quan & PCA":
        st.title("B4 — Ma Trận Tương Quan & PCA")
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Cần ít nhất 2 cột định lượng để phân tích")
            return
        
        # B4.1 - CORRELATION MATRIX
        st.header("📊 B4.1 - Ma Trận Tương Quan")
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
        ))
        fig.update_layout(title="Ma Trận Tương Quan", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # B4.2 - HIGH CORRELATION PAIRS
        st.header("📊 B4.2 - Các Cặp Tương Quan Cao")
        
        high_corr_threshold = st.slider("Ngưỡng tương quan", 0.5, 1.0, 0.9, 0.05)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                    high_corr_pairs.append({
                        'Var1': corr_matrix.columns[i],
                        'Var2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            st.dataframe(high_corr_df, use_container_width=True)
        else:
            st.info(f"Không có cặp tương quan > {high_corr_threshold}")
        
        # B4.3 - PCA
        st.header("🔗 B4.3 - Phân Tích PCA")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])
        
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
                   pca_full.explained_variance_ratio_*100,
                   alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance (%)')
        axes[0].set_title('Scree Plot', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var*100, 'bo-', linewidth=2, markersize=8)
        axes[1].axhline(y=95, color='red', linestyle='--', linewidth=2, label='95%')
        axes[1].axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90%')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance (%)')
        axes[1].set_title('Cumulative Variance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2D PCA Projection
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)',
                               'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)'},
                        title='PCA 2D Projection')
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # B5: MÔ HÌNH & DỰ BÁO
    # ============================================================
    elif page == "🤖 B5: Mô hình & Dự báo":
        st.title("B5 — Mô Hình Machine Learning & Dự Báo")
        st.markdown("---")
        
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        tab1, tab2 = st.tabs(["🎯 K-Means Clustering", "📈 Linear Regression"])
        
        # ============================================================
        # K-MEANS
        # ============================================================
        with tab1:
            st.header("🎯 K-Means Clustering")
            
            k = st.sidebar.slider("Số cụm (K)", min_value=2, max_value=8, value=3, step=1)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols])
            
            with st.spinner("Đang huấn luyện K-Means..."):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                df_plot = df.copy()
                df_plot['Cluster'] = kmeans.fit_predict(X_scaled)
            
            st.success(f"✅ Phân cụm thành {k} nhóm")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = df_plot['Cluster'].value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           title="Phân Phối Clusters",
                           labels={'x': 'Cluster', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                silhouette = silhouette_score(X_scaled, df_plot['Cluster'])
                st.metric("Silhouette Score", f"{silhouette:.4f}")
                if silhouette > 0.5:
                    st.success("✅ Phân cụm tốt")
                else:
                    st.warning("⚠️ Phân cụm có thể được cải thiện")
            
            st.subheader("📊 Thống Kê Clusters")
            cluster_stats = df_plot.groupby('Cluster')[numeric_cols].mean()
            st.dataframe(cluster_stats, use_container_width=True)
        
        # ============================================================
        # LINEAR REGRESSION
        # ============================================================
        with tab2:
            st.header("📈 Linear Regression - Dự Báo Giá")
            
            target_col = [col for col in numeric_cols if 'close' in col.lower() or 'price' in col.lower()]
            if not target_col:
                target_col = numeric_cols[0]
            else:
                target_col = target_col[0]
            
            if 'Date' in df.columns:
                df_model = df.copy()
                df_model['Days'] = (df_model['Date'] - df_model['Date'].min()).dt.days
                X = df_model[['Days']].values
            else:
                X = np.arange(len(df)).reshape(-1, 1)
                df_model = df.copy()
                df_model['Days'] = np.arange(len(df))
            
            y = df_model[target_col].values
            
            with st.spinner("Đang huấn luyện Linear Regression..."):
                lr = LinearRegression()
                lr.fit(X, y)
                y_pred = lr.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"${rmse:.2f}")
            with col3:
                st.metric("MAE", f"${mae:.2f}")
            with col4:
                st.metric("Slope", f"${lr.coef_[0]:.4f}/day")
            
            end_year = st.sidebar.selectbox("Dự báo đến năm", [2025, 2026, 2027, 2028, 2030], index=2)
            
            if 'Date' in df.columns:
                last_date = df_model['Date'].max()
                target_date = pd.Timestamp(f'{end_year}-12-31')
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=target_date, freq='D')
                future_days = (future_dates - df_model['Date'].min()).days.values.reshape(-1, 1)
                future_prices = lr.predict(future_days)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_model['Date'], y=y, name='Actual',
                                        mode='lines', line=dict(color='steelblue', width=2)))
                fig.add_trace(go.Scatter(x=df_model['Date'], y=y_pred, name='Fit',
                                        mode='lines', line=dict(color='orange', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=future_dates, y=future_prices, name=f'Forecast to {end_year}',
                                        mode='lines', line=dict(color='red', width=2)))
                
                fig.update_layout(title=f"Dự Báo Giá Vàng Đến {end_year}", height=500,
                                 xaxis_title="Ngày", yaxis_title=f"Giá ({target_col})")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("🔮 Thống Kê Dự Báo")
                pred_price = lr.predict([[((pd.Timestamp(f'{end_year}-12-31') - df_model['Date'].min()).days)]])[0]
                std_error = np.std(y - y_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Giá dự kiến {end_year}", f"${pred_price:.2f}")
                with col2:
                    st.metric("Khoảng tin cậy ±", f"${1.96*std_error:.2f}")
            
            st.warning("⚠️ Lưu ý: Dự báo dài hạn có độ tin cậy thấp do giả định xu hướng tuyến tính")

if __name__ == "__main__":
    main()
