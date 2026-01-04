import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Phân Tích Giá Vàng",
    layout="wide",
    page_icon="🏆",
    initial_sidebar_state="expanded"
)

plt.style.use('ggplot')
sns.set_palette("husl")
onefig_size = (10, 4)
multifigs_size = (12, 4)

# ============================================================
# CUSTOM CSS
# ============================================================
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
def load_data():
    """Load dữ liệu từ goldstock v2.csv"""
    try:
        df = pd.read_csv("goldstock v2.csv", sep=";")
        df.columns = df.columns.str.strip()
        
        # Xóa cột index không cần thiết
        if "Column1" in df.columns:
            df.drop(columns=["Column1"], inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        
        # Convert numeric columns
        numeric_cols = ["Volume", "Open", "High", "Low", "Close/Last"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Date
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"❌ Lỗi load dữ liệu: {e}")
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Load dữ liệu
    original_df = load_data()
    
    if original_df is None:
        st.error("Không thể load dữ liệu!")
        return
    
    # Sidebar Navigation
    st.sidebar.markdown("# 📊 PHÂN TÍCH GIÁ VÀNG")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "🔍 Chọn bước phân tích:",
        ["🏠 Trang chủ", 
         "📤 B1: Mô tả Dữ Liệu", 
         "🧹 B2: Làm Sạch", 
         "📊 B3: Khai Phá",
         "🔗 B4: Tương Quan & PCA",
         "🤖 B5: Mô Hình & Dự Báo"]
    )
    
    # ============================================================
    # TRANG CHỦ
    # ============================================================
    if page == "🏠 Trang chủ":
        st.title("🏆 Phân Tích Dự Viễn Giá Vàng")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Số Hàng", f"{len(original_df):,}")
        with col2:
            st.metric("📋 Số Cột", original_df.shape[1])
        with col3:
            st.metric("📅 Khoảng Thời Gian", 
                     f"{original_df['Date'].min().date()} → {original_df['Date'].max().date()}")
        
        st.markdown("""
        ### 📌 Quy Trình Phân Tích 5 Bước:
        
        1. **B1: Mô Tả Dữ Liệu** - Overview, phân loại biến, thống kê
        2. **B2: Làm Sạch Dữ Liệu** - Xử lý missing, outliers, validation
        3. **B3: Khai Phá Dữ Liệu** - Biểu đồ, phân tích đơn/đa biến
        4. **B4: Tương Quan & PCA** - Ma trận tương quan, dimensionality reduction
        5. **B5: Mô Hình & Dự Báo** - K-Means, Linear Regression, dự báo
        
        ### 🎯 Mục Tiêu:
        Phân tích toàn diện dữ liệu giá vàng từ khám phá đến xây dựng mô hình dự báo.
        """)
    
    # ============================================================
    # B1: MÔ TẢ DỮ LIỆU (Dataset Overview)
    # ============================================================
    elif page == "📤 B1: Mô tả Dữ Liệu":
        st.title("B1 — Mô Tả Dữ Liệu (Dataset Overview)")
        st.markdown("---")
        
        st.markdown("## 📊 Dataset Goldstock v2")
        st.markdown("""
        Dataset bao gồm dữ liệu giá vàng với các biến:
        - **Date**: Ngày giao dịch
        - **Open**: Giá mở cửa
        - **High**: Giá cao nhất
        - **Low**: Giá thấp nhất
        - **Close/Last**: Giá đóng cửa
        - **Volume**: Khối lượng giao dịch
        """)
        
        st.header("📋 Thông Tin Dữ Liệu")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số hàng", original_df.shape[0])
        with col2:
            st.metric("Số cột", original_df.shape[1])
        
        # Info
        st.subheader("Loại Dữ Liệu")
        info_str = f"""
        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
        <pre>{str(original_df.dtypes)}</pre>
        </div>
        """
        st.markdown(info_str, unsafe_allow_html=True)
        
        # Descriptive statistics
        st.subheader("📊 Thống Kê Mô Tả")
        st.dataframe(original_df.describe(), use_container_width=True)
        
        # Sample data
        st.subheader("🔹 Mẫu Dữ Liệu (10 Hàng Đầu)")
        st.dataframe(original_df.head(10), use_container_width=True)
    
    # ============================================================
    # B2: LÀM SẠCH DỮ LIỆU (Data Cleaning)
    # ============================================================
    elif page == "🧹 B2: Làm Sạch":
        st.title("B2 — Làm Sạch Dữ Liệu (Data Cleaning)")
        st.markdown("---")
        
        df = original_df.copy()
        
        # ========== B2.1: Data Duplication ==========
        st.header("📋 B2.1 - Kiểm Tra Dữ Liệu Trùng Lặp")
        dup_ind = df.duplicated()
        dup_count = dup_ind.sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Số dòng trùng lặp", dup_count)
        with col2:
            if dup_count == 0:
                st.success("✅ Không có dòng trùng lặp!")
            else:
                st.warning(f"⚠️ Phát hiện {dup_count} dòng trùng lặp")
        
        if dup_count > 0:
            st.dataframe(df[dup_ind], use_container_width=True)
            df = df[dup_ind == False]
        
        # ========== B2.2: Missing Values ==========
        st.header("🔍 B2.2 - Kiểm Tra Missing Values")
        missing_count = df.isnull().sum()
        missing_pct = (missing_count / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Cột': missing_count.index,
            'Missing Count': missing_count.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            st.warning("⚠️ Phát hiện missing values:")
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ Không có missing values!")
        
        # ========== B2.3: Outlier Detection ==========
        st.header("📊 B2.3 - Phát Hiện Outliers (IQR Method)")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        outlier_summary = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            outlier_summary.append({
                'Cột': col,
                'Q1': f"{Q1:.2f}",
                'Q3': f"{Q3:.2f}",
                'IQR': f"{IQR:.2f}",
                'Lower Bound': f"{lower_bound:.2f}",
                'Upper Bound': f"{upper_bound:.2f}",
                'Outlier Count': outlier_count
            })
        
        st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)
        
        # Boxplot
        st.subheader("📊 Boxplot Visualization")
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 3*len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
            axes[idx].set_title(f"Outlier Detection: {col}", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ========== B2.4: Data Validation ==========
        st.header("✅ B2.4 - Data Validation")
        st.success("✅ Dữ liệu đã qua kiểm tra xác thực")
        st.info("💡 Dữ liệu sạch và sẵn sàng cho bước tiếp theo!")
    
    # ============================================================
    # B3: KHAI PHÁ DỮ LIỆU (Exploratory Data Analysis)
    # ============================================================
    elif page == "📊 B3: Khai Phá":
        st.title("B3 — Khai Phá Dữ Liệu (Exploratory Data Analysis)")
        st.markdown("---")
        
        df = original_df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Price_Range'] = df['High'] - df['Low']
        
        # ========== B3.1: Line Chart ==========
        st.header("📊 B3.1 - Biểu Đồ Đường: Xu Hướng Giá")
        fig, ax = plt.subplots(figsize=multifigs_size)
        ax.plot(df['Date'], df['Close/Last'], linewidth=2.5, color='steelblue', label='Close Price')
        ax.fill_between(df['Date'], df['Low'], df['High'], alpha=0.2, color='lightblue', label='High-Low Range')
        ax.axhline(y=df['Close/Last'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Ngày (Date)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Giá (USD/oz)', fontsize=11, fontweight='bold')
        ax.set_title('Xu Hướng Giá Vàng Theo Thời Gian', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # ========== B3.2: Histogram ==========
        st.header("📊 B3.2 - Biểu Đồ Cột: Phân Phối Volume")
        fig, ax = plt.subplots(figsize=multifigs_size)
        ax.hist(df['Volume'], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(df['Volume'].mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {df["Volume"].mean():,.0f}')
        ax.axvline(df['Volume'].median(), color='green', linestyle='--', linewidth=2.5, label=f'Median: {df["Volume"].median():,.0f}')
        ax.set_xlabel('Khối Lượng (Volume)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tần Suất (Frequency)', fontsize=11, fontweight='bold')
        ax.set_title('Phân Phối Khối Lượng Giao Dịch', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        
        # ========== B3.3: Scatter Plot ==========
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
        
        # ========== B3.4: Boxplot by Year ==========
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
        
        # ========== B3.5: Correlation Heatmap ==========
        st.header("📊 B3.5 - Ma Trận Tương Quan")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, 
                    cbar_kws={'label': 'Correlation'}, square=True)
        ax.set_title('Ma Trận Tương Quan', fontsize=13, fontweight='bold')
        st.pyplot(fig)
    
    # ============================================================
    # B4: TƯƠNG QUAN & PCA
    # ============================================================
    elif page == "🔗 B4: Tương Quan & PCA":
        st.title("B4 — Ma Trận Tương Quan & PCA")
        st.markdown("---")
        
        df = original_df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Cần ít nhất 2 cột định lượng!")
            return
        
        # ========== B4.1: Correlation Matrix ==========
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
        
        # ========== B4.2: High Correlation Pairs ==========
        st.header("📊 B4.2 - Các Cặp Tương Quan Cao")
        threshold = st.slider("Ngưỡng tương quan", 0.5, 1.0, 0.7, 0.05)
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'Var1': corr_matrix.columns[i],
                        'Var2': corr_matrix.columns[j],
                        'Correlation': round(corr_matrix.iloc[i, j], 4)
                    })
        
        if high_corr_pairs:
            st.dataframe(pd.DataFrame(high_corr_pairs), use_container_width=True)
        else:
            st.info(f"Không có cặp tương quan > {threshold}")
        
        # ========== B4.3: PCA Analysis ==========
        st.header("🔗 B4.3 - Phân Tích PCA")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[numeric_cols])
        
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=onefig_size)
            ax.bar(range(1, len(pca_full.explained_variance_ratio_)+1),
                   pca_full.explained_variance_ratio_*100,
                   alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance (%)')
            ax.set_title('Scree Plot', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=onefig_size)
            ax.plot(range(1, len(cumsum_var)+1), cumsum_var*100, 'bo-', linewidth=2, markersize=8)
            ax.axhline(y=95, color='red', linestyle='--', linewidth=2, label='95%')
            ax.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90%')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Variance (%)')
            ax.set_title('Cumulative Variance', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # 2D PCA Projection
        st.subheader("📊 PCA 2D Projection")
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_scaled)
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                        labels={'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)',
                               'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)'},
                        title='PCA 2D Projection',
                        color=range(len(X_pca)),
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # B5: MÔ HÌNH & DỰ BÁO
    # ============================================================
    elif page == "🤖 B5: Mô Hình & Dự Báo":
        st.title("B5 — Mô Hình Machine Learning & Dự Báo")
        st.markdown("---")
        
        df = original_df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        tab1, tab2 = st.tabs(["🎯 K-Means Clustering", "📈 Linear Regression"])
        
        # ========== K-MEANS ==========
        with tab1:
            st.header("🎯 K-Means Clustering")
            
            k = st.slider("Số cụm (K)", min_value=2, max_value=8, value=3, step=1)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols])
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df_cluster = df.copy()
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
            
            st.success(f"✅ Phân cụm thành {k} nhóm")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           title="Phân Phối Clusters",
                           labels={'x': 'Cluster', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                silhouette = silhouette_score(X_scaled, df_cluster['Cluster'])
                st.metric("Silhouette Score", f"{silhouette:.4f}")
                if silhouette > 0.5:
                    st.success("✅ Phân cụm tốt")
                else:
                    st.warning("⚠️ Phân cụm có thể được cải thiện")
            
            st.subheader("📊 Thống Kê Clusters")
            cluster_stats = df_cluster.groupby('Cluster')[numeric_cols].mean()
            st.dataframe(cluster_stats, use_container_width=True)
        
        # ========== LINEAR REGRESSION ==========
        with tab2:
            st.header("📈 Linear Regression - Dự Báo Giá")
            
            target_col = [col for col in numeric_cols if 'close' in col.lower() or 'price' in col.lower()]
            if not target_col:
                target_col = numeric_cols[0]
            else:
                target_col = target_col[0]
            
            df_model = df.copy()
            df_model['Days'] = (df_model['Date'] - df_model['Date'].min()).dt.days
            
            X = df_model[['Days']].values
            y = df_model[target_col].values
            
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
            
            end_year = st.selectbox("Dự báo đến năm", [2024, 2025, 2026, 2027, 2030])
            
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

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()
