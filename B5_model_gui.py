# ==========================================================
# B5 – MODEL MACHINE LEARNING & GUI
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score

plt.style.use('ggplot')

# ==========================================================
# LOAD DỮ LIỆU
# ==========================================================
@st.cache_data
def load_data():
    """Load và xử lý dữ liệu"""
    try:
        df = pd.read_csv("goldstock_cleaned_B2.csv")
        df["Date"] = pd.to_datetime(df["Date"])
    except:
        df = pd.read_csv("goldstock v2.csv", sep=";")
        if "Column1" in df.columns:
            df.drop(columns=["Column1"], inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        df.columns = df.columns.str.strip()
        numeric_cols = ["Volume", "Open", "High", "Low", "Close/Last"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')
        except:
            df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True, errors='coerce')
        df = df.dropna()
        df = df[df.duplicated() == False].reset_index(drop=True)
    
    return df

def render_app():
    """Render B5 app content inside an existing Streamlit page."""
    st.title("📊 B5 - Machine Learning Model & Visualization")
    st.markdown("---")

    # Load data
    df = load_data()
    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.success(f"✅ Dữ liệu đã load: {len(df)} hàng, {df.shape[1]} cột")

    # Tạo tabs
    tab1, tab2 = st.tabs(["🎯 K-Means Clustering", "📈 Linear Regression Prediction"])

    # ==========================================================
    # TAB 1: K-MEANS CLUSTERING
    # ==========================================================
    with tab1:
        st.header("🎯 Phân cụm dữ liệu (K-Means Clustering)")
        
        st.info("⚠️ **Lưu ý:** Mô hình K-Means dưới đây chỉ mang tính minh họa để phân cụm dữ liệu. Mục đích là làm rõ cấu trúc dữ liệu, không đánh giá cao hiệu suất dự báo.")
        
        # Sidebar controls
        st.sidebar.header("⚙️ K-Means Settings")
        k = st.sidebar.slider(
            "Số cụm (Number of clusters)",
            min_value=2,
            max_value=6,
            value=3,
            step=1
        )
        
        random_state = st.sidebar.number_input("Random State", value=42, step=1)
        
        # Apply KMeans
        with st.spinner("Đang phân cụm dữ liệu..."):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            df["Cluster"] = kmeans.fit_predict(df[quantitative_cols])
        
        # ==========================================================
        # CLUSTER VISUALIZATION
        # ==========================================================
        st.subheader("📊 Trực quan hóa phân cụm")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scatter Plot: Open vs Close**")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df["Open"], df["Close/Last"], 
                               c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, 
                               edgecolors='black', linewidth=0.5)
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                      c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                      label='Centroids', zorder=5)
            ax.set_xlabel("Open Price ($)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Close Price ($)", fontsize=11, fontweight='bold')
            ax.set_title(f"K-Means Clustering (K={k})", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        
        with col2:
            st.write("**Scatter Plot: Low vs High**")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df["Low"], df["High"],
                               c=df["Cluster"], cmap='Set2', s=80, alpha=0.6, 
                               edgecolors='black', linewidth=0.5)
            ax.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
                      c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                      label='Centroids', zorder=5)
            ax.set_xlabel("Low Price ($)", fontsize=11, fontweight='bold')
            ax.set_ylabel("High Price ($)", fontsize=11, fontweight='bold')
            ax.set_title(f"K-Means Clustering (K={k})", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        
        # Time series with clusters
        st.write("### ⏰ Phân bố cụm theo thời gian")
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, k))
        for cluster in range(k):
            cluster_data = df[df["Cluster"] == cluster]
            ax.scatter(cluster_data["Date"], cluster_data["Close/Last"],
                      label=f"Cluster {cluster}", alpha=0.6, s=40, color=colors[cluster])
        ax.set_xlabel("Date", fontsize=11, fontweight='bold')
        ax.set_ylabel("Close Price ($)", fontsize=11, fontweight='bold')
        ax.set_title(f"Gold Price with K-Means Clusters (K={k})", fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # ==========================================================
        # CLUSTER STATISTICS
        # ==========================================================
        st.write("### 📊 Thống kê chi tiết từng cụm")
        cluster_stats = df.groupby("Cluster")[quantitative_cols].agg(['mean', 'min', 'max', 'std'])
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Cluster sizes
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Kích thước cụm")
            cluster_sizes = df["Cluster"].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors, 
                   edgecolor='black', linewidth=1.5)
            ax.set_xlabel("Cluster", fontsize=11, fontweight='bold')
            ax.set_ylabel("Number of Data Points", fontsize=11, fontweight='bold')
            ax.set_title(f"Cluster Size Distribution (K={k})", fontsize=12, fontweight='bold')
            ax.set_xticks(range(k))
            for i, v in enumerate(cluster_sizes.values):
                ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)
        
        with col2:
            st.write("### 📊 Tỷ lệ phần trăm")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(cluster_sizes.values, 
                   labels=[f"Cluster {i}\n({v} points)" for i, v in enumerate(cluster_sizes.values)],
                   colors=colors, autopct='%1.1f%%', startangle=90, explode=[0.05]*k)
            ax.set_title(f"Cluster Distribution (K={k})", fontsize=12, fontweight='bold')
            st.pyplot(fig)
        
        # ==========================================================
        # CLUSTER CHARACTERISTICS
        # ==========================================================
        st.write("### 🔍 Đặc điểm của từng cụm")
        for cluster in range(k):
            with st.expander(f"📌 Cluster {cluster} - {len(df[df['Cluster'] == cluster])} điểm dữ liệu"):
                cluster_data = df[df["Cluster"] == cluster]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Giá Close TB", f"${cluster_data['Close/Last'].mean():.2f}")
                    st.metric("Giá Min", f"${cluster_data['Close/Last'].min():.2f}")
                
                with col2:
                    st.metric("Giá Max", f"${cluster_data['Close/Last'].max():.2f}")
                    st.metric("Volume TB", f"{cluster_data['Volume'].mean():,.0f}")
                
                with col3:
                    st.metric(
                        "Từ ngày",
                        cluster_data['Date'].min().strftime("%d/%m/%Y")
                    )
                    st.metric(
                        "Đến ngày",
                        cluster_data['Date'].max().strftime("%d/%m/%Y")
                    )
        
        # ==========================================================
        # MODEL EVALUATION
        # ==========================================================
        st.write("### 📊 Đánh giá mô hình K-Means")
        
        inertia = kmeans.inertia_
        silhouette = silhouette_score(df[quantitative_cols], df["Cluster"])
        davies_bouldin = davies_bouldin_score(df[quantitative_cols], df["Cluster"])
        calinski = calinski_harabasz_score(df[quantitative_cols], df["Cluster"])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Inertia", f"{inertia:.2f}")
            st.caption("Lower is better")
        
        with col2:
            st.metric("Silhouette Score", f"{silhouette:.4f}")
            st.caption("Higher is better (-1 to 1)")
        
        with col3:
            st.metric("Davies-Bouldin", f"{davies_bouldin:.4f}")
            st.caption("Lower is better")
        
        with col4:
            st.metric("Calinski-Harabasz", f"{calinski:.2f}")
            st.caption("Higher is better")
        
        # Interpretation
        if silhouette > 0.5:
            st.success(f"✅ Silhouette Score = {silhouette:.4f} - Phân cụm TỐT")
        elif silhouette > 0.3:
            st.warning(f"⚠️ Silhouette Score = {silhouette:.4f} - Phân cụm TRUNG BÌNH")
        else:
            st.error(f"❌ Silhouette Score = {silhouette:.4f} - Phân cụm CẦN CẢI THIỆN")
        
        # Sample data
        st.write("### 📝 Dữ liệu mẫu sau phân cụm")
        display_cols = ["Date", "Open", "High", "Low", "Close/Last", "Volume", "Cluster"]
        st.dataframe(df[display_cols].head(20), use_container_width=True)

    # ==========================================================
    # TAB 2: LINEAR REGRESSION
    # ==========================================================
    with tab2:
        st.header("📈 Dự đoán giá vàng (Linear Regression)")
        
        st.info("🔮 **Mô hình Linear Regression để dự đoán giá vàng đến năm 2027**")
        
        # Prepare data
        df_model = df.copy()
        df_model['Days'] = (df_model['Date'] - df_model['Date'].min()).dt.days
        
        X = df_model[['Days']].values
        y = df_model['Close/Last'].values
        
        # Train model
        with st.spinner("Đang huấn luyện mô hình..."):
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            y_pred_train = lr_model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred_train)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred_train)
        r2 = r2_score(y, y_pred_train)
        
        # Display metrics
        st.write("### 📊 Hiệu suất mô hình")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
            st.caption("Closer to 1 is better")
        
        with col2:
            st.metric("RMSE", f"${rmse:.2f}")
            st.caption("Root Mean Squared Error")
        
        with col3:
            st.metric("MAE", f"${mae:.2f}")
            st.caption("Mean Absolute Error")
        
        with col4:
            st.metric("MSE", f"${mse:.2f}")
            st.caption("Mean Squared Error")
        
        # Model interpretation
        if r2 > 0.7:
            st.success(f"✅ R² = {r2:.4f} - Mô hình TỐT")
        elif r2 > 0.5:
            st.warning(f"⚠️ R² = {r2:.4f} - Mô hình TRUNG BÌNH")
        else:
            st.error(f"❌ R² = {r2:.4f} - Mô hình YẾU")
        
        # ==========================================================
        # FUTURE PREDICTION
        # ==========================================================
        st.write("### 🔮 Dự đoán tương lai")
        
        # Sidebar controls
        st.sidebar.header("⚙️ Prediction Settings")
        end_year = st.sidebar.selectbox("Dự đoán đến năm", [2025, 2026, 2027, 2028, 2030], index=2)
        
        # Generate future dates
        last_date = df_model['Date'].max()
        target_date = pd.Timestamp(f'{end_year}-12-31')
        
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=target_date, freq='D')
        future_days = (future_dates - df_model['Date'].min()).days.values.reshape(-1, 1)
        
        # Predict
        future_prices = lr_model.predict(future_days)
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })
        
        # ==========================================================
        # VISUALIZATION
        # ==========================================================
        st.write("### 📊 Biểu đồ dự đoán")
        
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Historical actual
        ax.plot(df_model['Date'], df_model['Close/Last'], 
                linewidth=2, color='steelblue', label='Historical Actual Price', alpha=0.8)
        
        # Historical fitted
        ax.plot(df_model['Date'], y_pred_train, 
                linewidth=2, color='orange', linestyle='--', label='Linear Regression Fit', alpha=0.7)
        
        # Future prediction
        ax.plot(future_df['Date'], future_df['Predicted_Price'], 
                linewidth=2.5, color='red', linestyle='-', label=f'Future Prediction (to {end_year})', alpha=0.8)
        
        # Confidence interval
        std_error = np.std(y - y_pred_train)
        ax.fill_between(future_df['Date'], 
                        future_df['Predicted_Price'] - 1.96*std_error,
                        future_df['Predicted_Price'] + 1.96*std_error,
                        alpha=0.2, color='red', label='95% Confidence Interval')
        
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel("Price (USD/oz)", fontsize=12, fontweight='bold')
        ax.set_title(f"Gold Price Prediction using Linear Regression (to {end_year})", 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ==========================================================
        # PREDICTION STATISTICS
        # ==========================================================
        st.write("### 📊 Thống kê dự đoán")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dự đoán giá vàng:**")
            prediction_stats = {
                "Date": [
                    f"Last Historical ({last_date.date()})",
                    f"End of 2025",
                    f"End of 2026",
                    f"End of {end_year}"
                ],
                "Predicted Price": [
                    f"${df_model['Close/Last'].iloc[-1]:.2f}",
                    f"${lr_model.predict([[((pd.Timestamp('2025-12-31') - df_model['Date'].min()).days)]])[0]:.2f}",
                    f"${lr_model.predict([[((pd.Timestamp('2026-12-31') - df_model['Date'].min()).days)]])[0]:.2f}",
                    f"${lr_model.predict([[((pd.Timestamp(f'{end_year}-12-31') - df_model['Date'].min()).days)]])[0]:.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(prediction_stats), use_container_width=True)
        
        with col2:
            st.write("**Thông số mô hình:**")
            model_params = {
                "Parameter": [
                    "Slope (Hệ số góc)",
                    "Intercept (Hằng số)",
                    "Daily Price Change",
                    "Yearly Price Change"
                ],
                "Value": [
                    f"{lr_model.coef_[0]:.4f}",
                    f"${lr_model.intercept_:.2f}",
                    f"${lr_model.coef_[0]:.4f}/day",
                    f"${lr_model.coef_[0]*365:.2f}/year"
                ]
            }
            st.dataframe(pd.DataFrame(model_params), use_container_width=True)
        
        # Model equation
        st.write("### 📐 Phương trình hồi quy")
        st.latex(f"Price = {lr_model.intercept_:.2f} + {lr_model.coef_[0]:.4f} \\times Days")
        
        # ==========================================================
        # INTERPRETATION
        # ==========================================================
        st.write("### 💡 Giải thích kết quả")
        
        trend_direction = "tăng" if lr_model.coef_[0] > 0 else "giảm"
        trend_emoji = "📈" if lr_model.coef_[0] > 0 else "📉"
        
        st.markdown(f"""
        **Ý nghĩa các chỉ số:**
        - **R² = {r2:.4f}**: Mô hình giải thích {r2*100:.2f}% sự biến động của giá vàng
        - **RMSE = ${rmse:.2f}**: Sai số trung bình khoảng ${rmse:.2f}
        - **Slope = {lr_model.coef_[0]:.4f}**: Giá vàng {trend_direction} trung bình ${abs(lr_model.coef_[0]):.4f}/ngày
        
        **Xu hướng:**
        {trend_emoji} Giá vàng có xu hướng **{trend_direction}** đều đặn với tốc độ **${abs(lr_model.coef_[0]*365):.2f}/năm**
        
        **Dự đoán đến {end_year}:**
        - Giá dự kiến: **${lr_model.predict([[((pd.Timestamp(f'{end_year}-12-31') - df_model['Date'].min()).days)]])[0]:.2f}**
        - Khoảng tin cậy 95%: **${lr_model.predict([[((pd.Timestamp(f'{end_year}-12-31') - df_model['Date'].min()).days)]])[0] - 1.96*std_error:.2f}** - **${lr_model.predict([[((pd.Timestamp(f'{end_year}-12-31') - df_model['Date'].min()).days)]])[0] + 1.96*std_error:.2f}**
        """)
        
        st.warning("⚠️ **Lưu ý:** Dự đoán dài hạn với Linear Regression có thể không chính xác do giả định xu hướng tuyến tính. Giá vàng bị ảnh hưởng bởi nhiều yếu tố kinh tế, chính trị phức tạp.")
        
        # ==========================================================
        # DOWNLOAD DATA
        # ==========================================================
        st.write("### 💾 Tải dữ liệu dự đoán")
        
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
            file_name=f"gold_price_prediction_to_{end_year}.csv",
            mime="text/csv"
        )

    # ==========================================================
    # FOOTER
    # ==========================================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><b>B5 - Machine Learning Model & GUI</b></p>
        <p>Gold Price Data Mining Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Gold Price Data Mining - B5", layout="wide", page_icon="📊")
    render_app()
