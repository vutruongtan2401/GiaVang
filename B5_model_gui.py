# ==========================================================
# B5 – LINEAR REGRESSION DỰ BÁO GIÁ VÀNG
# Mô hình đơn giản dựa trên dữ liệu đã làm sạch B2
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import timedelta

plt.style.use('ggplot')
sns.set_palette("husl")

def load_data():
    """Load dữ liệu đã làm sạch và verify với raw data"""
    try:
        # Load cleaned data
        df = pd.read_csv("goldstock_cleaned_B2.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        
        # Verify với raw data
        df_raw = pd.read_csv("goldstock v2.csv", sep=";")
        df_raw.columns = df_raw.columns.str.strip()
        
        # Kiểm tra thứ tự: cleaned phải giữ thứ tự raw (2024 trước)
        assert df["Date"].iloc[0] > df["Date"].iloc[-1], "❌ Dữ liệu không đúng thứ tự!"
        
        return df
    except Exception as e:
        st.error(f"❌ Lỗi load dữ liệu: {e}")
        return None

def create_features(df):
    """Tạo features với trend + seasonal components"""
    df = df.copy()
    
    # Sort ascending cho time series (model cần thứ tự cũ -> mới)
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. TIME FEATURES - Linear trend
    df['days'] = (df['Date'] - df['Date'].min()).dt.days
    df['days_normalized'] = df['days'] / df['days'].max()
    
    # 2. SEASONAL FEATURES - Cyclical patterns
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 3. LAG FEATURES - Giá ngày trước
    df['close_lag_1'] = df['Close/Last'].shift(1)
    df['close_lag_7'] = df['Close/Last'].shift(7)
    df['close_lag_30'] = df['Close/Last'].shift(30)
    
    # 4. MOVING AVERAGES - Multi-timeframe trends
    df['ma_7'] = df['Close/Last'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['Close/Last'].rolling(window=30, min_periods=1).mean()
    df['ma_90'] = df['Close/Last'].rolling(window=90, min_periods=1).mean()
    
    # 5. TREND INDICATORS
    df['trend_7_30'] = df['ma_7'] - df['ma_30']  # Short-term trend
    df['trend_30_90'] = df['ma_30'] - df['ma_90']  # Long-term trend
    
    # 6. MOMENTUM
    df['momentum_7'] = df['Close/Last'].diff(7)
    df['momentum_30'] = df['Close/Last'].diff(30)
    
    # 7. VOLATILITY - Biến động
    df['volatility_7'] = df['Close/Last'].rolling(window=7, min_periods=1).std()
    df['volatility_30'] = df['Close/Last'].rolling(window=30, min_periods=1).std()
    
    # Remove rows with NaN
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_model(df):
    """Train Linear Regression model"""
    
    # Features
    feature_cols = [
        'days_normalized',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'close_lag_1', 'close_lag_7', 'close_lag_30',
        'ma_7', 'ma_30', 'ma_90',
        'trend_7_30', 'trend_30_90',
        'momentum_7', 'momentum_30',
        'volatility_7', 'volatility_30'
    ]
    
    # Train/Test split (80/20)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    X_train = df_train[feature_cols].values
    y_train = df_train['Close/Last'].values
    X_test = df_test[feature_cols].values
    y_test = df_test['Close/Last'].values
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Metrics
    metrics = {
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mape': np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    }
    
    return model, scaler, feature_cols, df_train, df_test, y_test, y_test_pred, metrics

def predict_future(model, scaler, df, feature_cols, n_days):
    """Dự đoán với trend extrapolation + realistic volatility"""
    
    predictions = []
    future_dates = []
    
    # Tính trend từ 30 ngày gần nhất
    recent_30 = df['Close/Last'].tail(30).values
    linear_trend = np.polyfit(range(len(recent_30)), recent_30, 1)[0]  # Slope
    
    # Tính volatility từ historical
    historical_volatility = df['Close/Last'].tail(90).std()
    
    # Làm việc với bản copy
    df_pred = df.copy()
    last_date = df_pred['Date'].iloc[-1]
    
    for i in range(n_days):
        # Ngày tiếp theo
        next_date = last_date + timedelta(days=i+1)
        future_dates.append(next_date)
        
        # Time features
        days_total = (next_date - df['Date'].min()).days
        days_normalized = days_total / df['days'].max()
        
        # Seasonal features
        day_of_week = next_date.dayofweek
        month = next_date.month
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Lag features (mix historical + predictions)
        if i == 0:
            close_lag_1 = df_pred['Close/Last'].iloc[-1]
            close_lag_7 = df_pred['Close/Last'].iloc[-7] if len(df_pred) >= 7 else close_lag_1
            close_lag_30 = df_pred['Close/Last'].iloc[-30] if len(df_pred) >= 30 else close_lag_1
        else:
            close_lag_1 = predictions[-1]
            if len(predictions) >= 7:
                close_lag_7 = predictions[-7]
            else:
                close_lag_7 = df_pred['Close/Last'].iloc[-(7-i)] if len(df_pred) >= (7-i) else close_lag_1
            
            if len(predictions) >= 30:
                close_lag_30 = predictions[-30]
            else:
                close_lag_30 = df_pred['Close/Last'].iloc[-(30-i)] if len(df_pred) >= (30-i) else close_lag_1
        
        # MA features (combine historical + predictions)
        if i < 7:
            recent_prices = list(df_pred['Close/Last'].tail(7-i).values) + predictions[:i]
            ma_7 = np.mean(recent_prices)
        else:
            ma_7 = np.mean(predictions[-7:])
        
        if i < 30:
            recent_prices_30 = list(df_pred['Close/Last'].tail(30-i).values) + predictions[:i]
            ma_30 = np.mean(recent_prices_30)
        else:
            ma_30 = np.mean(predictions[-30:])
        
        if i < 90:
            recent_prices_90 = list(df_pred['Close/Last'].tail(90-i).values) + predictions[:i]
            ma_90 = np.mean(recent_prices_90)
        else:
            ma_90 = np.mean(predictions[-90:])
        
        # Trend indicators
        trend_7_30 = ma_7 - ma_30
        trend_30_90 = ma_30 - ma_90
        
        # Momentum
        momentum_7 = close_lag_1 - close_lag_7
        momentum_30 = close_lag_1 - close_lag_30
        
        # Volatility
        if i < 7:
            recent_vol_7 = list(df_pred['Close/Last'].tail(7-i).values) + predictions[:i]
            volatility_7 = np.std(recent_vol_7) if len(recent_vol_7) > 1 else historical_volatility
        else:
            volatility_7 = np.std(predictions[-7:])
        
        if i < 30:
            recent_vol_30 = list(df_pred['Close/Last'].tail(30-i).values) + predictions[:i]
            volatility_30 = np.std(recent_vol_30) if len(recent_vol_30) > 1 else historical_volatility
        else:
            volatility_30 = np.std(predictions[-30:])
        
        # Build feature vector
        X_future = np.array([[
            days_normalized,
            day_sin, day_cos, month_sin, month_cos,
            close_lag_1, close_lag_7, close_lag_30,
            ma_7, ma_30, ma_90,
            trend_7_30, trend_30_90,
            momentum_7, momentum_30,
            volatility_7, volatility_30
        ]])
        
        # Predict
        X_scaled = scaler.transform(X_future)
        pred = model.predict(X_scaled)[0]
        
        # Add trend component + small random noise for realism
        trend_component = linear_trend * 0.3  # Giảm 70% trend để không quá mạnh
        noise = np.random.normal(0, historical_volatility * 0.1)  # 10% volatility
        
        pred = pred + trend_component + noise
        predictions.append(pred)
    
    return predictions, future_dates

def plot_predictions(df, predictions, future_dates, n_days, title):
    """Vẽ biểu đồ dự đoán"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Historical data (90 ngày gần nhất)
    df_recent = df.tail(90)
    ax.plot(df_recent['Date'], df_recent['Close/Last'], 
            label='Lịch sử', color='steelblue', linewidth=2, marker='o', markersize=3)
    
    # Last point
    last_date = df['Date'].iloc[-1]
    last_price = df['Close/Last'].iloc[-1]
    
    # Bridge (nối lịch sử với dự đoán)
    ax.plot([last_date, future_dates[0]], [last_price, predictions[0]], 
            color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Predictions
    ax.plot(future_dates, predictions, 
            label=f'Dự đoán {n_days} ngày', color='orange', 
            linewidth=2, marker='s', markersize=4, linestyle='--')
    
    # Today line
    ax.axvline(x=last_date, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Hôm nay')
    
    ax.set_xlabel('Ngày', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá Close (USD)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def render_app():
    """Streamlit App"""
    
    st.title("📈 B5 — Linear Regression: Dự Báo Giá Vàng")
    st.markdown("---")
    
    # Load data
    with st.spinner("📂 Đang load dữ liệu..."):
        df = load_data()
    
    if df is None:
        return
    
    st.success(f"✅ Đã load {len(df)} dòng | Từ {df['Date'].min().date()} đến {df['Date'].max().date()}")
    
    # Verify thứ tự
    if df['Date'].iloc[0] > df['Date'].iloc[-1]:
        st.info("✅ Dữ liệu đúng thứ tự: Mới nhất trước (2024 → 2014)")
    else:
        st.warning("⚠️ Dữ liệu đã được sort ascending cho model")
    
    # Create features
    with st.spinner("🔧 Tạo features..."):
        df_features = create_features(df)
    
    st.success(f"✅ Đã tạo features | {len(df_features)} dòng sau khi xử lý")
    
    # Train model
    with st.spinner("🎓 Training model..."):
        model, scaler, feature_cols, df_train, df_test, y_test, y_test_pred, metrics = train_model(df_features)
    
    # Show metrics
    st.markdown("### 📊 Đánh Giá Model")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Train R²", f"{metrics['train_r2']:.4f}")
    col2.metric("Test R²", f"{metrics['test_r2']:.4f}")
    col3.metric("MAE", f"${metrics['test_mae']:.2f}")
    col4.metric("RMSE", f"${metrics['test_rmse']:.2f}")
    col5.metric("MAPE", f"{metrics['test_mape']:.2f}%")
    
    # Đánh giá
    if metrics['test_r2'] > 0.9:
        st.success("✅ Model rất tốt (R² > 0.9)")
    elif metrics['test_r2'] > 0.8:
        st.info("✅ Model tốt (R² > 0.8)")
    else:
        st.warning("⚠️ Model cần cải thiện (R² < 0.8)")
    
    st.markdown("---")
    
    # Tabs cho dự đoán
    tab1, tab2, tab3 = st.tabs(["📅 30 Ngày", "📅 60 Ngày", "🎯 Tùy Chỉnh"])
    
    with tab1:
        st.subheader("Dự đoán 30 ngày tiếp theo")
        
        with st.spinner("🔮 Đang dự đoán..."):
            preds_30, dates_30 = predict_future(model, scaler, df_features, feature_cols, 30)
        
        # Plot
        fig = plot_predictions(df_features, preds_30, dates_30, 30, 
                              "Dự đoán giá vàng 30 ngày tiếp theo")
        st.pyplot(fig)
        
        # Stats
        st.markdown("**📈 Thống kê dự đoán:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Giá TB", f"${np.mean(preds_30):.2f}")
        col2.metric("Giá cao nhất", f"${max(preds_30):.2f}")
        col3.metric("Giá thấp nhất", f"${min(preds_30):.2f}")
        
        # Table
        df_pred = pd.DataFrame({
            'Ngày': dates_30,
            'Giá dự đoán ($)': [f"{p:.2f}" for p in preds_30]
        })
        st.dataframe(df_pred, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("Dự đoán 60 ngày tiếp theo")
        
        with st.spinner("🔮 Đang dự đoán..."):
            preds_60, dates_60 = predict_future(model, scaler, df_features, feature_cols, 60)
        
        # Plot
        fig = plot_predictions(df_features, preds_60, dates_60, 60, 
                              "Dự đoán giá vàng 60 ngày tiếp theo")
        st.pyplot(fig)
        
        # Stats
        st.markdown("**📈 Thống kê dự đoán:**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giá TB", f"${np.mean(preds_60):.2f}")
        col2.metric("Giá cao nhất", f"${max(preds_60):.2f}")
        col3.metric("Giá thấp nhất", f"${min(preds_60):.2f}")
        col4.metric("Biên độ", f"${max(preds_60) - min(preds_60):.2f}")
        
        # Weekly summary
        df_pred_60 = pd.DataFrame({
            'Date': dates_60,
            'Price': preds_60
        })
        df_pred_60['Week'] = (df_pred_60.index // 7) + 1
        summary = df_pred_60.groupby('Week')['Price'].agg(['mean', 'min', 'max']).reset_index()
        summary.columns = ['Tuần', 'Giá TB ($)', 'Giá Min ($)', 'Giá Max ($)']
        summary['Tuần'] = 'Tuần ' + summary['Tuần'].astype(str)
        
        st.dataframe(summary.style.format({
            'Giá TB ($)': '{:.2f}',
            'Giá Min ($)': '{:.2f}',
            'Giá Max ($)': '{:.2f}'
        }), use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Chọn ngày cụ thể để dự đoán")
        
        # Lấy ngày cuối cùng của data
        last_date = df_features['Date'].iloc[-1].date()
        
        # Date picker
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_date = st.date_input(
                "Chọn ngày dự đoán:",
                value=last_date + timedelta(days=30),
                min_value=last_date + timedelta(days=1),
                max_value=last_date + timedelta(days=120)
            )
        
        with col2:
            n_days = (selected_date - last_date).days
            st.metric("Số ngày từ hôm nay", f"{n_days} ngày")
        
        st.info(f"📅 Ngày cuối trong dữ liệu: **{last_date}** | Ngày dự đoán: **{selected_date}**")
        
        if st.button("🔮 Dự đoán", type="primary"):
            with st.spinner(f"🔮 Đang dự đoán đến {selected_date}..."):
                preds, dates = predict_future(model, scaler, df_features, feature_cols, n_days)
            
            # Plot
            fig = plot_predictions(df_features, preds, dates, n_days, 
                                  f"Dự đoán giá vàng đến ngày {selected_date}")
            st.pyplot(fig)
            
            # Highlight giá ngày được chọn
            predicted_price = preds[-1]
            st.success(f"💰 **Giá dự đoán ngày {selected_date}: ${predicted_price:.2f}**")
            
            # Stats
            st.markdown("**📈 Thống kê dự đoán:**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Giá TB", f"${np.mean(preds):.2f}")
            col2.metric("Giá cao nhất", f"${max(preds):.2f}")
            col3.metric("Giá thấp nhất", f"${min(preds):.2f}")
            
            # So sánh với hiện tại
            current_price = df_features['Close/Last'].iloc[-1]
            change = predicted_price - current_price
            change_pct = (change / current_price) * 100
            
            if change > 0:
                col4.metric(f"Thay đổi", f"+${change:.2f}", f"+{change_pct:.2f}%")
            else:
                col4.metric(f"Thay đổi", f"${change:.2f}", f"{change_pct:.2f}%")

if __name__ == "__main__":
    render_app()
