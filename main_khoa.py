import io
import os
import contextlib
import pandas as pd
import streamlit as st

# Import step modules
import B1_data_description as B1
import B2_data_cleaning as B2
import B3_data_exploration_streamlit as B3
import B4_correlation_pca as B4
import B5_model_gui as B5

st.set_page_config(page_title="Gold Price Project - B1→B5", layout="wide", page_icon="💰")

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Override Streamlit default background */
    .main {
        background-color: #000000;
    }
    
    .stApp {
        background: #000000;
    }
    
    /* Main title styling with animation */
    .main-title {
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 30%, #FF8C00 60%, #FFD700 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: shine 3s linear infinite;
        letter-spacing: 2px;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.1);
    }
    
    /* Tab styling with gradient */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 12px 24px;
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 12px;
        font-weight: 600;
        color: #e0e0e0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        box-shadow: 0 2px 8px rgba(255,215,0,0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-color: #FFD700;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [aria-selected="true"] button {
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab"] button {
        color: inherit;
    }
    
    /* Info boxes styling */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid #FFD700;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling with gradient */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        color: white;
        box-shadow: 4px 0 20px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h2 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 215, 0, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%) !important;
        color: #FFD700 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #FFD700 !important;
    }
    
    .dataframe tbody tr {
        background-color: rgba(26, 26, 26, 0.5) !important;
        transition: background-color 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(255, 215, 0, 0.1) !important;
    }
    
    /* Professional Cards */
    .pro-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .pro-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.3);
        border-color: #FFD700;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #000;
    }
    
    .status-info {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
    }
    
    /* Progress Indicator */
    .progress-bar {
        width: 100%;
        height: 6px;
        background: rgba(255, 215, 0, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #FFD700 0%, #FFA500 100%);
        border-radius: 10px;
        animation: progressAnimation 2s ease-in-out;
    }
    
    @keyframes progressAnimation {
        from { width: 0%; }
        to { width: 100%; }
    }
    
    /* Section Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #FFD700 50%, transparent 100%);
        margin: 30px 0;
    }
    
    /* Info Box Professional */
    .info-box {
        background: linear-gradient(135deg, rgba(23, 162, 184, 0.15) 0%, rgba(19, 132, 150, 0.15) 100%);
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 15px 0;
        color: #87ceeb;
    }
    
    /* Code Block Enhancement */
    .stCodeBlock {
        background: #1a1a1a !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Metric Enhancement */
    [data-testid="stMetricLabel"] {
        color: #b0b0b0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Image styling */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 215, 0, 0.2);
    }
    
    img:hover {
        transform: scale(1.03);
        box-shadow: 0 8px 35px rgba(255, 215, 0, 0.4);
        border-color: #FFD700;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 10px;
        font-weight: 600;
        padding: 12px;
        transition: all 0.3s ease;
        color: #e0e0e0;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, #3a3a3a 0%, #2a2a2a 100%);
        box-shadow: 0 2px 8px rgba(255,215,0,0.3);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #FFD700;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Caption styling */
    .caption {
        color: #b0b0b0;
        font-style: italic;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #FFD700 transparent transparent transparent !important;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.5);
    }
    
    /* File Info Display */
    .file-info {
        background: rgba(255, 215, 0, 0.05);
        border-left: 3px solid #FFD700;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    
    /* Success/Info/Warning box enhancements */
    .stSuccess {
        background: linear-gradient(135deg, #1a3a1a 0%, #153515 100%);
        border-radius: 12px;
        border-left: 5px solid #28a745;
        color: #90ee90;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #1a2a3a 0%, #152535 100%);
        border-radius: 12px;
        border-left: 5px solid #17a2b8;
        color: #87ceeb;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #3a3a1a 0%, #353515 100%);
        border-radius: 12px;
        border-left: 5px solid #ffc107;
        color: #ffeb3b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">💰 Gold Price Data Mining Pipeline</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📊 Phân Tích & Dự Đoán Giá Vàng Toàn Diện | B1 → B2 → B3 → B4 → B5</p>', unsafe_allow_html=True)

# Add visual separator
st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌟 Dự Án Data Mining")
st.sidebar.markdown(*🏆 Phân Tích & Dự Đoán Giá Vàng**")
st.sidebar.info("""
🔬 *Phương pháp:* Linear Regression  
📈 *Dataset:* 2511 ngày (2014-2024)  
🎯 *Mục tiêu:* Dự đoán giá vàng ngắn hạn
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Nhóm Thực Hiện")
st.sidebar.markdown("""
<div style='background: rgba(255, 215, 0, 0.1); padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
    <div style='text-align: center; margin-bottom: 8px;'>
        <b style='color: #FFD700; font-size: 1.1rem;'>Nguyễn Lê Đăng Khoa</b>
    </div>
    <div style='text-align: center; color: #ddd;'>MSSV: 23AI023</div>
</div>
<div style='background: rgba(255, 215, 0, 0.1); padding: 15px; border-radius: 10px;'>
    <div style='text-align: center; margin-bottom: 8px;'>
        <b style='color: #FFD700; font-size: 1.1rem;'>Trương Tấn Vũ</b>
    </div>
    <div style='text-align: center; color: #ddd;'>MSSV: 23AI056</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.success("✨ *Pipeline:* B1 → B2 → B3 → B4 → B5")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; margin-top: 20px;'>
    <small style='color: #aaa;'>Made with ❤️ using Streamlit</small>
</div>
""", unsafe_allow_html=True)

# Helper: capture stdout from run() functions
@contextlib.contextmanager
def capture_stdout():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer

# Helper: show file if exists
def show_file_head(path: str, n: int = None):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            st.write(f"📄 {os.path.basename(path)} — {len(df)} hàng, {df.shape[1]} cột")
            if n is None:
                st.dataframe(df, use_container_width=True, height=600)
            else:
                st.dataframe(df.head(n), use_container_width=True)
        except Exception as e:
            st.warning(f"Không thể đọc {path}: {e}")
    else:
        st.info(f"Chưa thấy file: {path}")

# Auto-run all preprocessing steps on first load
@st.cache_data
def run_all_preprocessing():
    """Run B1-B2-B4 once and cache results"""
    logs = {}
    
    # B1
    with capture_stdout() as buf:
        B1.run()
    logs['B1'] = buf.getvalue()
    
    # B2
    with capture_stdout() as buf:
        B2.run()
    logs['B2'] = buf.getvalue()
    
    # B4
    with capture_stdout() as buf:
        B4.run()
    logs['B4'] = buf.getvalue()
    
    return logs

# Run preprocessing automatically
with st.spinner("🔄 Đang xử lý dữ liệu (B1, B2, B4)..."):
    preprocessing_logs = run_all_preprocessing()

# Tabs for steps
TAB_B1, TAB_B2, TAB_B3, TAB_B4, TAB_B5 = st.tabs([
    "📋 B1: Mô tả dữ liệu",
    "🧹 B2: Làm sạch dữ liệu",
    "🔍 B3: Khám phá dữ liệu",
    "📊 B4: Tương quan & PCA",
    "🤖 B5: Linear Regression"
])


with TAB_B1:
    st.markdown("### 📋 B1 — Mô tả Dữ liệu")
    st.caption("📝 Tải và chuẩn hóa dữ liệu, mô tả thống kê, phân loại định lượng/định tính.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Bước", "B1", delta="Hoàn thành")
    with col2:
        st.metric("📄 Output", "goldstock_processed_B1.csv")
    with col3:
        st.metric("🔧 Chức năng", "Mô tả & Chuẩn hóa")
    
    with st.expander("📜 Nhật ký chạy B1", expanded=False):
        st.code(preprocessing_logs['B1'], language='text')
    
    show_file_head("goldstock_processed_B1.csv", n=None)  # Hiển thị toàn bộ data

with TAB_B2:
    st.markdown("### 🧹 B2 — Làm Sạch Dữ liệu")
    st.caption("🛠️ Xử lý thiếu, trùng, logic giá và phát hiện ngoại lệ.")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Bước", "B2", delta="Hoàn thành")
    with col2:
        st.metric("📄 Output", "goldstock_cleaned_B2.csv")
    with col3:
        st.metric("🔧 Chức năng", "Làm sạch & Phát hiện lỗi")
    
    with st.expander("📜 Nhật ký chạy B2", expanded=False):
        st.code(preprocessing_logs['B2'], language='text')
    
    show_file_head("goldstock_cleaned_B2.csv")
    
    # Show outlier plot if exists
    outlier_png = "B2_outliers_detection.png"
    if os.path.exists(outlier_png):
        st.markdown("#### 📈 Phát hiện ngoại lệ")
        st.image(outlier_png, caption="🎯 Phát hiện ngoại lệ bằng phương pháp IQR", use_container_width=True)

with TAB_B3:
    st.header("B3 — Khám phá dữ liệu (EDA)")
    st.caption("Phân tích xu hướng giá, thanh khoản, mối quan hệ Volume-Price và biến động theo năm.")
    
    # Load data for comments
    try:
        df_b3 = pd.read_csv("goldstock_cleaned_B2.csv")
        df_b3["Date"] = pd.to_datetime(df_b3["Date"])
        df_b3['Year'] = df_b3['Date'].dt.year
        df_b3['Price_Range'] = df_b3['High'] - df_b3['Low']
    except:
        df_b3 = None
    
    # 1. Line Chart - Close Price
    if os.path.exists("B3_line_chart_close_price.png"):
        st.subheader("📈 1. Xu hướng giá vàng theo thời gian")
        st.image("B3_line_chart_close_price.png", use_container_width=True)
        
        if df_b3 is not None:
            # Tính toán các chỉ số
            pct_change_total = ((df_b3['Close/Last'].iloc[-1] - df_b3['Close/Last'].iloc[0]) / df_b3['Close/Last'].iloc[0] * 100)
            
            # Tính giá tăng trung bình hằng năm
            num_years = (df_b3['Date'].max() - df_b3['Date'].min()).days / 365.25
            avg_annual_growth = pct_change_total / num_years
            
            # Tìm năm tăng mạnh nhất
            yearly_growth = {}
            for year in sorted(df_b3['Year'].unique()):
                year_data = df_b3[df_b3['Year'] == year].sort_values('Date')
                if len(year_data) > 1:
                    start_price = year_data['Close/Last'].iloc[0]
                    end_price = year_data['Close/Last'].iloc[-1]
                    growth = ((end_price - start_price) / start_price * 100)
                    yearly_growth[year] = growth
            
            best_year = max(yearly_growth, key=yearly_growth.get)
            best_year_growth = yearly_growth[best_year]
            
            # Hiển thị metrics (chỉ 3 metrics chính)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Thay đổi tổng (từ đầu đến cuối)", f"+{pct_change_total:.2f}%")
            with col2:
                st.metric("Tăng trung bình hằng năm", f"+{avg_annual_growth:.2f}%/năm")
            with col3:
                st.metric(f"Năm tăng mạnh nhất ({best_year})", f"+{best_year_growth:.2f}%")
            
            st.markdown("**💡 Nhận xét:**")
            
            # Tính toán các số liệu thực tế
            avg_price = df_b3['Close/Last'].mean()
            max_price = df_b3['Close/Last'].max()
            min_price = df_b3['Close/Last'].min()
            start_price = df_b3['Close/Last'].iloc[0]
            end_price = df_b3['Close/Last'].iloc[-1]
            
            # Tính từ năm 2019 trở đi
            df_2019_onwards = df_b3[df_b3['Year'] >= 2019]
            if len(df_2019_onwards) > 0:
                avg_price_2019 = df_2019_onwards['Close/Last'].mean()
            else:
                avg_price_2019 = avg_price
            
            st.write(f"- **Xu hướng chính:** Tăng trưởng mạnh mẽ trong dài hạn (từ {df_b3['Year'].min()} đến {df_b3['Year'].max()}).")
            st.write(f"- **Mốc đột phá:** Từ năm 2019, giá vàng bắt đầu bứt phá và liên tục nằm trên mức trung bình ${avg_price_2019:,.0f}.")
            st.write(f"- **Đỉnh điểm:** Cuối năm 2023 đến 2024 chứng kiến tốc độ tăng phi mã, lập đỉnh lịch sử trên ${max_price:,.0f}/oz.")
            st.write(f"- **Tính chất:** Biến động mạnh dần theo thời gian, thể hiện giá trị tài sản trú ẩn cao (từ ${start_price:,.0f} → ${end_price:,.0f}).")
        
        st.divider()
    
    # 2. Histogram Volume
    if os.path.exists("B3_histogram_volume.png"):
        st.subheader("📊 2. So sánh Khối lượng giao dịch và Giá (2014-2024)")
        st.image("B3_histogram_volume.png", use_container_width=True)
        
        if df_b3 is not None:
            st.markdown("**📊 Chi tiết các giai đoạn thanh khoản:**")
            
            # Phase details table
            phases_data = {
                "Giai đoạn": ["2014 - 2017", "2018 - 2019", "2020 - 2022", "2023 - 2024"],
                "Đặc điểm thanh khoản": ["Tăng trưởng ổn định", "Đỉnh cao thanh khoản", "Suy giảm dần", "Phục hồi mạnh"],
                "Giá trị giao dịch TB/phiên": ["~ 25 - 45 tỷ", "~ 55 - 65 tỷ", "~ 35 - 40 tỷ", "~ 45 - 50 tỷ"],
                "Tổng giá trị trong giai đoạn": ["Thấp nhất lịch sử chu kỳ", "Cao nhất toàn giai đoạn", "Dòng tiền rút bớt", "Đang quay lại mức đỉnh cũ"]
            }
            phases_df = pd.DataFrame(phases_data)
            st.dataframe(phases_df, use_container_width=True, hide_index=True)
            
            st.markdown("**💡 Nhận xét:**")
            st.write("- **Xu hướng tổng thể:** Thanh khoản có sự biến động mạnh, chia làm hai giai đoạn rõ rệt: tăng trưởng mạnh từ 2014 đến 2018, sau đó có xu hướng sụt giảm dần và phục hồi nhẹ vào năm 2024.")
            st.write()
            st.write("- **Giai đoạn bùng nổ (2014 - 2019):** Thanh khoản tăng trưởng liên tục và duy trì ở mức rất cao, đạt đỉnh vào khoảng năm 2018 - 2019 với khối lượng giao dịch tiệm cận mức 2.5 triệu.")
            st.write()
            st.write("- **Giai đoạn sụt giảm (2020 - 2022):** Khối lượng giao dịch giảm dần qua từng năm, cho thấy sự thu hẹp về thanh khoản, chạm mức thấp nhất trong chu kỳ gần đây vào năm 2022 (khoảng 1.6 triệu).")
            st.write()
            st.write("- **Dấu hiệu phục hồi (2023 - 2024):** Thanh khoản bắt đầu có sự cải thiện trở lại trong hai năm gần nhất, đặc biệt là năm 2024 khi khối lượng vượt mức 2.0 triệu, cho thấy thị trường đang sôi động trở lại.")
        
        st.divider()
    
    # 4. Scatter Plot
    if os.path.exists("B3_scatter_volume_vs_price.png"):
        st.subheader("🔷 3. Mối quan hệ giữa Volume và Close Price")
        st.image("B3_scatter_volume_vs_price.png", use_container_width=True)
        
        if df_b3 is not None:
            correlation = df_b3['Volume'].corr(df_b3['Close/Last'])
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Hệ số tương quan (r)", f"{correlation:.4f}")
            with col2:
                st.metric("Mức độ", "Yếu", delta="Không liên quan")
            
            st.markdown("**💡 Nhận xét:**")
            st.markdown("** Mối quan hệ giữa Khối lượng và Giá**")
            st.write(f"• **Tương quan rất yếu:** Đường xu hướng (Trend line) gần như nằm ngang với hệ số tương quan cực thấp r = {correlation:.3f}. Điều này cho thấy **không có mối quan hệ tuyến tính rõ ràng** giữa khối lượng giao dịch và giá vàng.")
            st.write("• **Khối lượng tập trung:** Phần lớn các giao dịch tập trung ở mức khối lượng dưới 4 × 10⁶. Những phiên có khối lượng đột biến (trên 6 × 10⁶) thường rơi vào giai đoạn 2018–2022 nhưng không nhất thiết kéo theo mức giá cao nhất.")
            
            st.markdown("** Đặc điểm phân phối**")
            st.write("• **Giai đoạn 2014-2019:** Giá khá ổn định theo chiều ngang, biến động chủ yếu ở khối lượng giao dịch.")
            st.write("• **Giai đoạn 2020-2024:** Giá \"nhảy bậc\" lên các vùng cao mới. Đặc biệt là giai đoạn 2023–2024, giá vàng duy trì ở mức cao kỷ lục bất kể khối lượng giao dịch cao hay thấp.")
            
            st.markdown("**Tóm lại:**")
            st.write("Giá vàng chủ yếu tăng trưởng theo thời gian (yếu tố vĩ mô/chu kỳ) chứ không phụ thuộc trực tiếp vào khối lượng giao dịch trong ngày.")
        
        st.divider()
    
    # 5. Boxplot
    if os.path.exists("B3_boxplot_price_volatility_by_year.png"):
        st.subheader("📦 4. Biến động giá vàng qua các năm (Boxplot)")
        st.image("B3_boxplot_price_volatility_by_year.png", use_container_width=True)
        
        if df_b3 is not None:
            yearly_volatility = df_b3.groupby('Year')['Price_Range'].mean().sort_values()
            most_stable_year = yearly_volatility.index[0]
            most_volatile_year = yearly_volatility.index[-1]
            
            # Bảng so sánh hai giai đoạn
            comparison_data = {
                "Metric": [
                    "Biến động trung vị (Median)",
                    "Tần suất Outliers",
                    "Độ ổn định giá"
                ],
                "Giai đoạn 2014–2019": [
                    "~13.5 USD/oz",
                    "Thấp (Rải rác)",
                    "Cao (Đi ngang)"
                ],
                "Giai đoạn 2020–2024": [
                    "~26.8 USD/oz",
                    "Rất cao (Dày đặc)",
                    "Thấp (Tăng mạnh)"
                ]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            st.markdown("**💡 Nhận xét chi tiết:**")
            st.write("**1. Giải thích Boxplot:**")
            st.write("   - **Hộp (Box)**: Chứa 50% dữ liệu giữa (Q1-Q3), thể hiện biến động thông thường")
            st.write("   - **Đường đỏ (Median)**: Biến động trung vị mỗi năm")
            st.write("   - **Râu (Whiskers)**: Phạm vi biến động bình thường (không phải outlier)")
            st.write("   - **Chấm đỏ (Outliers)**: Những ngày biến động bất thường (cao/thấp đột biến)")
            
            st.write("**2. Xu hướng tăng dần:**")
            st.write("   Mức độ biến động giá vàng hàng ngày có xu hướng tăng rõ rệt, đặc biệt là từ năm 2020 trở đi. Thị trường vàng ngày càng trở nên \"nhạy cảm\" và khó lường hơn.")
            
            st.write("**3. Cột mốc 2020:**")
            st.write("   Đây là năm có sự thay đổi đột biến nhất với biên độ dao động trung bình nhảy vọt và xuất hiện nhiều ngày có mức biến động cực lớn (kỷ lục lên tới hơn 120 USD/oz).")
            
            st.write("**4. Giai đoạn ổn định (2014 - 2019):**")
            st.write("   Giá vàng khá \"hiền hòa\", biên độ dao động phần lớn duy trì ở mức thấp và ổn định (dưới 20 USD/oz).")
            
            st.write("**5. Thực trạng hiện tại (2024):**")
            st.write("   Biến động đang duy trì ở mức cao nhất trong toàn bộ giai đoạn (trung vị đạt mức xấp xỉ 30 USD/oz), cho thấy rủi ro và cơ hội lướt sóng trong ngày đều tăng cao.")

with TAB_B4:
    st.markdown("### 📊 B4 — Ma Trận Tương Quan & PCA")
    st.caption("🔗 Xác định cột giữ/bỏ theo tương quan, trực quan hóa giảm chiều.")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Bước", "B4", delta="Hoàn thành")
    with col2:
        st.metric("📄 Feature Selection", "2 features")
    with col3:
        st.metric("📄 PCA Output", "2 components")
    with col4:
        st.metric("🔧 Chức năng", "Tương quan & Giảm chiều")
    
    with st.expander("📜 Nhật ký chạy B4", expanded=False):
        st.code(preprocessing_logs['B4'], language='text')
    
    # Show files in columns
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📄 Selected Features")
        show_file_head("goldstock_selected_features_B4.csv")
    with col_right:
        st.markdown("#### 📄 PCA Components")
        show_file_head("goldstock_pca_B4.csv")
    
    st.markdown("---")
    st.markdown("### 📈 Biểu đồ Phân tích")
    
    # Show correlation & PCA plots
    for img, title, desc in [
        ("B4_correlation_matrix.png", "🔗 Ma trận Tương quan", "Phân tích tương quan giữa các biến"),
        ("B4_pca_variance_explained.png", "📊 Phương sai Giải thích", "Scree plot và phương sai tích lũy"),
        ("B4_pca_projection.png", "🎯 PCA Projection & Biplot", "Chiếu dữ liệu lên không gian 2D")
    ]:
        if os.path.exists(img):
            st.markdown(f"#### {title}")
            st.caption(desc)
            st.image(img, use_container_width=True)
            st.markdown("---")

with TAB_B5:
    st.markdown("### 🤖 B5 — Mô Hình Linear Regression & Dự đoán")
    st.caption("📈 Dự đoán giá vàng ngắn hạn (30/60/120 ngày) bằng Linear Regression.")
    st.markdown("---")
    
    # Render B5 module inside this tab
    B5.render_app()
