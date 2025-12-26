import io
import os
import glob
import contextlib
import pandas as pd
import streamlit as st

# Import step modules
import B1_data_description as B1
import B2_data_cleaning as B2
import B3_data_exploration as B3
import B4_correlation_pca as B4
import B5_model_gui as B5

st.set_page_config(page_title="Gold Price Project - B1→B5", layout="wide", page_icon="🏁")

st.title("🏁 Gold Price Data Mining — Orchestrator (B1 → B5)")

st.sidebar.success("Chọn tab để xem kết quả từng bước.")

# Helper: capture stdout from run() functions
@contextlib.contextmanager
def capture_stdout():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer

# Helper: show file if exists
def show_file_head(path: str, n: int = 10):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            st.write(f"📄 {os.path.basename(path)} — {len(df)} hàng, {df.shape[1]} cột")
            st.dataframe(df.head(n), use_container_width=True)
        except Exception as e:
            st.warning(f"Không thể đọc {path}: {e}")
    else:
        st.info(f"Chưa thấy file: {path}")

# Auto-run all preprocessing steps on first load
@st.cache_data
def run_all_preprocessing():
    """Run B1-B4 once and cache results"""
    logs = {}
    
    # B1
    with capture_stdout() as buf:
        B1.run()
    logs['B1'] = buf.getvalue()
    
    # B2
    with capture_stdout() as buf:
        B2.run()
    logs['B2'] = buf.getvalue()
    
    # B3
    with capture_stdout() as buf:
        B3.run()
    logs['B3'] = buf.getvalue()
    
    # B4
    with capture_stdout() as buf:
        B4.run()
    logs['B4'] = buf.getvalue()
    
    return logs

# Run preprocessing automatically
with st.spinner("🔄 Đang xử lý dữ liệu (B1→B4)..."):
    preprocessing_logs = run_all_preprocessing()

st.success("✅ Dữ liệu đã được xử lý sẵn (B1→B4). Chọn tab để xem chi tiết.")

# Tabs for steps
TAB_B1, TAB_B2, TAB_B3, TAB_B4, TAB_B5 = st.tabs([
    "B1: Mô tả dữ liệu",
    "B2: Làm sạch dữ liệu",
    "B3: Khám phá dữ liệu",
    "B4: Tương quan & PCA",
    "B5: Mô hình & GUI"
])


with TAB_B1:
    st.header("B1 — Mô tả dữ liệu")
    st.caption("Tải và chuẩn hóa dữ liệu, mô tả thống kê, phân loại định lượng/định tính.")
    
    with st.expander("📜 Nhật ký chạy B1", expanded=False):
        st.code(preprocessing_logs['B1'])
    
    show_file_head("goldstock_processed_B1.csv")

with TAB_B2:
    st.header("B2 — Làm sạch dữ liệu")
    st.caption("Xử lý thiếu, trùng, logic giá và phát hiện ngoại lệ.")
    
    with st.expander("📜 Nhật ký chạy B2", expanded=False):
        st.code(preprocessing_logs['B2'])
    
    show_file_head("goldstock_cleaned_B2.csv")
    
    # Show outlier plot if exists
    outlier_png = "B2_outliers_detection.png"
    if os.path.exists(outlier_png):
        st.image(outlier_png, caption="Phát hiện ngoại lệ (IQR)", use_column_width=True)

with TAB_B3:
    st.header("B3 — Khám phá dữ liệu (EDA)")
    st.caption("Phân tích đơn biến, song biến, chuỗi thời gian và volume.")
    
    with st.expander("📜 Nhật ký chạy B3", expanded=False):
        st.code(preprocessing_logs['B3'])
    
    # Show generated figures
    for pattern in ["B3_univariate_*.png", "B3_bivariate_*.png", "B3_time_series_analysis.png", "B3_volume_analysis.png"]:
        for img in glob.glob(pattern):
            st.image(img, caption=os.path.basename(img), use_column_width=True)

with TAB_B4:
    st.header("B4 — Ma trận tương quan & PCA")
    st.caption("Xác định cột giữ/bỏ theo tương quan, trực quan hóa PCA.")
    
    with st.expander("📜 Nhật ký chạy B4", expanded=False):
        st.code(preprocessing_logs['B4'])
    
    show_file_head("goldstock_selected_features_B4.csv")
    show_file_head("goldstock_pca_B4.csv")
    
    # Show correlation & PCA plots
    for img in ["B4_correlation_matrix.png", "B4_pca_variance_explained.png", "B4_pca_projection.png"]:
        if os.path.exists(img):
            st.image(img, caption=os.path.basename(img), use_column_width=True)

with TAB_B5:
    st.header("B5 — Giao diện mô hình (Streamlit)")
    st.caption("Phân cụm K-Means và dự đoán Linear Regression.")
    # Render B5 module inside this tab
    B5.render_app()
