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
        st.image(outlier_png, caption="Phát hiện ngoại lệ (IQR)", use_container_width=True)

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
    st.header("B4 — Ma trận tương quan & PCA")
    st.caption("Xác định cột giữ/bỏ theo tương quan, trực quan hóa PCA.")
    
    with st.expander("📜 Nhật ký chạy B4", expanded=False):
        st.code(preprocessing_logs['B4'])
    
    show_file_head("goldstock_selected_features_B4.csv")
    show_file_head("goldstock_pca_B4.csv")
    
    # Show correlation & PCA plots
    for img in ["B4_correlation_matrix.png", "B4_pca_variance_explained.png", "B4_pca_projection.png"]:
        if os.path.exists(img):
            st.image(img, caption=os.path.basename(img), use_container_width=True)

with TAB_B5:
    st.header("B5 — Giao diện mô hình (Streamlit)")
    st.caption("Phân cụm K-Means và dự đoán Linear Regression.")
    # Render B5 module inside this tab
    B5.render_app()