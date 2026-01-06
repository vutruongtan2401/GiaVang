# ==========================================================
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('ggplot')
print("\n" + "=" * 60)

def run():
    # ==========================================================
    # LOAD DỮ LIỆU ĐÃ LÀM SẠCH
    # ==========================================================
    print("=" * 70)
    print("B3 - KHAI PHÁ DỮ LIỆU (Exploratory Data Analysis - EDA)")
    print("=" * 70)

    # Load từ file đã làm sạch (B2)
    try:
        df = pd.read_csv("goldstock_cleaned_B2.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"\n✅ Đã load dữ liệu từ B2: {len(df)} hàng")
    except:
        print("\n⚠️ Không tìm thấy file B2, load từ file gốc...")
        df = pd.read_csv("goldstock v2.csv", sep=";")
        # Xử lý tương tự B1
        if "Column1" in df.columns:
            df.drop(columns=["Column1"], inplace=True)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        df.columns = df.columns.str.strip()
        numeric_cols = ["Open", "High", "Low", "Close/Last", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')
        df = df.dropna()
        df = df[df.duplicated() == False].reset_index(drop=True)

    # Thêm cột phục vụ phân tích
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Price_Range'] = df['High'] - df['Low']  # Biến động giá hàng ngày
    
    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ==========================================================
    # B3.1 - PHÂN TÍCH ĐƠN BIẾN: BIỂU ĐỒ ĐƯỜNG (LINE CHART)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.1 - PHÂN TÍCH ĐƠN BIẾN: BIỂU ĐỒ ĐƯỜNG")
    print("=" * 70)
    print("\n📊 Biểu đồ: Close/Last - Xu hướng giá vàng theo thời gian")
    print(f"   Khoảng thời gian: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"   Giá Close trung bình: ${df['Close/Last'].mean():.2f}")
    print(f"   Giá Close cao nhất: ${df['Close/Last'].max():.2f}")
    print(f"   Giá Close thấp nhất: ${df['Close/Last'].min():.2f}")
    print(f"   Độ biến động (Std): ${df['Close/Last'].std():.2f}")

    # Vẽ biểu đồ đường
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df['Date'], df['Close/Last'], linewidth=2.5, color='steelblue', label='Close Price')
    ax.fill_between(df['Date'], df['Low'], df['High'], alpha=0.2, color='lightblue', label='High-Low Range')
    ax.axhline(y=df['Close/Last'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["Close/Last"].mean():.2f}')
    
    ax.set_xlabel('Ngày (Date)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá (USD/oz)', fontsize=12, fontweight='bold')
    ax.set_title('Xu hướng giá vàng theo thời gian (Gold Price Trend)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("B3_line_chart_close_price.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_line_chart_close_price.png")
    
    # NHẬN XÉT
    print("\n   📝 NHẬN XÉT:")
    price_trend = "tăng" if df['Close/Last'].iloc[-1] > df['Close/Last'].iloc[0] else "giảm"
    pct_change = ((df['Close/Last'].iloc[-1] - df['Close/Last'].iloc[0]) / df['Close/Last'].iloc[0] * 100)
    print(f"   • Xu hướng tổng thể: Giá vàng có xu hướng {price_trend} ({pct_change:+.2f}%) trong giai đoạn quan sát")
    print(f"   • Biên độ dao động: Vùng High-Low cho thấy độ biến động hàng ngày của thị trường")
    print(f"   • Đường trung bình (đỏ đứt nét) là mốc tham chiếu để đánh giá giá hiện tại cao/thấp")
    
    # Phân tích xu hướng theo giai đoạn
    mid_point = len(df) // 2
    first_half_mean = df['Close/Last'].iloc[:mid_point].mean()
    second_half_mean = df['Close/Last'].iloc[mid_point:].mean()
    if second_half_mean > first_half_mean:
        print(f"   • Nửa sau giai đoạn có giá trung bình cao hơn nửa đầu (${second_half_mean:.2f} vs ${first_half_mean:.2f})")
    else:
        print(f"   • Nửa đầu giai đoạn có giá trung bình cao hơn nửa sau (${first_half_mean:.2f} vs ${second_half_mean:.2f})")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.2 - PHÂN TÍCH ĐƠN BIẾN: SO SÁNH KHỐI LƯỢNG GIAO DỤC VÀ GIÁ THEO NĂM
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.2 - SO SÁNH KHỐI LƯỢNG GIAO DỤC VÀ GIÁ ĐÓNG CỬA (2014-2024)")
    print("=" * 70)
    
    # Tính toán theo năm
    yearly_stats = df.groupby('Year').agg({
        'Volume': 'mean',
        'Close/Last': 'mean'
    }).reset_index()
    
    print(f"\n📊 Biểu đồ: So sánh Khối lượng giao dịch và Giá đóng cửa theo năm")
    print(f"   Năm từ {yearly_stats['Year'].min()} đến {yearly_stats['Year'].max()}")
    
    # Tạo figure với 2 trục Y
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Trục 1: Khối lượng (cột xanh)
    color1 = 'steelblue'
    ax1.set_xlabel('Năm (Year)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Khối lượng giao dịch (Triệu)', fontsize=12, fontweight='bold', color=color1)
    bars = ax1.bar(yearly_stats['Year'], yearly_stats['Volume']/1e6, color=color1, alpha=0.7, label='Khối lượng', width=0.6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Trục 2: Giá đóng cửa (đường cam)
    ax2 = ax1.twinx()
    color2 = 'orange'
    ax2.set_ylabel('Giá đóng cửa (USD)', fontsize=12, fontweight='bold', color=color2)
    line = ax2.plot(yearly_stats['Year'], yearly_stats['Close/Last'], color=color2, marker='o', 
                    linewidth=3, markersize=8, label='Giá', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Tiêu đề
    plt.title('So sánh Khối lượng giao dịch và Giá đóng cửa (2014-2024)', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
    
    fig.tight_layout()
    plt.savefig("B3_histogram_volume.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_histogram_volume.png")
    
    # NHẬN XÉT
    print("\n   📝 NHẬN XÉT:")
    max_volume_year = yearly_stats.loc[yearly_stats['Volume'].idxmax(), 'Year']
    max_volume = yearly_stats['Volume'].max()
    max_price_year = yearly_stats.loc[yearly_stats['Close/Last'].idxmax(), 'Year']
    max_price = yearly_stats['Close/Last'].max()
    
    print("   **Xu hướng tổng thể:** Thanh khoản có sự biến động mạnh, chia làm hai giai đoạn rõ rệt: tăng trưởng mạnh từ 2014 đến 2018, sau đó có xu hướng sụt giảm dần và phục hồi nhẹ vào năm 2024.")
    print()
    print("   **Giai đoạn bùng nổ (2014 - 2019):** Thanh khoản tăng trưởng liên tục và duy trì ở mức rất cao, đạt đỉnh vào khoảng năm 2018 - 2019 với khối lượng giao dịch tiệm cận mức 2.5 triệu.")
    print()
    print("   **Giai đoạn sụt giảm (2020 - 2022):** Khối lượng giao dịch giảm dần qua từng năm, cho thấy sự thu hẹp về thanh khoản, chạm mức thấp nhất trong chu kỳ gần đây vào năm 2022 (khoảng 1.6 triệu).")
    print()
    print("   **Dấu hiệu phục hồi (2023 - 2024):** Thanh khoản bắt đầu có sự cải thiện trở lại trong hai năm gần nhất, đặc biệt là năm 2024 khi khối lượng vượt mức 2.0 triệu, cho thấy thị trường đang sôi động trở lại.")
    
    # NHẬN XÉT
    print("\n   📝 NHẬN XÉT:")
    skewness = df['Volume'].skew()
    if skewness > 1:
        print(f"   • Phân phối lệch phải mạnh (skewness={skewness:.2f}): Hầu hết ngày có khối lượng thấp")
    elif skewness > 0.5:
        print(f"   • Phân phối lệch phải vừa phải (skewness={skewness:.2f}): Xu hướng khối lượng thấp")
    else:
        print(f"   • Phân phối gần đối xứng (skewness={skewness:.2f})")
    
    mean_median_diff = abs(df['Volume'].mean() - df['Volume'].median())
    print(f"   • Khoảng cách Mean-Median: {mean_median_diff:,.0f} → {'Có ngoại lệ giá trị cao' if mean_median_diff > df['Volume'].std() else 'Phân phối tương đối đồng đều'}")
    print(f"   • Hầu hết các phiên giao dịch có khối lượng quanh mức {df['Volume'].median():,.0f}")
    
    # Tính % ngày có volume trên/dưới trung bình
    above_mean_pct = (df['Volume'] > df['Volume'].mean()).sum() / len(df) * 100
    print(f"   • {above_mean_pct:.1f}% ngày có khối lượng > trung bình, {100-above_mean_pct:.1f}% < trung bình")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.2.1 - PHÂN TÍCH ĐƠN BIẾN: HISTOGRAM TẤT CẢ CÁC BIẾN GIÁ
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.2.1 - PHÂN TÍCH ĐƠN BIẾN: HISTOGRAM - PHÂN PHỐI TẤT CẢ CÁC BIẾN")
    print("=" * 70)
    print("\n📊 Biểu đồ tổng hợp: Phân phối của Close, Volume, Open, High, Low")
    
    # Danh sách các biến cần vẽ
    variables = ['Close/Last', 'Volume', 'Open', 'High', 'Low']
    titles = ['Histogram of Close', 'Histogram of Volume', 'Histogram of Open', 
              'Histogram of High', 'Histogram of Low']
    
    # Tạo figure với 2 hàng, 3 cột
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (var, title) in enumerate(zip(variables, titles)):
        ax = axes[idx]
        
        # Vẽ histogram
        n, bins, patches = ax.hist(df[var], bins=30, alpha=0.7, color='skyblue', 
                                    edgecolor='black', linewidth=1.2, density=True)
        
        # Vẽ đường KDE (Kernel Density Estimation)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(df[var].dropna())
        x_range = np.linspace(df[var].min(), df[var].max(), 200)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2.5, label='KDE')
        
        # Thiết lập tiêu đề và nhãn
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(var.replace('Close/Last', 'Close'), fontsize=11)
        ax.set_ylabel('Count' if idx < 3 else 'Count', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # In thống kê
        print(f"\n   {var}:")
        print(f"      Mean: {df[var].mean():.2f}")
        print(f"      Median: {df[var].median():.2f}")
        print(f"      Std: {df[var].std():.2f}")
        print(f"      Skewness: {df[var].skew():.3f}")
    
    # Ẩn subplot thừa (subplot thứ 6)
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("B3_univariate_histograms_all.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_univariate_histograms_all.png")
    
    # NHẬN XÉT TỔNG HỢP
    print("\n   📝 NHẬN XÉT TỔNG HỢP:")
    print("\n   1. PHÂN PHỐI CÁC BIẾN GIÁ (Close, Open, High, Low):")
    print(f"      • Tất cả đều có phân phối bimodal (2 đỉnh) rõ rệt")
    print(f"      • Đỉnh thấp (~1200-1300): Giai đoạn giá vàng ổn định")
    print(f"      • Đỉnh cao (~1800-2000): Giai đoạn giá vàng tăng mạnh")
    print(f"      • Cho thấy thị trường vàng có 2 giai đoạn giá rõ rệt")
    
    print("\n   2. SO SÁNH CLOSE, OPEN, HIGH, LOW:")
    print(f"      • Các biến này có phân phối tương tự nhau (correlation cao)")
    close_open_corr = df['Close/Last'].corr(df['Open'])
    high_low_corr = df['High'].corr(df['Low'])
    print(f"      • Close-Open correlation: {close_open_corr:.4f}")
    print(f"      • High-Low correlation: {high_low_corr:.4f}")
    print(f"      • Đều phản ánh cùng một xu hướng giá vàng")
    
    print("\n   3. VOLUME:")
    print(f"      • Phân phối khác biệt hoàn toàn so với giá")
    print(f"      • Lệch phải mạnh: Phần lớn ngày có volume thấp, ít ngày volume rất cao")
    print(f"      • Volume không tương quan mạnh với mức giá")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.3 - PHÂN TÍCH ĐA BIẾN: SCATTER PLOT (VOLUME vs CLOSE/LAST)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.3 - PHÂN TÍCH ĐA BIẾN: SCATTER PLOT")
    print("=" * 70)
    print("\n📊 Biểu đồ: Volume vs Close/Last - Mối quan hệ khối lượng và giá")
    
    # Tính tương quan
    correlation = df['Volume'].corr(df['Close/Last'])
    print(f"   Hệ số tương quan Pearson: {correlation:.4f}")
    
    if abs(correlation) > 0.8:
        print(f"   → Tương quan RẤT MẠNH")
    elif abs(correlation) > 0.6:
        print(f"   → Tương quan MẠNH")
    elif abs(correlation) > 0.4:
        print(f"   → Tương quan VỪA PHẢI")
    elif abs(correlation) > 0.2:
        print(f"   → Tương quan YẾU")
    else:
        print(f"   → Tương quan RẤT YẾU hoặc KHÔNG CÓ")

    # Define phases for coloring
    phases = []
    colors_map = {
        '2014-2017': '#1f77b4',      # Blue
        '2018-2019': '#ff7f0e',      # Orange
        '2020-2022': '#d62728',      # Red
        '2023-2024': '#2ca02c'       # Green
    }
    
    for year in df['Year']:
        if 2014 <= year <= 2017:
            phases.append('2014-2017')
        elif 2018 <= year <= 2019:
            phases.append('2018-2019')
        elif 2020 <= year <= 2022:
            phases.append('2020-2022')
        else:
            phases.append('2023-2024')
    
    df['Phase'] = phases
    phase_colors = [colors_map[phase] for phase in df['Phase']]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each phase with different color
    for phase, color in colors_map.items():
        mask = df['Phase'] == phase
        ax.scatter(df[mask]['Volume'], df[mask]['Close/Last'], 
                  c=color, alpha=0.6, s=100, edgecolors='black', linewidth=0.8, label=phase)
    
    # Thêm đường xu hướng
    z = np.polyfit(df['Volume'], df['Close/Last'], 1)
    p = np.poly1d(z)
    volume_sorted = df['Volume'].sort_values()
    ax.plot(volume_sorted, p(volume_sorted), "r--", linewidth=2.5, label=f'Trend line (r={correlation:.3f})')
    
    ax.set_xlabel('Khối lượng giao dịch (Volume)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá đóng cửa (Close Price - USD/oz)', fontsize=12, fontweight='bold')
    ax.set_title('Mối quan hệ giữa Khối lượng giao dịch và Giá vàng - Phân chia theo giai đoạn', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', title='Giai đoạn', title_fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("B3_scatter_volume_vs_price.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_scatter_volume_vs_price.png")
    
    # NHẬN XÉT
    print("\n   📝 NHẬN XÉT:")
    print(f"   **1. Tương quan rất yếu (r={correlation:.3f}): Volume và Giá gần như KHÔNG liên quan**")
    print(f"   • Đường trend line gần như ngang (hệ số góc ≈ 0)")
    print(f"   • → Khối lượng giao dịch KHÔNG phải là yếu tố dự báo tốt cho giá vàng")
    print(f"   • → Giá vàng được xác định bởi các yếu tố khác (tỷ giá, lạm phát, địa chính trị...)")
    
    print(f"\n   **2. Phân tích theo giai đoạn (thấy rõ xu hướng giá tăng theo thời gian):**")
    print(f"   • Xanh dương (2014-2017): Cluster ở vùng thấp nhất")
    print(f"      - Volume: 0.5-2.5 triệu")
    print(f"      - Giá: $1000-1400 (thấp nhất)")
    print(f"      - Đặc điểm: Tập trung chặt chẽ, ổn định")
    
    print(f"   • Cam (2018-2019): Bắt đầu tăng")
    print(f"      - Volume: 1-3 triệu (tăng đôi chút)")
    print(f"      - Giá: $1300-1500 (tăng nhẹ)")
    print(f"      - Đặc điểm: Cluster bắt đầu rộng hơn, giá có dấu hiệu tăng")
    
    print(f"   • Đỏ (2020-2022): Giá tăng mạnh, volume phân tán")
    print(f"      - Volume: 0.5-4 triệu (phân tán rất lớn)")
    print(f"      - Giá: $1600-2000 (tăng vọt)")
    print(f"      - Đặc điểm: Cluster lớn nhất, giá bắt đầu đạt mức cao, nhưng volume không ảnh hưởng")
    
    print(f"   • Xanh lá (2023-2024): Giá cao nhất, volume phục hồi")
    print(f"      - Volume: 1-5 triệu (phục hồi mạnh)")
    print(f"      - Giá: $2000-2400 (cao nhất lịch sử)")
    print(f"      - Đặc điểm: Nằm ở vùng trên cùng, giá lập đỉnh, volume phục hồi")
    
    print(f"\n   **3. Kết luận:**")
    print(f"   • Giá vàng có xu hướng tăng theo thời gian (từ 2014 → 2024)")
    print(f"   • Nhưng volume không ảnh hưởng trực tiếp đến giá")
    print(f"   • Cùng volume có thể tương ứng với giá rất khác nhau (từ $1000 → $2400)")
    
    # Phân tích các điểm outlier
    high_volume_threshold = df['Volume'].quantile(0.95)
    high_volume_days = df[df['Volume'] > high_volume_threshold]
    if len(high_volume_days) > 0:
        avg_price_high_vol = high_volume_days['Close/Last'].mean()
        avg_price_overall = df['Close/Last'].mean()
        print(f"\n   • Phân tích 5% ngày có volume cao nhất:")
        print(f"      - Giá trung bình: ${avg_price_high_vol:.2f} vs ${avg_price_overall:.2f} (tổng thể)")
        if avg_price_high_vol > avg_price_overall * 1.05:
            print(f"      - Ngày volume cao thường đi kèm giá cao hơn")
        elif avg_price_high_vol < avg_price_overall * 0.95:
            print(f"      - Ngày volume cao thường đi kèm giá thấp hơn")
        else:
            print(f"      - Volume cao xuất hiện ở cả vùng giá cao và thấp")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.4 - PHÂN TÍCH ĐA BIẾN: BOXPLOT (BIẾN ĐỘNG GIÁ QUA CÁC NĂM)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.4 - PHÂN TÍCH ĐA BIẾN: BOXPLOT - BIẾN ĐỘNG GIÁ QUA CÁC NĂM")
    print("=" * 70)
    print("\n📊 Biểu đồ: Boxplot so sánh mức độ biến động giá qua các năm")
    print(f"   Biến động giá = High - Low (khoảng dao động hàng ngày)")
    
    # Thống kê biến động theo năm
    print(f"\n   Thống kê biến động theo năm:")
    yearly_stats = df.groupby('Year')['Price_Range'].describe()
    print(yearly_stats)

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Chuẩn bị dữ liệu cho boxplot
    years = sorted(df['Year'].unique())
    data_by_year = [df[df['Year'] == year]['Price_Range'].values for year in years]
    
    # Vẽ boxplot
    bp = ax.boxplot(data_by_year, labels=years, patch_artist=True,
                    widths=0.6,
                    boxprops=dict(facecolor='lightblue', color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
    
    ax.set_xlabel('Năm (Year)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biến động giá hàng ngày (USD/oz)', fontsize=12, fontweight='bold')
    ax.set_title('So sánh mức độ biến động giá vàng qua các năm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Thêm ghi chú
    ax.text(0.02, 0.98, 'Biến động giá = High - Low (khoảng dao động hàng ngày)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("B3_boxplot_price_volatility_by_year.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_boxplot_price_volatility_by_year.png")
    
    # NHẬN XÉT CHI TIẾT
    print("\n   📝 NHẬN XÉT CHI TIẾT:")
    
    # Tìm năm biến động cao/thấp nhất
    yearly_volatility = df.groupby('Year')['Price_Range'].mean().sort_values()
    most_stable_year = yearly_volatility.index[0]
    most_volatile_year = yearly_volatility.index[-1]
    
    print(f"\n   1. XU HƯỚNG BIẾN ĐỘNG THEO THỜI GIAN:")
    print(f"      • Năm ổn định nhất: {most_stable_year} (biến động trung bình: ${yearly_volatility.iloc[0]:.2f}/ngày)")
    print(f"      • Năm biến động nhất: {most_volatile_year} (biến động trung bình: ${yearly_volatility.iloc[-1]:.2f}/ngày)")
    print(f"      • Tỷ lệ biến động: {(yearly_volatility.iloc[-1]/yearly_volatility.iloc[0]):.2f}x")
    
    print(f"\n   2. PHÂN TÍCH BOXPLOT:")
    print(f"      • Hộp (Box): Chứa 50% dữ liệu giữa (Q1-Q3), thể hiện biến động thông thường")
    print(f"      • Đường đỏ (Median): Biến động trung vị mỗi năm")
    print(f"      • Râu (Whiskers): Phạm vi biến động bình thường (không phải outlier)")
    print(f"      • Chấm đỏ (Outliers): Những ngày biến động bất thường (cao/thấp đột biến)")
    
    # Đếm outliers theo năm
    print(f"\n   3. PHÂN TÍCH OUTLIERS (BIẾN ĐỘNG BẤT THƯỜNG):")
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]['Price_Range']
        Q1 = year_data.quantile(0.25)
        Q3 = year_data.quantile(0.75)
        IQR = Q3 - Q1
        outliers = year_data[(year_data < Q1 - 1.5*IQR) | (year_data > Q3 + 1.5*IQR)]
        outlier_pct = len(outliers) / len(year_data) * 100
        print(f"      • Năm {year}: {len(outliers)} ngày bất thường ({outlier_pct:.1f}%)")
    
    print(f"\n   4. Ý NGHĨA THỰC TIỄN:")
    print(f"      • Năm có biến động cao → Thị trường bất ổn, rủi ro đầu tư cao")
    print(f"      • Năm có biến động thấp → Thị trường ổn định, dễ dự đoán")
    print(f"      • Nhiều outliers → Có sự kiện bất thường tác động đến thị trường")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.5 - TỔNG HỢP & KẾT LUẬN
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.5 - TỔNG HỢP & KẾT LUẬN")
    print("=" * 70)

    print("\n✅ NHỮNG PHÁT HIỆN CHÍNH TỪ EDA:")
    print("\n1️⃣  PHÂN TÍCH ĐƠN BIẾN:")
    print(f"   • Giá Close/Last: Trung bình ${df['Close/Last'].mean():.2f}, dao động từ ${df['Close/Last'].min():.2f} - ${df['Close/Last'].max():.2f}")
    print(f"   • Volume: Trung bình {df['Volume'].mean():,.0f}, phân phối {('lệch phải' if df['Volume'].skew() > 0 else 'lệch trái')}")
    
    print(f"\n2️⃣  PHÂN TÍCH ĐA BIẾN:")
    correlation = df['Volume'].corr(df['Close/Last'])
    print(f"   • Mối quan hệ Volume vs Close: r = {correlation:.4f} ({('yếu' if abs(correlation) < 0.3 else 'trung bình' if abs(correlation) < 0.6 else 'mạnh')})")
    
    print(f"\n3️⃣  PHÂN TÍCH BIẾN ĐỘNG GIÁ:")
    print(f"   • Biến động giá trung bình: ${df['Price_Range'].mean():.2f}/oz")
    print(f"   • Năm {df.groupby('Year')['Price_Range'].mean().idxmax()}: Biến động cao nhất (${df.groupby('Year')['Price_Range'].mean().max():.2f})")
    print(f"   • Năm {df.groupby('Year')['Price_Range'].mean().idxmin()}: Biến động thấp nhất (${df.groupby('Year')['Price_Range'].mean().min():.2f})")

    print("\n" + "=" * 70)
    print("KẾT THÚC B3 - KHAI PHÁ DỮ LIỆU (EDA)")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run()