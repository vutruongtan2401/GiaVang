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
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.2 - PHÂN TÍCH ĐƠN BIẾN: HISTOGRAM (KHỐI LƯỢNG GIAO DỊCH)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B3.2 - PHÂN TÍCH ĐƠN BIẾN: HISTOGRAM - PHÂN PHỐI VOLUME")
    print("=" * 70)
    print("\n📊 Biểu đồ: Volume - Phân phối khối lượng giao dịch")
    print(f"   Khối lượng trung bình: {df['Volume'].mean():,.0f}")
    print(f"   Khối lượng cao nhất: {df['Volume'].max():,.0f}")
    print(f"   Khối lượng thấp nhất: {df['Volume'].min():,.0f}")
    print(f"   Độ lệch chuẩn: {df['Volume'].std():,.0f}")
    print(f"   Skewness (Độ lệch): {df['Volume'].skew():.3f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(df['Volume'], bins=40, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.axvline(df['Volume'].mean(), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {df["Volume"].mean():,.0f}')
    ax.axvline(df['Volume'].median(), color='green', linestyle='--', linewidth=2.5, label=f'Median: {df["Volume"].median():,.0f}')
    
    ax.set_xlabel('Khối lượng giao dịch (Volume)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tần suất (Frequency)', fontsize=12, fontweight='bold')
    ax.set_title('Phân phối khối lượng giao dịch (Volume Distribution)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("B3_histogram_volume.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_histogram_volume.png")
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

    fig, ax = plt.subplots(figsize=(14, 7))
    scatter = ax.scatter(df['Volume'], df['Close/Last'], 
                        c=range(len(df)), cmap='viridis', 
                        alpha=0.6, s=80, edgecolors='black', linewidth=0.8)
    
    # Thêm đường xu hướng
    z = np.polyfit(df['Volume'], df['Close/Last'], 1)
    p = np.poly1d(z)
    volume_sorted = df['Volume'].sort_values()
    ax.plot(volume_sorted, p(volume_sorted), "r--", linewidth=2.5, label=f'Trend line (r={correlation:.3f})')
    
    ax.set_xlabel('Khối lượng giao dịch (Volume)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá đóng cửa (Close Price - USD/oz)', fontsize=12, fontweight='bold')
    ax.set_title('Mối quan hệ giữa Khối lượng giao dịch và Giá vàng', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax, label='Thứ tự thời gian')
    
    plt.tight_layout()
    plt.savefig("B3_scatter_volume_vs_price.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B3_scatter_volume_vs_price.png")
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
