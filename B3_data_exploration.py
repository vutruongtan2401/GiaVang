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
    print("=" * 60)
    print("B3 - KHAI PHÁ DỮ LIỆU (EDA)")
    print("=" * 60)

    # Load từ file đã làm sạch (B2)
    try:
        df = pd.read_csv("goldstock_cleaned_B2.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"\n✅ Đã load dữ liệu từ B2: {len(df)} hàng")
    except:
        print("\n⚠️ Không tìm thấy file B2, load từ file gốc...")
        df = pd.read_csv("goldstock v2.csv", sep=";")
        # Xử lý tương tự B2
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

    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ==========================================================
    # B3.1 - PHÂN TÍCH ĐƠN BIẾN (UNIVARIATE ANALYSIS)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B3.1 - PHÂN TÍCH ĐƠN BIẾN (UNIVARIATE ANALYSIS)")
    print("=" * 60)

    for col in quantitative_cols:
        print(f"\n{'='*60}")
        print(f"Phân tích: {col}")
        print(f"{'='*60}")
        
        # Thống kê mô tả
        print(f"\n📊 THỐNG KÊ MÔ TẢ:")
        stats_dict = {
            "Count": df[col].count(),
            "Mean": df[col].mean(),
            "Std Dev": df[col].std(),
            "Min": df[col].min(),
            "25%": df[col].quantile(0.25),
            "Median": df[col].median(),
            "75%": df[col].quantile(0.75),
            "Max": df[col].max(),
            "Range": df[col].max() - df[col].min(),
            "Skewness": df[col].skew(),
            "Kurtosis": df[col].kurtosis()
        }
        
        for key, value in stats_dict.items():
            print(f"   {key:<15}: {value:>15,.2f}")
        
        # Kiểm tra phân phối
        print(f"\n📈 KIỂM TRA PHÂN PHỐI:")
        if abs(df[col].skew()) < 0.5:
            print(f"   ✓ Phân phối gần đối xứng (Skewness = {df[col].skew():.2f})")
        elif df[col].skew() > 0:
            print(f"   → Phân phối lệch phải (Skewness = {df[col].skew():.2f})")
        else:
            print(f"   ← Phân phối lệch trái (Skewness = {df[col].skew():.2f})")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram with KDE
        axes[0].hist(df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {df[col].mean():.2f}')
        axes[0].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median = {df[col].median():.2f}')
        axes[0].set_title(f'Histogram: {col}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # KDE Plot
        df[col].plot(kind='kde', ax=axes[1], color='steelblue', linewidth=2)
        axes[1].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[1].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label='Median')
        axes[1].set_title(f'Density Plot: {col}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Box Plot
        axes[2].boxplot(df[col], vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        medianprops=dict(color='red', linewidth=2))
        axes[2].set_title(f'Box Plot: {col}', fontsize=12, fontweight='bold')
        axes[2].set_ylabel(col)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        axes[2].text(0.5, 0.02, f'Outliers: {outliers}', 
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    horizontalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f"B3_univariate_{col.replace('/', '_')}.png", dpi=300, bbox_inches='tight')
        print(f"   ✓ Biểu đồ đã lưu: B3_univariate_{col.replace('/', '_')}.png")
        try:
            plt.close(fig)
        except Exception:
            pass

    # ==========================================================
    # B3.2 - PHÂN TÍCH ĐA BIẾN (BIVARIATE ANALYSIS)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B3.2 - PHÂN TÍCH ĐA BIẾN (BIVARIATE ANALYSIS)")
    print("=" * 60)

    # Phân tích quan hệ giữa các cặp biến
    pairs = [
        ("Open", "Close/Last"),
        ("Low", "High"),
        ("Volume", "Close/Last"),
        ("Open", "Volume")
    ]

    for x_col, y_col in pairs:
        print(f"\n{'='*60}")
        print(f"Phân tích quan hệ: {x_col} vs {y_col}")
        print(f"{'='*60}")
        
        # Tính tương quan
        correlation = df[x_col].corr(df[y_col])
        print(f"\n📊 Hệ số tương quan Pearson: {correlation:.4f}")
        
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
        
        if correlation > 0:
            print(f"   ↗ Quan hệ THUẬN (khi {x_col} tăng, {y_col} cũng tăng)")
        else:
            print(f"   ↘ Quan hệ NGHỊCH (khi {x_col} tăng, {y_col} giảm)")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot with regression line
        axes[0].scatter(df[x_col], df[y_col], alpha=0.5, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        axes[0].plot(df[x_col].sort_values(), p(df[x_col].sort_values()), 
                    "r--", linewidth=2, label=f'Trend line (r={correlation:.3f})')
        
        axes[0].set_xlabel(x_col, fontsize=11, fontweight='bold')
        axes[0].set_ylabel(y_col, fontsize=11, fontweight='bold')
        axes[0].set_title(f'Scatter Plot: {x_col} vs {y_col}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Hexbin plot (for density)
        hexbin = axes[1].hexbin(df[x_col], df[y_col], gridsize=25, cmap='Blues', mincnt=1)
        axes[1].set_xlabel(x_col, fontsize=11, fontweight='bold')
        axes[1].set_ylabel(y_col, fontsize=11, fontweight='bold')
        axes[1].set_title(f'Density Plot: {x_col} vs {y_col}', fontsize=12, fontweight='bold')
        plt.colorbar(hexbin, ax=axes[1], label='Count')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"B3_bivariate_{x_col.replace('/', '_')}_vs_{y_col.replace('/', '_')}.png", 
                    dpi=300, bbox_inches='tight')
        print(f"   ✓ Biểu đồ đã lưu: B3_bivariate_{x_col.replace('/', '_')}_vs_{y_col.replace('/', '_')}.png")
        try:
            plt.close(fig)
        except Exception:
            pass

    # ==========================================================
    # B3.3 - PHÂN TÍCH CHUỖI THỜI GIAN (TIME SERIES ANALYSIS)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B3.3 - PHÂN TÍCH CHUỖI THỜI GIAN")
    print("=" * 60)

    print("\n📊 THỐNG KÊ CHUỖI THỜI GIAN:")
    print(f"   Khoảng thời gian: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"   Số ngày giao dịch: {len(df)}")
    print(f"   Giá Close trung bình: ${df['Close/Last'].mean():.2f}")
    print(f"   Giá Close cao nhất: ${df['Close/Last'].max():.2f} (ngày {df.loc[df['Close/Last'].idxmax(), 'Date'].date()})")
    print(f"   Giá Close thấp nhất: ${df['Close/Last'].min():.2f} (ngày {df.loc[df['Close/Last'].idxmin(), 'Date'].date()})")
    print(f"   Biên độ giá: ${df['Close/Last'].max() - df['Close/Last'].min():.2f}")
    print(f"   Độ biến động (Std): ${df['Close/Last'].std():.2f}")

    # Visualize time series
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Price trend
    axes[0].plot(df['Date'], df['Close/Last'], linewidth=2, color='steelblue', label='Close Price')
    axes[0].fill_between(df['Date'], df['Low'], df['High'], alpha=0.2, color='lightblue', label='High-Low Range')
    axes[0].axhline(y=df['Close/Last'].mean(), color='red', linestyle='--', linewidth=2, label='Mean Price')
    axes[0].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Price (USD/oz)', fontsize=11, fontweight='bold')
    axes[0].set_title('Xu hướng giá vàng (Gold Price Trend)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Volume over time
    axes[1].bar(df['Date'], df['Volume'], color='steelblue', alpha=0.7, width=1)
    axes[1].axhline(y=df['Volume'].mean(), color='red', linestyle='--', linewidth=2, label='Mean Volume')
    axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=11, fontweight='bold')
    axes[1].set_title('Khối lượng giao dịch (Trading Volume)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Daily returns (% change)
    df['Daily_Return'] = df['Close/Last'].pct_change() * 100
    axes[2].plot(df['Date'], df['Daily_Return'], linewidth=1.5, color='steelblue', alpha=0.7)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[2].fill_between(df['Date'], 0, df['Daily_Return'], 
                         where=(df['Daily_Return'] > 0), color='green', alpha=0.3, label='Positive Return')
    axes[2].fill_between(df['Date'], 0, df['Daily_Return'], 
                         where=(df['Daily_Return'] < 0), color='red', alpha=0.3, label='Negative Return')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Daily Return (%)', fontsize=11, fontweight='bold')
    axes[2].set_title('Biến động hàng ngày (Daily Returns)', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("B3_time_series_analysis.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ chuỗi thời gian đã lưu: B3_time_series_analysis.png")
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.4 - PHÂN TÍCH KHỐI LƯỢNG (VOLUME ANALYSIS)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B3.4 - PHÂN TÍCH KHỐI LƯỢNG GIAO DỊCH")
    print("=" * 60)

    print("\n📊 THỐNG KÊ KHỐI LƯỢNG:")
    print(f"   Khối lượng trung bình: {df['Volume'].mean():,.0f}")
    print(f"   Khối lượng cao nhất: {df['Volume'].max():,.0f} (ngày {df.loc[df['Volume'].idxmax(), 'Date'].date()})")
    print(f"   Khối lượng thấp nhất: {df['Volume'].min():,.0f} (ngày {df.loc[df['Volume'].idxmin(), 'Date'].date()})")
    print(f"   Tổng khối lượng: {df['Volume'].sum():,.0f}")
    print(f"   Độ lệch chuẩn: {df['Volume'].std():,.0f}")

    # Volume distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    axes[0].hist(df['Volume'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(df['Volume'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean = {df["Volume"].mean():,.0f}')
    axes[0].axvline(df['Volume'].median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median = {df["Volume"].median():,.0f}')
    axes[0].set_xlabel('Volume', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Phân phối khối lượng (Volume Distribution)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Volume vs Price
    scatter = axes[1].scatter(df['Volume'], df['Close/Last'], 
                             c=range(len(df)), cmap='viridis', 
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[1].set_xlabel('Volume', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Close Price (USD/oz)', fontsize=11, fontweight='bold')
    axes[1].set_title('Quan hệ Volume vs Price', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Time Index')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("B3_volume_analysis.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ phân tích volume đã lưu: B3_volume_analysis.png")
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B3.5 - TỔNG HỢP & KẾT LUẬN
    # ==========================================================
    print("\n" + "=" * 60)
    print("B3.5 - TỔNG HỢP & KẾT LUẬN")
    print("=" * 60)

    print("\n✅ NHỮNG PHÁT HIỆN CHÍNH:")
    print("   1. Phân tích đơn biến cho thấy phân phối của các biến giá")
    print("   2. Phân tích đa biến phát hiện tương quan cao giữa Open, High, Low, Close")
    print("   3. Chuỗi thời gian cho thấy xu hướng và biến động giá vàng")
    print("   4. Volume có mối quan hệ với biến động giá")

    print("\n" + "=" * 60)
    print("KẾT THÚC B3 - KHAI PHÁ DỮ LIỆU")
    print("=" * 60)

if __name__ == "__main__":
    run()
print("KẾT THÚC B3 - KHAI PHÁ DỮ LIỆU")
print("=" * 60)
