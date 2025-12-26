# ==========================================================
# B2 – TIỀN XỬ LÝ DỮ LIỆU (DATA CLEANING)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

def run():
    # ==========================================================
    # LOAD DỮ LIỆU GỐC
    # ==========================================================
    print("=" * 60)
    print("B2 - TIỀN XỬ LÝ DỮ LIỆU (DATA CLEANING)")
    print("=" * 60)

    # Load dữ liệu gốc
    original_df = pd.read_csv("goldstock v2.csv", sep=";")

    # Xóa cột index không cần thiết
    if "Column1" in original_df.columns:
        original_df.drop(columns=["Column1"], inplace=True)
    if "Unnamed: 0" in original_df.columns:
        original_df.drop(columns=["Unnamed: 0"], inplace=True)

    # Xử lý khoảng trắng
    original_df.columns = original_df.columns.str.strip()

    # Chuyển đổi kiểu dữ liệu
    numeric_cols = ["Volume", "Open", "High", "Low", "Close/Last"]
    for col in numeric_cols:
        if col in original_df.columns:
            original_df[col] = pd.to_numeric(original_df[col], errors='coerce')

    # Chuyển Date sang datetime
    try:
        original_df["Date"] = pd.to_datetime(original_df["Date"], format="%d/%m/%Y", errors='coerce')
    except:
        original_df["Date"] = pd.to_datetime(original_df["Date"], infer_datetime_format=True, errors='coerce')

    original_df.sort_values(by="Date", inplace=True, ascending=True)
    original_df.reset_index(drop=True, inplace=True)

    print(f"\n✅ Dữ liệu gốc đã load: {len(original_df)} hàng, {original_df.shape[1]} cột")

    # ==========================================================
    # B2.1 - KIỂM TRA DỮ LIỆU THIẾU (MISSING DATA)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.1 - KIỂM TRA DỮ LIỆU THIẾU")
    print("=" * 60)

    missing_count = original_df.isnull().sum()
    missing_percent = (missing_count / len(original_df)) * 100

    print("\n📊 MISSING DATA SUMMARY:")
    print("-" * 60)
    print(f"{'Column':<20} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 60)

    for col in original_df.columns:
        if missing_count[col] > 0:
            print(f"{col:<20} {missing_count[col]:<15} {missing_percent[col]:<15.2f}%")

    if missing_count.sum() == 0:
        print("✅ Không có dữ liệu thiếu!")
    else:
        print(f"\n⚠️ Tổng cộng: {missing_count.sum()} giá trị thiếu")

    # ==========================================================
    # B2.2 - KIỂM TRA DỮ LIỆU TRÙNG LẶP (DUPLICATE DATA)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.2 - KIỂM TRA DỮ LIỆU TRÙNG LẶP")
    print("=" * 60)

    duplicate_count = original_df.duplicated().sum()
    print(f"\n📊 Số dòng trùng lặp: {duplicate_count}")

    if duplicate_count > 0:
        print(f"⚠️ Phát hiện {duplicate_count} dòng trùng lặp - sẽ được loại bỏ")
    else:
        print("✅ Không có dòng trùng lặp!")

    # ==========================================================
    # B2.3 - XỬ LÝ DỮ LIỆU (CLEANING PROCESS)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.3 - QUÁ TRÌNH LÀM SẠCH DỮ LIỆU")
    print("=" * 60)

    # Clone dữ liệu để giữ nguyên bản gốc
    df = original_df.copy()

    print("\n🔧 Bước 1: Loại bỏ dữ liệu trùng lặp...")
    df = df[df.duplicated() == False].reset_index(drop=True)
    print(f"   ✓ Đã loại bỏ {len(original_df) - len(df)} dòng trùng lặp")

    print("\n🔧 Bước 2: Loại bỏ giá trị null/NaN...")
    before_null = len(df)
    df = df.dropna(subset=["Date", "Open", "High", "Low", "Close/Last", "Volume"])
    print(f"   ✓ Đã loại bỏ {before_null - len(df)} dòng có giá trị null")

    print("\n🔧 Bước 3: Kiểm tra logic giá (High >= Low, etc.)...")
    before_logic = len(df)
    df = df[
        (df["High"] >= df["Open"]) &
        (df["High"] >= df["Close/Last"]) &
        (df["High"] >= df["Low"]) &
        (df["Low"] <= df["Open"]) &
        (df["Low"] <= df["Close/Last"])
    ]
    df.reset_index(drop=True, inplace=True)
    print(f"   ✓ Đã loại bỏ {before_logic - len(df)} dòng có logic giá không hợp lệ")

    print(f"\n✅ Dữ liệu sau làm sạch: {len(df)} hàng")

    # ==========================================================
    # B2.4 - PHÁT HIỆN NOISE & OUTLIERS
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.4 - PHÁT HIỆN NOISE & OUTLIERS")
    print("=" * 60)

    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print("\n📊 OUTLIER DETECTION (IQR Method):")
    print("-" * 100)
    print(f"{'Column':<15} {'Q1':<12} {'Q3':<12} {'IQR':<12} {'Lower':<12} {'Upper':<12} {'Outliers':<10}")
    print("-" * 100)

    outlier_summary = []
    for col in quantitative_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        print(f"{col:<15} {Q1:<12.2f} {Q3:<12.2f} {IQR:<12.2f} {lower_bound:<12.2f} {upper_bound:<12.2f} {outlier_count:<10}")
        
        outlier_summary.append({
            "Column": col,
            "Outlier_Count": outlier_count
        })

    # Visualize Outliers
    print("\n📊 Visualizing Outliers...")
    fig, axes = plt.subplots(len(quantitative_cols), 1, figsize=(12, 4*len(quantitative_cols)))
    if len(quantitative_cols) == 1:
        axes = [axes]

    for idx, col in enumerate(quantitative_cols):
        sns.boxplot(data=df, y=col, ax=axes[idx], color='steelblue')
        axes[idx].set_title(f"Outlier Detection: {col}", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(col, fontsize=10)
        
        # Add statistics
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        axes[idx].text(0.02, 0.98, f"Outliers: {outlier_count}", 
                       transform=axes[idx].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("B2_outliers_detection.png", dpi=300, bbox_inches='tight')
    print("   ✓ Biểu đồ đã lưu: B2_outliers_detection.png")
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B2.5 - QUY TẮC XÁC THỰC DỮ LIỆU
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.5 - QUY TẮC XÁC THỰC DỮ LIỆU")
    print("=" * 60)

    print("\n✅ Các quy tắc đã áp dụng:")
    print("   1. High >= Open, Close/Last, Low")
    print("   2. Low <= Open, Close/Last")
    print("   3. Volume >= 0")
    print("   4. Date sorted in ascending order")
    print("   5. Duplicates removed")
    print("   6. Missing values removed")

    # Kiểm tra validation
    validation_passed = True

    # Check rule 1 & 2
    invalid_prices = df[
        ~((df["High"] >= df["Open"]) & 
          (df["High"] >= df["Close/Last"]) & 
          (df["High"] >= df["Low"]) & 
          (df["Low"] <= df["Open"]) & 
          (df["Low"] <= df["Close/Last"]))
    ]

    if len(invalid_prices) > 0:
        print(f"\n⚠️ Phát hiện {len(invalid_prices)} dòng có giá không hợp lệ")
        validation_passed = False
    else:
        print("\n✓ Tất cả giá đều hợp lệ")

    # Check rule 3
    if (df["Volume"] < 0).any():
        print("⚠️ Có giá trị Volume âm")
        validation_passed = False
    else:
        print("✓ Volume >= 0")

    # Check rule 4
    if not df["Date"].is_monotonic_increasing:
        print("⚠️ Date chưa được sắp xếp đúng")
        validation_passed = False
    else:
        print("✓ Date sorted correctly")

    if validation_passed:
        print("\n✅ Tất cả quy tắc validation đều PASSED!")

    # ==========================================================
    # B2.6 - SO SÁNH TRƯỚC & SAU LÀM SẠCH
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.6 - SO SÁNH TRƯỚC & SAU LÀM SẠCH")
    print("=" * 60)

    comparison_data = {
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicates", "Date Range"],
        "Before": [
            original_df.shape[0],
            original_df.shape[1],
            original_df.isnull().sum().sum(),
            original_df.duplicated().sum(),
            f"{original_df['Date'].min().date()} to {original_df['Date'].max().date()}"
        ],
        "After": [
            df.shape[0],
            df.shape[1],
            df.isnull().sum().sum(),
            df.duplicated().sum(),
            f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        ],
        "Change": [
            original_df.shape[0] - df.shape[0],
            original_df.shape[1] - df.shape[1],
            original_df.isnull().sum().sum() - df.isnull().sum().sum(),
            original_df.duplicated().sum() - df.duplicated().sum(),
            "N/A"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))

    # ==========================================================
    # B2.7 - LƯU DỮ LIỆU ĐÃ LÀM SẠCH
    # ==========================================================
    print("\n" + "=" * 60)
    print("B2.7 - LƯU DỮ LIỆU")
    print("=" * 60)

    df.to_csv("goldstock_cleaned_B2.csv", index=False)
    print("\n✅ Dữ liệu đã làm sạch được lưu vào: goldstock_cleaned_B2.csv")

    print("\n📝 Sample cleaned data (first 10 rows):")
    print(df.head(10).to_string())

    print("\n" + "=" * 60)
    print("KẾT THÚC B2 - TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)

if __name__ == "__main__":
    run()
