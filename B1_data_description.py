# ==========================================================
# B1 – DATA DESCRIPTION (YAHOO FINANCE STYLE)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os

def run():
    print("=" * 70)
    print("B1 - MÔ TẢ DỮ LIỆU CHỨNG KHOÁN (YAHOO FINANCE FORMAT)")
    print("=" * 70)

    # ==========================================================
    # LOAD DATA - TỰ ĐỘNG TÌM FILE
    # ==========================================================
    # Ưu tiên file đã xử lý, nếu không có thì dùng file gốc
    data_files = [
        "goldstock_processed_B1.csv",
        "goldstock_cleaned_B2.csv",
        "goldstock v2.csv"
    ]
    
    df = None
    selected_file = None
    
    for file in data_files:
        if os.path.exists(file):
            try:
                if file == "goldstock v2.csv":
                    df = pd.read_csv(file, sep=";")
                else:
                    df = pd.read_csv(file)
                selected_file = file
                break
            except Exception as e:
                print(f"⚠️  Lỗi khi đọc {file}: {e}")
                continue
    
    if df is None:
        raise FileNotFoundError("❌ Không tìm thấy file CSV nào!")
    
    print(f"\n✅ File được tải: {selected_file}")
    print(f"📋 Columns in CSV: {df.columns.tolist()}")

    # Chuẩn hoá tên cột
    df.columns = df.columns.str.strip()

    # ==========================================================
    # REMOVE UNNECESSARY COLUMNS (Column1, Unnamed)
    # ==========================================================
    cols_to_drop = [col for col in df.columns if "column" in col.lower() or "unnamed" in col.lower()]
    if cols_to_drop:
        print(f"🗑️  Xoá cột không cần thiết: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    # ==========================================================
    # CONVERT DATA TYPES
    # ==========================================================
    # Convert numeric columns
    numeric_cols = ["Open", "High", "Low", "Close/Last", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Chuẩn hoá đơn vị giá nếu bị phóng đại 10x (23694 -> 2369.4)
    price_cols = ["Open", "High", "Low", "Close/Last"]
    if "Close/Last" in df.columns:
        price_median = df["Close/Last"].median()
        if price_median > 5000:  # nhận diện giá bị lệch 1 bậc thập phân
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] / 10.0
            print("ℹ️ Đã chuẩn hóa đơn vị giá (chia 10) cho các cột Open/High/Low/Close/Last")

    # Convert Date to datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    # Remove invalid rows
    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n✅ Data loaded successfully: {df.shape[0]} rows")

    # ==========================================================
    # B1.1 - DATASET OVERVIEW (SHAPE - KÍCH THƯỚC DỮ LIỆU)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.1 - KÍCH THƯỚC DỮ LIỆU (Data Shape)")
    print("=" * 70)

    print(f"\n📊 SHAPE (Số hàng, Số cột): {df.shape}")
    print(f"   • Số hàng (Rows/Observations): {df.shape[0]}")
    print(f"   • Số cột (Columns/Features): {df.shape[1]}")
    
    # Handle NaT values in Date column
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        print(f"\n📅 Khoảng thời gian: {min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}")
    else:
        print(f"\n📅 Khoảng thời gian: {min_date} → {max_date}")

    # ==========================================================
    # B1.2 - DATA TYPE CLASSIFICATION (PHÂN LOẠI DỮ LIỆU)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.2 - PHÂN LOẠI DỮ LIỆU (Data Classification)")
    print("=" * 70)

    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    qualitative_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # Tạo bảng phân loại chi tiết
    classification_data = []
    for i, col in enumerate(df.columns, 1):
        if col in quantitative_cols:
            col_type = "Numerical (Định lượng)"
            description = "Biến số, có thể tính toán (Mean, Max, Min, ...)"
        else:
            col_type = "Categorical/Ordinal (Định tính)"
            description = "Biến phân loại (thời gian, chỉ số, ...)"
        
        classification_data.append({
            "STT": i,
            "Column Name": col,
            "Data Type": col_type,
            "Description": description
        })
    
    classification_df = pd.DataFrame(classification_data)
    print("\n" + classification_df.to_string(index=False))

    print(f"\n" + "=" * 70)
    print(f"📊 PHÂN LOẠI CHI TIẾT:")
    print("=" * 70)
    
    print(f"\n🔢 DỮ LIỆU ĐỊNH LƯỢNG (Numerical): {len(quantitative_cols)} cột")
    print(f"   Các biến số thực/nguyên - có thể tính toán (Mean, Max, Min, Std, ...)")
    for col in quantitative_cols:
        print(f"   ✓ {col}")

    print(f"\n📝 DỮ LIỆU ĐỊNH TÍNH (Categorical/Ordinal): {len(qualitative_cols)} cột")
    print(f"   Các biến phân loại như thời gian (Date), chỉ số thứ tự, ...")
    for col in qualitative_cols:
        print(f"   ✓ {col}")

    # ==========================================================
    # B1.3 - DESCRIPTIVE STATISTICS (THỐNG KÊ MÔ TẢ)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.3 - THỐNG KÊ MÔ TẢ (Descriptive Statistics)")
    print("=" * 70)
    print("\nThống kê cho các biến Định lượng (Numerical):")
    print(df[quantitative_cols].describe().T)

    # ==========================================================
    # B1.4 - DETAILED COLUMN INFORMATION (THÔNG TIN CHI TIẾT CỘT)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.4 - THÔNG TIN CHI TIẾT CỘT (Column Information)")
    print("=" * 70)

    print(f"\n{'Column':<15}{'Data Type':<15}{'Non-Null':<12}{'Min/Unique':<20}{'Max':<15}")
    print("-" * 77)

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()

        if col in quantitative_cols:
            min_val = f"{df[col].min():.2f}"
            max_val = f"{df[col].max():.2f}"
        else:
            min_val = f"{df[col].nunique()} unique values"
            max_val = "—"

        print(f"{col:<15}{str(dtype):<15}{non_null:<12}{min_val:<20}{max_val:<15}")

    # ==========================================================
    # B1.5 - SAMPLE DATA (MẪU DỮ LIỆU)
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.5 - MẪU DỮ LIỆU (Sample Data)")
    print("=" * 70)

    print("\n🔹 5 dòng đầu tiên:")
    print(df.head().to_string(index=False))

    print("\n\n🔹 5 dòng cuối cùng:")
    print(df.tail().to_string(index=False))

    # ==========================================================
    # SAVE PROCESSED DATA
    # ==========================================================
    df.to_csv("goldstock_processed_B1.csv", index=False)
    print("\n✅ Saved: goldstock_processed_B1.csv")
    print("=" * 70)
    print("KẾT THÚC B1")
    print("=" * 70)

if __name__ == "__main__":
    run()
