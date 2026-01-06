# ==========================================================
# B1 – DATA DESCRIPTION (YAHOO FINANCE STYLE)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

def run():
    print("=" * 70)
    print("B1 - MÔ TẢ DỮ LIỆU CHỨNG KHOÁN (YAHOO FINANCE FORMAT)")
    print("=" * 70)

    # ==========================================================
    # LOAD DATA
    # ==========================================================
    df = pd.read_csv("goldstock v2.csv", sep=";", decimal=",")  # Dùng dấu phẩy cho decimal

    print("\n📋 Columns in CSV:")
    print(df.columns.tolist())

    # Chuẩn hoá tên cột
    df.columns = df.columns.str.strip()

    # ==========================================================
    # RENAME COLUMNS nếu dữ liệu cũ
    # ==========================================================
    rename_map = {
        "Close/Last": "Close",
        "Adj_Close": "Adj Close",
        "AdjClose": "Adj Close"
    }
    df.rename(columns=rename_map, inplace=True)

    # ==========================================================
    # CONVERT DATA TYPES
    # ==========================================================
    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date chuẩn Yahoo Finance
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=['Date', 'Close'])  # Chỉ kiểm tra các cột quan trọng
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    print(f"\n✅ Data loaded successfully: {df.shape[0]} rows")

    # ==========================================================
    # B1.1 - DATASET OVERVIEW
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.1 - TỔNG QUAN DỮ LIỆU")
    print("=" * 70)

    print(f"📊 Số dòng: {df.shape[0]}")
    print(f"📊 Số cột: {df.shape[1]}")
    print(f"📅 Khoảng thời gian: {df['Date'].min().strftime('%Y-%m-%d')} → {df['Date'].max().strftime('%Y-%m-%d')}")

    # ==========================================================
    # B1.2 - DATA TYPE CLASSIFICATION
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.2 - PHÂN LOẠI DỮ LIỆU")
    print("=" * 70)

    quantitative_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    qualitative_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    print("\n📈 DỮ LIỆU ĐỊNH LƯỢNG:")
    for col in quantitative_cols:
        print(f"  ✓ {col}")

    print("\n📝 DỮ LIỆU ĐỊNH TÍNH:")
    for col in qualitative_cols:
        print(f"  ✓ {col}")

    # ==========================================================
    # B1.3 - DESCRIPTIVE STATISTICS
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.3 - THỐNG KÊ MÔ TẢ")
    print("=" * 70)

    print(df[quantitative_cols].describe())

    # ==========================================================
    # B1.4 - DETAILED COLUMN INFO
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.4 - THÔNG TIN CHI TIẾT CỘT")
    print("=" * 70)

    print(f"{'Column':<15}{'Type':<15}{'Non-Null':<12}{'Min':<15}{'Max':<15}")
    print("-" * 70)

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()

        if col in quantitative_cols:
            min_val = f"{df[col].min():.2f}"
            max_val = f"{df[col].max():.2f}"
        else:
            min_val = "N/A"
            max_val = "N/A"

        print(f"{col:<15}{str(dtype):<15}{non_null:<12}{min_val:<15}{max_val:<15}")

    # ==========================================================
    # B1.5 - SAMPLE DATA
    # ==========================================================
    print("\n" + "=" * 70)
    print("B1.5 - MẪU DỮ LIỆU")
    print("=" * 70)

    print("\n🔹 5 dòng đầu:")
    print(df.head().to_string(index=False))

    print("\n🔹 5 dòng cuối:")
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
