# ==========================================================
# QUYTRÌNH PHÂN TÍCH DỰ VIỄN GIÁ VÀNG
# Gold Price Forecasting Pipeline
# ==========================================================
# 
# QUY TRÌNH:
# B1: Mô tả dữ liệu (Data Description) - Tổng quan & Làm quen
# B2: Tiền xử lý dữ liệu (Data Cleaning) - Làm sạch
# B3: Khai phá dữ liệu (EDA) - Phân tích đơn biến & đa biến
# B4: Ma trận tương quan & PCA - Lựa chọn features
# B5: Mô hình & Dự báo - K-Means Clustering & Linear Regression
#
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Thêm các module cần thiết
import B1_data_description
import B2_data_cleaning
import B3_data_exploration
import B4_correlation_pca
# B5 chạy riêng qua Streamlit

def print_header(text, width=70):
    """In header với định dạng đẹp"""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)

def main():
    print_header("🏆 QUYTRÌNH PHÂN TÍCH DỰ VIỄN GIÁ VÀNG", 70)
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     Gold Price Forecasting Analysis Pipeline          ║
    ╚════════════════════════════════════════════════════════╝
    
    Quy trình gồm 5 bước chính:
    
    📊 B1: Mô tả dữ liệu (Data Description)
       └─ Tổng quan dữ liệu, phân loại biến, thống kê mô tả
    
    🧹 B2: Tiền xử lý dữ liệu (Data Cleaning)
       └─ Kiểm tra missing, duplicates, outliers, validation
    
    📈 B3: Khai phá dữ liệu (EDA)
       └─ Phân tích đơn biến, đa biến, chuỗi thời gian
    
    🔗 B4: Ma trận tương quan & PCA
       └─ Phân tích tương quan, lựa chọn features, giảm chiều
    
    🤖 B5: Mô hình & Dự báo
       └─ K-Means Clustering, Linear Regression (Streamlit GUI)
    
    """)
    
    print_header("Chọn bước muốn chạy", 70)
    print("""
    1. Chạy B1 - Mô tả dữ liệu
    2. Chạy B2 - Tiền xử lý dữ liệu
    3. Chạy B3 - Khai phá dữ liệu
    4. Chạy B4 - Ma trận tương quan & PCA
    5. Chạy B5 - Mô hình & Dự báo (Streamlit)
    6. Chạy tất cả (B1-B5)
    0. Thoát
    """)
    
    while True:
        choice = input("\n📍 Nhập lựa chọn (0-6): ").strip()
        
        if choice == "0":
            print("\n👋 Tạm biệt! Cảm ơn bạn đã sử dụng dịch vụ.")
            break
        
        elif choice == "1":
            print_header("▶️ Chạy B1: Mô tả dữ liệu", 70)
            print("\n📊 Phân tích: Shape, Phân loại dữ liệu, Thống kê mô tả, Mẫu dữ liệu...")
            try:
                B1_data_description.run()
                print("\n✅ B1 hoàn tất!")
            except Exception as e:
                print(f"\n❌ Lỗi B1: {e}")
        
        elif choice == "2":
            print_header("▶️ Chạy B2: Tiền xử lý dữ liệu", 70)
            print("\n🧹 Xử lý: Missing, Duplicates, Outliers, Validation...")
            try:
                B2_data_cleaning.run()
                print("\n✅ B2 hoàn tất!")
            except Exception as e:
                print(f"\n❌ Lỗi B2: {e}")
        
        elif choice == "3":
            print_header("▶️ Chạy B3: Khai phá dữ liệu", 70)
            print("\n📈 Phân tích: Line Chart, Histogram, Scatter Plot, Boxplot...")
            try:
                B3_data_exploration.run()
                print("\n✅ B3 hoàn tất!")
            except Exception as e:
                print(f"\n❌ Lỗi B3: {e}")
        
        elif choice == "4":
            print_header("▶️ Chạy B4: Ma trận tương quan & PCA", 70)
            print("\n🔗 Phân tích: Correlation Matrix, High Correlations, Features Selection, PCA...")
            try:
                B4_correlation_pca.run()
                print("\n✅ B4 hoàn tất!")
            except Exception as e:
                print(f"\n❌ Lỗi B4: {e}")
        
        elif choice == "5":
            print_header("▶️ Chạy B5: Mô hình & Dự báo (Streamlit)", 70)
            print("\n🤖 Chạy Streamlit UI: K-Means Clustering & Linear Regression Prediction...")
            print("\n💡 Gợi ý: Mở terminal mới và chạy lệnh:")
            print("   streamlit run main.py")
            os.system("streamlit run main.py")
        
        elif choice == "6":
            print_header("▶️ Chạy tất cả (B1-B5)", 70)
            
            try:
                print("\n⏳ Bước 1/5: B1 - Mô tả dữ liệu...")
                B1_data_description.run()
                print("✅ B1 hoàn tất!\n")
            except Exception as e:
                print(f"❌ Lỗi B1: {e}\n")
            
            try:
                print("\n⏳ Bước 2/5: B2 - Tiền xử lý dữ liệu...")
                B2_data_cleaning.run()
                print("✅ B2 hoàn tất!\n")
            except Exception as e:
                print(f"❌ Lỗi B2: {e}\n")
            
            try:
                print("\n⏳ Bước 3/5: B3 - Khai phá dữ liệu...")
                B3_data_exploration.run()
                print("✅ B3 hoàn tất!\n")
            except Exception as e:
                print(f"❌ Lỗi B3: {e}\n")
            
            try:
                print("\n⏳ Bước 4/5: B4 - Ma trận tương quan & PCA...")
                B4_correlation_pca.run()
                print("✅ B4 hoàn tất!\n")
            except Exception as e:
                print(f"❌ Lỗi B4: {e}\n")
            
            print("\n⏳ Bước 5/5: B5 - Mô hình & Dự báo (Streamlit)")
            print("\n🤖 Chạy Streamlit UI: K-Means Clustering & Linear Regression Prediction...")
            print("\n💡 Gợi ý: Mở terminal mới và chạy lệnh:")
            print("   streamlit run main.py\n")
            
            input("\n📍 Nhấn Enter để hoàn tất...")
        
        else:
            print("\n❌ Lựa chọn không hợp lệ. Vui lòng chọn lại.")
        
        print("\n" + "-" * 70)
        again = input("Bạn muốn tiếp tục? (y/n): ").strip().lower()
        if again != 'y':
            print("\n👋 Tạm biệt!")
            break

if __name__ == "__main__":
    main()
