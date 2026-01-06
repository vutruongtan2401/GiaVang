# ==========================================================
# B4 – MA TRẬN TƯƠNG QUAN & GIẢM CHIỀU DỮ LIỆU
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.style.use('ggplot')

def run():
    # ==========================================================
    # LOAD DỮ LIỆU
    # ==========================================================
    print("=" * 60)
    print("B4 - MA TRẬN TƯƠNG QUAN & GIẢM CHIỀU")
    print("=" * 60)

    # Load từ file đã làm sạch
    try:
        df = pd.read_csv("goldstock_cleaned_B2.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        print(f"\n✅ Đã load dữ liệu từ B2: {len(df)} hàng")
    except:
        print("\n⚠️ Không tìm thấy file B2, load từ file gốc...")
        df = pd.read_csv("goldstock v2.csv", sep=";")
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
    print(f"📊 Các cột định lượng ban đầu: {quantitative_cols}")
    
    # ==========================================================
    # B4.0 - FEATURE ENGINEERING (ĐẶC TRƯNG PHÁI SINH)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.0 - FEATURE ENGINEERING (ĐẶC TRƯNG PHÁI SINH)")
    print("=" * 60)
    
    # Sắp xếp theo Date để tính Return đúng
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. Return (Lợi nhuận): % thay đổi giá Close
    df['Return'] = df['Close/Last'].pct_change() * 100
    print("\n1️⃣ Return (Lợi nhuận):")
    print("   • Công thức: (Close_today - Close_yesterday) / Close_yesterday * 100")
    print("   • Ý nghĩa: % thay đổi giá so với ngày trước")
    
    # 2. Range (Biên độ): High - Low
    df['Range'] = df['High'] - df['Low']
    print("\n2️⃣ Range (Biên độ dao động):")
    print("   • Công thức: High - Low")
    print("   • Ý nghĩa: Độ biến động giá trong ngày")
    
    # 3. Volatility (Độ biến động 7 ngày)
    df['Volatility_7d'] = df['Return'].rolling(window=7, min_periods=1).std()
    print("\n3️⃣ Volatility (Độ biến động 7 ngày):")
    print("   • Công thức: Std deviation của Return trong 7 ngày")
    print("   • Ý nghĩa: Mức độ rủi ro/biến động giá")
    
    # 4. Volume Change (% thay đổi khối lượng)
    df['Volume_Change'] = df['Volume'].pct_change() * 100
    print("\n4️⃣ Volume Change (Thay đổi khối lượng):")
    print("   • Công thức: (Volume_today - Volume_yesterday) / Volume_yesterday * 100")
    print("   • Ý nghĩa: % thay đổi khối lượng giao dịch")
    
    # Loại bỏ NaN
    df_features = df.dropna().reset_index(drop=True)
    
    print(f"\n✅ Đã tạo 4 features mới!")
    print(f"📊 Dữ liệu sau feature engineering: {len(df_features)} hàng")
    print(f"\n📋 Sample 5 dòng đầu:")
    print(df_features[['Date', 'Close/Last', 'Return', 'High', 'Low', 'Range', 'Volume', 'Volume_Change', 'Volatility_7d']].head(5).to_string(index=False))
    
    # Cập nhật danh sách cột định lượng
    quantitative_cols = ['Close/Last', 'Volume', 'Open', 'High', 'Low', 'Return', 'Range', 'Volatility_7d', 'Volume_Change']
    df = df_features.copy()
    print(f"\n📊 Các cột định lượng mới: {quantitative_cols}")

    # ==========================================================
    # B4.1 - MA TRẬN TƯƠNG QUAN (CORRELATION MATRIX)
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.1 - MA TRẬN TƯƠNG QUAN")
    print("=" * 60)

    # Tính ma trận tương quan
    corr_matrix = df[quantitative_cols].corr()

    print("\n📊 MA TRẬN TƯƠNG QUAN:")
    print(corr_matrix.round(4))

    # Visualize correlation matrix
    fig, ax = plt.subplots(figsize=(12, 11))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                square=True, fmt=".3f", cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=1, linecolor='black',
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Ma trận tương quan (Correlation Matrix)", fontsize=14, fontweight='bold', pad=20)
    
    # Thêm nhận xét tổng quan
    comment = (
        "💡 NHẬN XÉT: Các biến giá (Open, High, Low, Close/Last) có tương quan RẤT CAO (>0.95) → Chứa thông tin trùng lặp.\n"
        "Volume tương quan yếu với giá → Thông tin độc lập quan trọng. Features mới (Return, Range, Volatility) bổ sung thông tin."
    )
    fig.text(0.5, -0.01, comment, ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), wrap=True)
    
    plt.tight_layout()
    plt.savefig("B4_correlation_matrix.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_correlation_matrix.png")
    
    print("\n💡 NHẬN XÉT VỀ MA TRẬN TƯƠNG QUAN:")
    print("   📌 Màu sắc:")
    print("      • Đỏ đậm (gần +1): Tương quan THUẬN rất mạnh")
    print("      • Xanh đậm (gần -1): Tương quan NGHỊCH rất mạnh")
    print("      • Trắng (gần 0): Không có tương quan")
    print("\n   📌 Phát hiện chính:")
    
    # Tìm các cặp tương quan cao nhất
    high_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for idx, (col1, col2, val) in enumerate(high_pairs[:5], 1):
        print(f"      {idx}. {col1} ↔ {col2}: {val:.3f}")
    
    print("\n   📌 Ý nghĩa:")
    print("      • Các biến giá (Open, High, Low, Close/Last) có tương quan RẤT CAO")
    print("      • → Chứa thông tin trùng lặp → Cần loại bỏ để tránh multicollinearity")
    print("      • Volume tương quan yếu với giá → Thông tin độc lập quan trọng")
    print("      • Features mới (Return, Range, Volatility) mang thông tin bổ sung")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B4.2 - PHÂN TÍCH TƯƠNG QUAN CAO
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.2 - PHÂN TÍCH CÁC CẶP TƯƠNG QUAN CAO")
    print("=" * 60)

    # Tìm các cặp có tương quan cao (> 0.95)
    high_corr_threshold = 0.95
    high_corr_pairs = []

    print(f"\n🔍 Các cặp biến có tương quan > {high_corr_threshold}:")
    print("-" * 80)
    print(f"{'Variable 1':<20} {'Variable 2':<20} {'Correlation':<15} {'Interpretation':<25}")
    print("-" * 80)

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > high_corr_threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                interpretation = "Very Strong Positive" if corr_value > 0 else "Very Strong Negative"
                print(f"{col1:<20} {col2:<20} {corr_value:<15.4f} {interpretation:<25}")
                high_corr_pairs.append({
                    "Variable 1": col1,
                    "Variable 2": col2,
                    "Correlation": round(corr_value, 4)
                })

    if not high_corr_pairs:
        print("✅ Không có cặp biến nào có tương quan > 0.95")
    
    # ==========================================================
    # B4.2.1 - PHÂN TÍCH VOLUME VỚI CÁC FEATURES MỚI
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.2.1 - PHÂN TÍCH VOLUME VỚI FEATURES PHÁI SINH")
    print("=" * 60)
    
    print("\n📊 Tương quan của Volume với các biến:")
    print("-" * 60)
    print(f"{'Biến':<25} {'Correlation với Volume':<25} {'Đánh giá':<20}")
    print("-" * 60)
    
    volume_corrs = {}
    for col in quantitative_cols:
        if col != 'Volume' and col in df.columns:
            corr_val = df['Volume'].corr(df[col])
            volume_corrs[col] = corr_val
            
            # Đánh giá mức độ tương quan
            if abs(corr_val) < 0.3:
                strength = "Yếu"
            elif abs(corr_val) < 0.7:
                strength = "Trung bình"
            else:
                strength = "Mạnh"
            
            print(f"{col:<25} {corr_val:<25.4f} {strength:<20}")
    
    print("\n💡 NHẬN XÉT:")
    
    # Tìm biến tương quan mạnh nhất với Volume
    max_corr_var = max(volume_corrs.items(), key=lambda x: abs(x[1]))
    print(f"   • Biến tương quan MẠNH NHẤT với Volume: {max_corr_var[0]} ({max_corr_var[1]:.4f})")
    
    # So sánh với giá thô
    if 'Range' in volume_corrs and 'Close/Last' in volume_corrs:
        if abs(volume_corrs['Range']) > abs(volume_corrs['Close/Last']):
            print(f"   • Volume tương quan với 'Range' ({volume_corrs['Range']:.4f}) MẠNH HƠN với 'Close/Last' ({volume_corrs['Close/Last']:.4f})")
            print("   • → Volume phản ánh độ biến động giá (Range) tốt hơn mức giá tuyệt đối")
        else:
            print(f"   • Volume tương quan với 'Close/Last' ({volume_corrs['Close/Last']:.4f}) hơn 'Range' ({volume_corrs['Range']:.4f})")
    
    if 'Volatility_7d' in volume_corrs:
        print(f"   • Volume vs Volatility_7d: {volume_corrs['Volatility_7d']:.4f}")
        if abs(volume_corrs['Volatility_7d']) > 0.3:
            print("   • → Có mối liên hệ giữa khối lượng giao dịch và độ biến động giá")

    # ==========================================================
    # B4.3 - LẬP LUẬN GIỮ/BỎ CỘT
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.3 - LẬP LUẬN GIỮ/BỎ CỘT (FEATURE SELECTION)")
    print("=" * 60)

    print("\n📋 PHÂN TÍCH & LẬP LUẬN:")
    print("-" * 80)

    print("\n1️⃣ NHÓM GIÁ (Open, High, Low, Close/Last):")
    price_cols = ["Open", "High", "Low", "Close/Last"]
    price_corr = df[price_cols].corr()
    print(f"\n   Ma trận tương quan nhóm giá:")
    print(price_corr.round(4))

    print("\n   📊 Phân tích:")
    print("   • Open, High, Low, Close/Last có tương quan RẤT CAO (> 0.95)")
    print("   • Điều này là HỢP LÝ vì tất cả đều là giá trong cùng 1 ngày giao dịch")
    print("   • Giữ tất cả 4 cột → DƯ THỪA THÔNG TIN (Multicollinearity)")
    print()
    print("   🎯 QUYẾT ĐỊNH:")
    print("   ✅ GIỮ: Close/Last")
    print("      → Lý do: Giá đóng cửa là chỉ báo quan trọng nhất")
    print("      → Phản ánh giá cuối ngày, thường dùng để phân tích xu hướng")
    print("      → Là baseline cho tính toán return")
    print()
    print("   ❌ BỎ: Open, High, Low")
    print("      → Lý do: Có thể suy luận từ Close/Last")
    print("      → Tương quan quá cao → không mang thông tin mới")
    print("      → Giảm redundancy, tránh overfitting")

    print("\n2️⃣ KHỐI LƯỢNG GIAO DỊCH (Volume):")
    print(f"\n   Tương quan với các biến giá:")
    for col in price_cols:
        if col in df.columns:
            corr_vol = df["Volume"].corr(df[col])
            print(f"   • Volume vs {col}: {corr_vol:.4f}")

    print("\n   📊 Phân tích:")
    print("   • Volume có tương quan YẾU với các biến giá")
    print("   • Volume phản ánh mức độ quan tâm/thanh khoản thị trường")
    print("   • Thông tin ĐỘC LẬP, không thể suy ra từ giá")
    print()
    print("   🎯 QUYẾT ĐỊNH:")
    print("   ✅ GIỮ: Volume")
    print("      → Lý do: Mang thông tin độc lập")
    print("      → Hữu ích cho phân tích khối lượng-giá")
    print("      → Chỉ báo quan trọng trong phân tích kỹ thuật")

    print("\n" + "=" * 80)
    print("✅ KẾT LUẬN CUỐI CÙNG:")
    print("=" * 80)
    print("\n   📌 CỘT GIỮ LẠI (2 cột):")
    print("      1. Close/Last (đại diện nhóm giá)")
    print("      2. Volume (thông tin độc lập)")
    print()
    print("   📌 CỘT BỎ ĐI (3 cột):")
    print("      1. Open")
    print("      2. High")
    print("      3. Low")
    print()
    print("   📊 KẾT QUẢ: Giảm từ 5 cột → 2 cột (giảm 60%)")
    print("   ✓ Giữ lại thông tin quan trọng")
    print("   ✓ Loại bỏ multicollinearity")
    print("   ✓ Tăng hiệu quả model, giảm overfitting")

    # ==========================================================
    # B4.4 - GIẢM CHIỀU DỮ LIỆU VỚI PCA
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.4 - GIẢM CHIỀU DỮ LIỆU (PCA)")
    print("=" * 60)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[quantitative_cols])

    print("\n🔧 Bước 1: Chuẩn hóa dữ liệu (Standardization)")
    print("   ✓ Mean = 0, Std = 1")

    # Áp dụng PCA với tất cả components
    pca_full = PCA()
    pca_full.fit(X_scaled)

    print("\n📊 Bước 2: Phân tích tất cả Principal Components")
    print("-" * 80)
    print(f"{'PC':<10} {'Explained Var %':<20} {'Cumulative %':<20} {'Eigenvalue':<15}")
    print("-" * 80)

    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    for i in range(len(pca_full.explained_variance_ratio_)):
        print(f"PC{i+1:<9} {pca_full.explained_variance_ratio_[i]*100:<20.2f} "
              f"{cumsum_var[i]*100:<20.2f} {pca_full.explained_variance_[i]:<15.4f}")

    # Visualize explained variance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scree plot
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1), 
                pca_full.explained_variance_ratio_*100,
                alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    axes[0].plot(range(1, len(pca_full.explained_variance_ratio_)+1), 
                 pca_full.explained_variance_ratio_*100,
                 'ro-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Principal Component', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Explained Variance (%)', fontsize=11, fontweight='bold')
    axes[0].set_title('Scree Plot - Individual Variance Explained', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(1, len(pca_full.explained_variance_ratio_)+1))
    axes[0].grid(True, alpha=0.3, axis='y')

    # Cumulative variance plot
    axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var*100, 
                 'bo-', linewidth=2, markersize=8)
    axes[1].axhline(y=95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
    axes[1].axhline(y=90, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
    axes[1].fill_between(range(1, len(cumsum_var)+1), 0, cumsum_var*100, alpha=0.2, color='steelblue')
    axes[1].set_xlabel('Number of Components', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Cumulative Explained Variance (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(1, len(cumsum_var)+1))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Thêm nhận xét tổng quan
    n_for_90 = np.argmax(cumsum_var >= 0.90) + 1
    n_for_95 = np.argmax(cumsum_var >= 0.95) + 1
    comment = (
        f"💡 NHẬN XÉT: PC1 chiếm ĐẠO {pca_full.explained_variance_ratio_[0]*100:.1f}%, cần {n_for_90} PC cho 90% thông tin, {n_for_95} PC cho 95%.\n"
        f"Scree Plot (trái) cho thấy 'điểm khuỷu tay' → Các PC sau ít quan trọng hơn. Dùng 2 PC (giữ ~{cumsum_var[1]*100:.1f}%) cho dễ trực quan hóa."
    )
    fig.text(0.5, -0.02, comment, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), wrap=True)
    
    plt.tight_layout()
    plt.savefig("B4_pca_variance_explained.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_pca_variance_explained.png")
    
    print("\n💡 NHẬN XÉT VỀ BIỂU ĐỒ PHƯƠNG SAI (VARIANCE EXPLAINED):")
    print("\n   📊 Biểu đồ bên TRÁI (Scree Plot - Individual Variance):")
    print("      • Cho thấy % phương sai mà MỖI thành phần chính (PC) giải thích")
    print(f"      • PC1 chiếm ĐẠO: {pca_full.explained_variance_ratio_[0]*100:.2f}% - Quan trọng nhất")
    print(f"      • PC2 chiếm: {pca_full.explained_variance_ratio_[1]*100:.2f}%")
    print("      • Các PC sau càng giảm dần → Ít quan trọng hơn")
    print("      • Đường đỏ giảm nhanh ban đầu, sau đó 'gối' (flatten) → Điểm 'khuỷu tay'")
    print("\n   📊 Biểu đồ bên PHẢI (Cumulative Variance):")
    print("      • Cho thấy % phương sai TÍCH LŨY khi thêm dần các PC")
    
    # Tìm số PC cần thiết cho 90% và 95%
    n_for_90 = np.argmax(cumsum_var >= 0.90) + 1
    n_for_95 = np.argmax(cumsum_var >= 0.95) + 1
    
    print(f"      • Đường cam (90%): Cần {n_for_90} PC để giữ 90% thông tin")
    print(f"      • Đường đỏ (95%): Cần {n_for_95} PC để giữ 95% thông tin")
    print("      • Đường xanh tăng dần → Cần cân bằng giữa số chiều và thông tin giữ lại")
    print("\n   🎯 KẾT LUẬN:")
    print(f"      • Chỉ cần {n_for_90}-{n_for_95} thành phần chính đầu tiên")
    print("      • Có thể giảm mạnh số chiều mà vẫn giữ được phần lớn thông tin")
    print(f"      • Trong project này, dùng 2 PC (giữ ~{cumsum_var[1]*100:.1f}% thông tin) để dễ trực quan hóa")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B4.5 - PCA VỚI 2 COMPONENTS
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.5 - PCA VỚI 2 PRINCIPAL COMPONENTS")
    print("=" * 60)

    # Áp dụng PCA với 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n📊 KẾT QUẢ PCA (2 Components):")
    print(f"   • PC1 giải thích: {pca.explained_variance_ratio_[0]*100:.2f}% phương sai")
    print(f"   • PC2 giải thích: {pca.explained_variance_ratio_[1]*100:.2f}% phương sai")
    print(f"   • Tổng cộng: {sum(pca.explained_variance_ratio_)*100:.2f}% phương sai")
    print(f"\n   ✅ Giảm từ {len(quantitative_cols)} chiều → 2 chiều")
    print(f"   ✅ Giữ lại {sum(pca.explained_variance_ratio_)*100:.2f}% thông tin")

    # Feature loadings
    print("\n📋 FEATURE LOADINGS (Đóng góp của từng biến):")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=quantitative_cols
    )
    print(loadings_df.round(4))

    # Visualize PCA projection
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 2D scatter plot
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=range(len(df)), cmap='viridis', 
                             alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', 
                       fontsize=11, fontweight='bold')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', 
                       fontsize=11, fontweight='bold')
    axes[0].set_title('PCA Projection (2D)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Time Index')

    # Biplot (PCA with loadings)
    for i, col in enumerate(quantitative_cols):
        axes[1].arrow(0, 0, 
                     pca.components_[0, i]*3, pca.components_[1, i]*3,
                     head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
        axes[1].text(pca.components_[0, i]*3.2, pca.components_[1, i]*3.2, 
                    col, fontsize=10, fontweight='bold', ha='center')

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                   alpha=0.3, s=30, color='steelblue', edgecolors='black', linewidth=0.3)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', 
                      fontsize=11, fontweight='bold')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', 
                      fontsize=11, fontweight='bold')
    axes[1].set_title('PCA Biplot (with Feature Loadings)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linewidth=0.5)

    # Thêm nhận xét tổng quan
    comment = (
        "💡 NHẬN XÉT: Biểu đồ trái: Mỗi điểm = 1 ngày giao dịch, màu sắc theo thời gian (tím→vàng: sớm→muộn).\n"
        "Biểu đồ phải (Biplot): Mũi tên đỏ = các biến gốc. Chiều dài = mức quan trọng, hướng = tương quan với PC1/PC2.\n"
        "Các biến giá cùng hướng → Tương quan cao. Volume hướng khác → Thông tin độc lập. PCA giúp giảm chiều, dễ phát hiện patterns."
    )
    fig.text(0.5, -0.02, comment, ha='center', va='top', fontsize=9.5,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), wrap=True)
    
    plt.tight_layout()
    plt.savefig("B4_pca_projection.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_pca_projection.png")
    
    print("\n💡 NHẬN XÉT VỀ BIỂU ĐỒ PCA PROJECTION:")
    print("\n   📊 Biểu đồ bên TRÁI (PCA Projection 2D):")
    print("      • Mỗi điểm = 1 ngày giao dịch, được chiếu xuống không gian 2 chiều (PC1, PC2)")
    print("      • Màu sắc từ tím → vàng: Thời gian từ sớm → muộn (theo thứ tự trong dataset)")
    print("      • Trục hoành (PC1): Thành phần chính 1 - Giải thích nhiều thông tin nhất")
    print("      • Trục tung (PC2): Thành phần chính 2 - Giải thích thông tin bổ sung")
    print("\n   📊 Biểu đồ bên PHẢI (Biplot - PCA với Feature Loadings):")
    print("      • Các mũi tên ĐỎ: Đại diện cho các biến gốc")
    print("      • Chiều dài mũi tên: Mức độ quan trọng của biến trong không gian PC")
    print("      • Hướng mũi tên: Chiều tương quan của biến với PC1/PC2")
    print("      • Các điểm xanh: Dữ liệu gốc chiếu xuống không gian PC")
    print("\n   📌 Phân tích Loadings:")
    
    # Phân tích hướng các biến
    for col in quantitative_cols:
        pc1_load = pca.components_[0, quantitative_cols.index(col)]
        pc2_load = pca.components_[1, quantitative_cols.index(col)]
        if abs(pc1_load) > 0.3:
            direction1 = "dương" if pc1_load > 0 else "âm"
            print(f"      • {col}: Loading PC1 = {pc1_load:.3f} ({direction1}) → Ảnh hưởng {'mạnh' if abs(pc1_load) > 0.4 else 'trung bình'} đến PC1")
        if abs(pc2_load) > 0.3:
            direction2 = "dương" if pc2_load > 0 else "âm"
            print(f"      • {col}: Loading PC2 = {pc2_load:.3f} ({direction2}) → Ảnh hưởng {'mạnh' if abs(pc2_load) > 0.4 else 'trung bình'} đến PC2")
    
    print("\n   🎯 Ý NGHĨA:")
    print("      • Các biến giá (Close, Open, High, Low) thường cùng hướng → Tương quan cao")
    print("      • Volume thường có hướng khác → Thông tin độc lập")
    print("      • Mô hình PCA giúp giảm chiều dữ liệu, dễ trực quan hóa mẫu (patterns)")
    print("      • Có thể phát hiện các nhóm (clusters) hoặc xu hướng theo thời gian")
    
    try:
        plt.close(fig)
    except Exception:
        pass

    # ==========================================================
    # B4.6 - GIẢI THÍCH PRINCIPAL COMPONENTS
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.6 - GIẢI THÍCH PRINCIPAL COMPONENTS")
    print("=" * 60)

    print("\n📖 PHÂN TÍCH LOADINGS:")

    print("\n🔵 PRINCIPAL COMPONENT 1 (PC1):")
    pc1_loadings = loadings_df["PC1"].abs().sort_values(ascending=False)
    print("   Đóng góp theo thứ tự:")
    for col in pc1_loadings.index:
        loading = loadings_df.loc[col, "PC1"]
        print(f"   • {col}: {loading:.4f} ({abs(loading)*100:.2f}%)")
    print("\n   💡 Ý nghĩa:")
    if abs(loadings_df.loc["Close/Last", "PC1"]) > 0.4:
        print("   → PC1 chủ yếu đại diện cho MỨC GIÁ CHUNG")
        print("   → Phản ánh xu hướng giá tổng thể của vàng")

    print("\n🔵 PRINCIPAL COMPONENT 2 (PC2):")
    pc2_loadings = loadings_df["PC2"].abs().sort_values(ascending=False)
    print("   Đóng góp theo thứ tự:")
    for col in pc2_loadings.index:
        loading = loadings_df.loc[col, "PC2"]
        print(f"   • {col}: {loading:.4f} ({abs(loading)*100:.2f}%)")
    print("\n   💡 Ý nghĩa:")
    if abs(loadings_df.loc["Volume", "PC2"]) > 0.3:
        print("   → PC2 liên quan đến KHỐI LƯỢNG GIAO DỊCH")
        print("   → Phản ánh mức độ hoạt động của thị trường")

    # ==========================================================
    # B4.7 - LƯU DỮ LIỆU SAU PCA
    # ==========================================================
    print("\n" + "=" * 60)
    print("B4.7 - LƯU DỮ LIỆU")
    print("=" * 60)

    # Tạo DataFrame với PCA components
    df_pca = df.copy()
    df_pca["PC1"] = X_pca[:, 0]
    df_pca["PC2"] = X_pca[:, 1]

    # Lưu dữ liệu sau feature selection
    df_selected = df[["Date", "Close/Last", "Volume"]].copy()
    df_selected.to_csv("goldstock_selected_features_B4.csv", index=False)
    print("\n✅ Dữ liệu sau feature selection đã lưu: goldstock_selected_features_B4.csv")

    # Lưu dữ liệu sau PCA
    df_pca[["Date", "PC1", "PC2"]].to_csv("goldstock_pca_B4.csv", index=False)
    print("✅ Dữ liệu sau PCA đã lưu: goldstock_pca_B4.csv")

    print("\n📊 Sample data sau feature selection:")
    print(df_selected.head(10))

    print("\n" + "=" * 60)
    print("KẾT THÚC B4 - MA TRẬN TƯƠNG QUAN & GIẢM CHIỀU")
    print("=" * 60)

if __name__ == "__main__":
    run()
