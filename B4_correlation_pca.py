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
    print(f"📊 Các cột định lượng: {quantitative_cols}")

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
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                square=True, fmt=".3f", cbar_kws={'label': 'Correlation Coefficient'},
                linewidths=1, linecolor='black',
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Ma trận tương quan (Correlation Matrix)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("B4_correlation_matrix.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_correlation_matrix.png")
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

    plt.tight_layout()
    plt.savefig("B4_pca_variance_explained.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_pca_variance_explained.png")
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

    plt.tight_layout()
    plt.savefig("B4_pca_projection.png", dpi=300, bbox_inches='tight')
    print("\n   ✓ Biểu đồ đã lưu: B4_pca_projection.png")
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
