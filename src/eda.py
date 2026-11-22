import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện biểu đồ chuyên nghiệp, hỗ trợ tiếng Việt
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def run_eda():
    clean_path = os.path.join("data", "processed", "clean_transactions.csv")
    monthly_path = os.path.join("data", "processed", "monthly_sales.csv")
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(clean_path)
    monthly = pd.read_csv(monthly_path)

    # 1. IN BẢNG THỐNG KÊ TỔNG QUAN RA MÀN HÌNH (Dùng viết nhận xét Chương 3)
    print("=" * 60)
    print("BÁO CÁO THỐNG KÊ KHÁM PHÁ DỮ LIỆU (EDA)")
    print("=" * 60)
    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique()
    avg_order_value = total_sales / total_orders

    print(f"- Tổng doanh thu (2015 - 2018): ${total_sales:,.2f}")
    print(f"- Tổng số đơn hàng duy nhất:    {total_orders:,} đơn")
    print(f"- Giá trị trung bình mỗi đơn:   ${avg_order_value:,.2f}")

    print("\n--- Doanh thu theo Năm ---")
    yearly = df.groupby("Year")["Sales"].sum()
    for year, val in yearly.items():
        print(f"  + Năm {year}: ${val:,.2f}")

    print("\n--- Doanh thu theo Danh mục (Category) ---")
    cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    for cat, val in cat_sales.items():
        pct = val / total_sales * 100
        print(f"  + {cat}: ${val:,.2f} ({pct:.1f}%)")

    print("\n--- Doanh thu theo Khu vực (Region) ---")
    reg_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    for reg, val in reg_sales.items():
        pct = val / total_sales * 100
        print(f"  + {reg}: ${val:,.2f} ({pct:.1f}%)")

    # 2. VẼ VÀ LƯU 5 BIỂU ĐỒ VÀO THƯ MỤC outputs/
    # Biểu đồ 1: Xu hướng doanh thu theo tháng
    plt.figure(figsize=(14, 5))
    plt.plot(monthly["YearMonth"], monthly["Sales"], marker="o", color="#1f77b4", linewidth=2)
    plt.title("Xu hướng Doanh thu Bán hàng theo Tháng (2015 - 2018)", fontsize=14, fontweight="bold")
    plt.xlabel("Thời gian (Năm - Tháng)", fontsize=11)
    plt.ylabel("Doanh thu (USD)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_doanh_thu_theo_thang.png"), dpi=300)
    plt.close()

    # Biểu đồ 2: Doanh thu theo Danh mục và Danh mục con
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(x=cat_sales.index, y=cat_sales.values, ax=axes[0], palette="Blues_d", hue=cat_sales.index, legend=False)
    axes[0].set_title("Doanh thu theo Danh mục chính", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Danh mục")
    axes[0].set_ylabel("Doanh thu (USD)")

    subcat_sales = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False)
    sns.barplot(x=subcat_sales.values, y=subcat_sales.index, ax=axes[1], palette="viridis", hue=subcat_sales.index, legend=False)
    axes[1].set_title("Doanh thu theo Danh mục con (Sub-Category)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Doanh thu (USD)")
    axes[1].set_ylabel("Danh mục con")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_doanh_thu_theo_danh_muc.png"), dpi=300)
    plt.close()

    # Biểu đồ 3: Doanh thu theo Khu vực
    plt.figure(figsize=(8, 5))
    sns.barplot(x=reg_sales.index, y=reg_sales.values, palette="Set2", hue=reg_sales.index, legend=False)
    plt.title("So sánh Doanh thu theo Khu vực Địa lý", fontsize=13, fontweight="bold")
    plt.xlabel("Khu vực")
    plt.ylabel("Doanh thu (USD)")
    for i, val in enumerate(reg_sales.values):
        plt.text(i, val + 5000, f"${val:,.0f}", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_doanh_thu_theo_khu_vuc.png"), dpi=300)
    plt.close()

    # Biểu đồ 4: Top 10 Sản phẩm có doanh thu cao nhất
    top10_prod = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
    short_names = [name[:45] + "..." if len(name) > 45 else name for name in top10_prod.index]
    plt.figure(figsize=(11, 6))
    sns.barplot(x=top10_prod.values, y=short_names, palette="magma", hue=short_names, legend=False)
    plt.title("Top 10 Sản phẩm mang lại Doanh thu cao nhất", fontsize=13, fontweight="bold")
    plt.xlabel("Doanh thu (USD)")
    plt.ylabel("Tên sản phẩm")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_top_10_san_pham.png"), dpi=300)
    plt.close()

    # Biểu đồ 5: Phân tích tính mùa vụ theo Tháng trong năm
    month_season = df.groupby("Month")["Sales"].sum()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=month_season.index, y=month_season.values, color="#2ca02c")
    plt.title("Tổng hợp Doanh thu theo 12 Tháng (Phân tích tính mùa vụ)", fontsize=13, fontweight="bold")
    plt.xlabel("Tháng trong năm")
    plt.ylabel("Tổng doanh thu cộng dồn (USD)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_mua_vu_theo_thang.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("ĐÃ XUẤT THÀNH CÔNG 5 BIỂU ĐỒ VÀO THƯ MỤC outputs/!")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()