import os
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV hoặc Excel, tự động xử lý lỗi bảng mã."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")

    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin1")
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Định dạng file không hỗ trợ. Vui lòng dùng .csv hoặc .xlsx")
    return df


def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch, chuẩn hóa kiểu dữ liệu và trích xuất đặc trưng thời gian."""
    print(f"Kích thước dữ liệu ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột")

    # 1. Kiểm tra và loại bỏ dòng trùng lặp
    dup_count = df.duplicated().sum()
    print(f"Số dòng trùng lặp: {dup_count}")
    if dup_count > 0:
        df = df.drop_duplicates()

    # 2. Kiểm tra giá trị khuyết thiếu (Missing values)
    missing = df.isnull().sum()
    print("\nSố lượng giá trị khuyết thiếu theo cột:")
    print(missing[missing > 0] if missing.sum() > 0 else "Không có giá trị khuyết thiếu.")

    # Loại bỏ các dòng thiếu thông tin cốt lõi (Ngày đặt hàng hoặc Doanh thu)
    df = df.dropna(subset=["Order Date", "Sales"]).copy()

    # 3. Chuẩn hóa kiểu dữ liệu ngày tháng (định dạng DD/MM/YYYY của Superstore)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", dayfirst=True)
    df = df.sort_values("Order Date").reset_index(drop=True)

    # 4. Đảm bảo doanh thu là số dương hợp lệ
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df[df["Sales"] > 0].copy()

    # 5. Trích xuất các thuộc tính thời gian phục vụ phân tích
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    df["DayOfWeek"] = df["Order Date"].dt.dayofweek  # 0: Thứ 2, 6: Chủ nhật
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

    print(f"\nKích thước dữ liệu sau làm sạch: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"Khoảng thời gian dữ liệu: từ {df['Order Date'].min().date()} đến {df['Order Date'].max().date()}")
    return df


def aggregate_sales(df: pd.DataFrame):
    """Tổng hợp doanh thu theo Ngày và theo Tháng."""
    # Tổng hợp theo Ngày (lấp đầy 0 cho các ngày không phát sinh giao dịch để liền mạch chuỗi thời gian)
    daily = df.groupby("Order Date")["Sales"].sum().reset_index()
    full_date_range = pd.date_range(start=daily["Order Date"].min(), end=daily["Order Date"].max(), freq="D")
    daily = daily.set_index("Order Date").reindex(full_date_range, fill_value=0).reset_index()
    daily.columns = ["Order Date", "Sales"]

    # Tổng hợp theo Tháng
    monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()

    return daily, monthly


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "superstore.csv")
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df_raw = load_data(raw_path)
    df_clean = clean_and_prepare_data(df_raw)
    daily_sales, monthly_sales = aggregate_sales(df_clean)

    # Lưu các file dữ liệu sạch
    df_clean.to_csv(os.path.join(processed_dir, "clean_transactions.csv"), index=False, encoding="utf-8-sig")
    daily_sales.to_csv(os.path.join(processed_dir, "daily_sales.csv"), index=False)
    monthly_sales.to_csv(os.path.join(processed_dir, "monthly_sales.csv"), index=False)

    print("\nĐã lưu thành công 3 file vào thư mục data/processed/:")
    print("- clean_transactions.csv (Dữ liệu giao dịch sạch)")
    print("- daily_sales.csv (Doanh thu tổng hợp theo ngày)")
    print("- monthly_sales.csv (Doanh thu tổng hợp theo tháng)")