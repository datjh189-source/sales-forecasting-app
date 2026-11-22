import numpy as np
import pandas as pd

FEATURE_COLS = [
    "time_index",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3",
]


def build_single_step_features(sales_history: list, period_obj: pd.Period, time_idx: int) -> dict:
    """Tạo vector đặc trưng cho 1 mốc thời gian từ lịch sử doanh thu trước đó."""
    if len(sales_history) < 12:
        raise ValueError("Cần tối thiểu 12 tháng dữ liệu lịch sử để tạo đặc trưng mùa vụ (lag_12).")

    month = period_obj.month
    quarter = period_obj.quarter

    return {
        "time_index": time_idx,
        "month": month,
        "quarter": quarter,
        "month_sin": np.sin(2 * np.pi * month / 12.0),
        "month_cos": np.cos(2 * np.pi * month / 12.0),
        "lag_1": float(sales_history[-1]),
        "lag_2": float(sales_history[-2]),
        "lag_3": float(sales_history[-3]),
        "lag_12": float(sales_history[-12]),
        "rolling_mean_3": float(np.mean(sales_history[-3:])),
    }


def create_supervised_dataset(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Chuyển đổi bảng doanh thu theo tháng thành tập dữ liệu có giám sát (X, y)."""
    df = monthly_df.sort_values("YearMonth").reset_index(drop=True)
    periods = pd.PeriodIndex(df["YearMonth"], freq="M")
    sales_values = df["Sales"].tolist()

    rows = []
    for i in range(12, len(df)):
        feat = build_single_step_features(sales_values[:i], periods[i], time_idx=i)
        feat["YearMonth"] = str(periods[i])
        feat["Sales"] = float(sales_values[i])
        rows.append(feat)

    return pd.DataFrame(rows)