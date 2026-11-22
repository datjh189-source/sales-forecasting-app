import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURE_COLS, build_single_step_features, create_supervised_dataset

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính các chỉ số sai số MAE, RMSE và MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def recursive_forecast(model, full_monthly_df: pd.DataFrame, steps: int) -> pd.DataFrame:
    """Dự báo đệ quy nhiều tháng tiếp theo trong tương lai."""
    df = full_monthly_df.sort_values("YearMonth").reset_index(drop=True)
    sales_history = df["Sales"].tolist()
    last_period = pd.Period(df["YearMonth"].iloc[-1], freq="M")
    start_idx = len(df)

    future_rows = []
    for step in range(1, steps + 1):
        next_period = last_period + step
        feat_dict = build_single_step_features(sales_history, next_period, time_idx=start_idx + step - 1)
        X_next = pd.DataFrame([feat_dict])[FEATURE_COLS]
        pred_sales = max(0.0, float(model.predict(X_next)[0]))

        sales_history.append(pred_sales)
        future_rows.append({"YearMonth": str(next_period), "Predicted_Sales": pred_sales})

    return pd.DataFrame(future_rows)


def train_and_evaluate():
    monthly_path = os.path.join("data", "processed", "monthly_sales.csv")
    out_dir = "outputs"
    model_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    monthly_df = pd.read_csv(monthly_path)
    supervised_df = create_supervised_dataset(monthly_df)

    # Chia tập Train / Test theo trình tự thời gian:
    # 12 tháng cuối cùng (năm 2018) làm tập Test, các tháng trước đó làm tập Train
    test_size = 12
    train_df = supervised_df.iloc[:-test_size].copy()
    test_df = supervised_df.iloc[-test_size:].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df["Sales"].values
    X_test, y_test = test_df[FEATURE_COLS], test_df["Sales"].values

    print("=" * 65)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH DỰ BÁO DOANH THU")
    print("=" * 65)
    print(f"- Tập huấn luyện (Train): {len(train_df)} tháng ({train_df['YearMonth'].iloc[0]} -> {train_df['YearMonth'].iloc[-1]})")
    print(f"- Tập kiểm thử (Test):    {len(test_df)} tháng ({test_df['YearMonth'].iloc[0]} -> {test_df['YearMonth'].iloc[-1]})")

    results = []
    predictions_dict = {}

    # 1. Mô hình đường cơ sở (Baseline: Lấy doanh thu cùng tháng năm trước - lag_12)
    baseline_pred = test_df["lag_12"].values
    base_metrics = evaluate_metrics(y_test, baseline_pred)
    base_metrics["Mô hình"] = "Baseline (Cùng kỳ năm trước)"
    results.append(base_metrics)
    predictions_dict["Baseline (Cùng kỳ)"] = baseline_pred

    # 2. Các mô hình học máy
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=3, random_state=42),
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        # Dự báo đệ quy trên 12 tháng tập Test để phản ánh đúng bài toán dự báo tương lai
        history_until_train = monthly_df.iloc[:-test_size].copy()
        rec_df = recursive_forecast(model, history_until_train, steps=test_size)
        y_pred = rec_df["Predicted_Sales"].values

        metrics = evaluate_metrics(y_test, y_pred)
        metrics["Mô hình"] = name
        results.append(metrics)
        predictions_dict[name] = y_pred
        trained_models[name] = model

    # Tổng hợp bảng so sánh
    results_df = pd.DataFrame(results)[["Mô hình", "MAE", "RMSE", "MAPE (%)"]].sort_values("MAPE (%)")
    print("\n--- BẢNG SO SÁNH CHỈ SỐ SAI SỐ TRÊN TẬP KIỂM THỬ (2018) ---")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    results_df.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False, encoding="utf-8-sig")

    # Chọn mô hình tốt nhất (có MAPE thấp nhất trong nhóm mô hình học máy) và huấn luyện lại trên toàn bộ dữ liệu
    ml_results = results_df[results_df["Mô hình"] != "Baseline (Cùng kỳ năm trước)"]
    best_model_name = ml_results.iloc[0]["Mô hình"]
    best_model = trained_models[best_model_name]

    # Huấn luyện lại mô hình tốt nhất trên 100% dữ liệu để sẵn sàng dự báo tương lai trên Streamlit
    X_all, y_all = supervised_df[FEATURE_COLS], supervised_df["Sales"].values
    best_model.fit(X_all, y_all)

    artifact = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics_table": results_df,
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(artifact, os.path.join(model_dir, "best_model.joblib"))

    # Vẽ biểu đồ so sánh đường Thực tế vs Dự báo trên tập Test (năm 2018)
    plt.figure(figsize=(13, 6))
    plt.plot(test_df["YearMonth"], y_test, marker="o", linewidth=2.5, color="black", label="Doanh thu Thực tế (2018)")

    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    styles = ["--", "-.", "-", "-"]
    for (m_name, y_pred), c, s in zip(predictions_dict.items(), colors, styles):
        plt.plot(test_df["YearMonth"], y_pred, marker="s", linestyle=s, linewidth=1.8, color=c, label=m_name)

    plt.title("So sánh Doanh thu Thực tế và Dự báo của các Mô hình (Năm 2018)", fontsize=14, fontweight="bold")
    plt.xlabel("Thời gian (Năm - Tháng)", fontsize=11)
    plt.ylabel("Doanh thu (USD)", fontsize=11)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_so_sanh_du_bao_mo_hinh.png"), dpi=300)
    plt.close()

    print(f"\n-> Mô hình học máy tối ưu nhất: {best_model_name}")
    print("-> Đã lưu mô hình vào: models/best_model.joblib")
    print("-> Đã lưu bảng so sánh vào: outputs/model_comparison.csv")
    print("-> Đã lưu biểu đồ đối chiếu vào: outputs/06_so_sanh_du_bao_mo_hinh.png")


if __name__ == "__main__":
    train_and_evaluate()