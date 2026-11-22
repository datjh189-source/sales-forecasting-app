import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocessing import clean_and_prepare_data, aggregate_sales
from features import FEATURE_COLS, create_supervised_dataset
from forecaster import recursive_forecast

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Analytics • Hệ thống Phân tích & Dự đoán Doanh thu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==================== BẢNG MÀU CHUẨN SAAS MOCKUP ====================
INDIGO = "#635BFF"
INDIGO_SOFT = "#A5B4FC"
TEAL = "#14B8A6"
SKY = "#38BDF8"
AMBER = "#FBBF24"
ROSE = "#F43F5E"


def is_dark_theme() -> bool:
    """Nhận diện tự động chế độ Sáng / Tối hiện tại của Streamlit."""
    try:
        theme_type = getattr(getattr(st.context, "theme", None), "type", None)
        return str(theme_type).lower() == "dark"
    except Exception:
        return False


# ==================== CUSTOM CSS (THÍCH ỨNG LIGHT & DARK) ====================
st.markdown(
    """
<style>
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1380px;
}

/* Thẻ thương hiệu Navy-Indigo ở đầu Sidebar giống ảnh mẫu */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    background: linear-gradient(135deg, #16182B 0%, #1E223D 100%);
    border: 1px solid rgba(99, 91, 255, 0.35);
    border-radius: 14px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
}
.brand-logo {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #635BFF;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99, 91, 255, 0.45);
}
.brand-title {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
}
.brand-sub {
    font-size: 12px;
    color: #CBD5E1;
    margin-top: 2px;
}

/* Tiêu đề trang (tự đổi màu theo theme) */
.page-title {
    font-size: 24px;
    font-weight: 800;
    color: var(--text-color);
    margin: 0;
    letter-spacing: -0.3px;
}
.page-subtitle {
    font-size: 13.5px;
    color: var(--text-color);
    opacity: 0.72;
    margin: 3px 0 0 0;
}

/* Thanh Tabs hiện đại với Material Icons */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: var(--secondary-background-color);
    padding: 5px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 9px;
    padding: 0 18px;
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-color) !important;
    opacity: 0.78;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    opacity: 1;
}
.stTabs [aria-selected="true"] {
    background-color: #635BFF !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
    box-shadow: 0 2px 8px rgba(99, 91, 255, 0.32);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Thẻ KPI có Sparkline */
.saas-kpi-card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.16s ease, box-shadow 0.16s ease;
    margin-bottom: 12px;
}
.saas-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99, 91, 255, 0.12);
    border-color: rgba(99, 91, 255, 0.45);
}
.saas-kpi-head {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.78;
    margin-bottom: 10px;
}
.saas-kpi-icon {
    width: 26px;
    height: 26px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 13.5px;
}
.saas-kpi-val {
    font-size: 26px;
    font-weight: 800;
    color: var(--text-color);
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 12px;
}
.saas-kpi-bottom {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.delta-pill-pos {
    display: inline-block;
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    font-size: 11.5px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
}
.delta-pill-neg {
    display: inline-block;
    background-color: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    font-size: 11.5px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 999px;
}
.delta-note {
    font-size: 11.5px;
    color: var(--text-color);
    opacity: 0.62;
    margin-left: 5px;
}

/* Tiêu đề từng thẻ biểu đồ */
.card-title {
    font-size: 15.5px;
    font-weight: 700;
    color: var(--text-color);
    margin-bottom: 2px;
}
.card-sub {
    font-size: 12.5px;
    color: var(--text-color);
    opacity: 0.68;
    margin-bottom: 10px;
}

/* Nút bấm chính màu tím Indigo */
.stDownloadButton button {
    background-color: #635BFF !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.45rem 1rem !important;
}
.stDownloadButton button:hover {
    background-color: #4F46E5 !important;
}

/* Thanh tiến độ (Hiệu suất khu vực) */
.prog-row {
    margin-bottom: 15px;
}
.prog-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 13px;
    margin-bottom: 6px;
}
.prog-name {
    font-weight: 600;
    color: var(--text-color);
}
.prog-stats {
    font-size: 12px;
    color: var(--text-color);
    opacity: 0.68;
}
.prog-pct {
    font-weight: 700;
    color: var(--text-color);
    margin-left: 8px;
}
.prog-track {
    width: 100%;
    height: 8px;
    background-color: rgba(128, 128, 128, 0.18);
    border-radius: 999px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 999px;
}

/* Danh sách Top nguồn doanh thu */
.source-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.16);
}
.source-row:last-child {
    border-bottom: none;
}
.source-left {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
}
.source-badge {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    flex-shrink: 0;
}
.source-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-color);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 175px;
}
.source-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
}
.source-val {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-color);
}
.source-share {
    font-size: 12px;
    font-weight: 700;
    color: #10B981;
}
</style>
""",
    unsafe_allow_html=True,
)


def make_sparkline_svg(values: list, stroke_color: str) -> str:
    if not values or len(values) < 2:
        values = [10, 14, 12, 18, 16, 22, 25]
    arr = np.array(values, dtype=float)
    min_v, max_v = arr.min(), arr.max()
    if max_v - min_v < 1e-9:
        norm = np.full_like(arr, 0.5)
    else:
        norm = (arr - min_v) / (max_v - min_v)

    w, h = 76, 26
    xs = np.linspace(2, w - 2, len(norm))
    ys = (h - 4) - norm * (h - 8) + 2
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline fill="none" stroke="{stroke_color}" stroke-width="2.2" '
        f'stroke-linecap="round" stroke-linejoin="round" points="{pts}"/>'
        f'</svg>'
    )


def render_saas_kpi(title: str, value: str, delta_text: str, note: str, icon: str, icon_bg: str, spark_vals: list, spark_color: str, is_positive: bool = True):
    pill_cls = "delta-pill-pos" if is_positive else "delta-pill-neg"
    svg_html = make_sparkline_svg(spark_vals, spark_color)
    html = (
        f'<div class="saas-kpi-card">'
        f'<div class="saas-kpi-head">'
        f'<span class="saas-kpi-icon" style="background:{icon_bg};">{icon}</span>'
        f'<span>{title}</span>'
        f'</div>'
        f'<div class="saas-kpi-val">{value}</div>'
        f'<div class="saas-kpi-bottom">'
        f'<div><span class="{pill_cls}">{delta_text}</span><span class="delta-note">{note}</span></div>'
        f'<div>{svg_html}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def smooth_series(y_vals: np.ndarray, points_per_seg: int = 12):
    n = len(y_vals)
    if n < 3:
        return np.arange(n, dtype=float), y_vals.astype(float)
    x_smooth, y_smooth = [], []
    for i in range(n - 1):
        p0 = y_vals[max(0, i - 1)]
        p1 = y_vals[i]
        p2 = y_vals[i + 1]
        p3 = y_vals[min(n - 1, i + 2)]
        t = np.linspace(0, 1, points_per_seg, endpoint=(i == n - 2))
        t2 = t * t
        t3 = t2 * t
        y_seg = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
        x_seg = i + t
        x_smooth.append(x_seg)
        y_smooth.append(y_seg)
    return np.concatenate(x_smooth), np.concatenate(y_smooth)


@st.cache_data
def load_default_data():
    raw_path = os.path.join("data", "raw", "superstore.csv")
    try:
        df = pd.read_csv(raw_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(raw_path, encoding="latin1")
    return df


@st.cache_resource
def load_saved_model():
    model_path = os.path.join("models", "best_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")
    return pd.read_excel(uploaded_file)


# ==================== SIDEBAR ĐIỀU KHIỂN ====================
with st.sidebar:
    st.markdown(
        """
<div class="brand-header">
<div class="brand-logo">❖</div>
<div>
<div class="brand-title">Analytics Studio</div>
<div class="brand-sub">SVTH: Ngô Văn Đạt • ĐHSP ĐN</div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("**1. Nguồn dữ liệu phân tích**")
    data_source = st.radio(
        "Chọn nguồn dữ liệu:",
        ("Dữ liệu chuẩn (Superstore)", "Tải lên file CSV / Excel"),
        label_visibility="collapsed",
    )

    raw_df = None
    if data_source == "Dữ liệu chuẩn (Superstore)":
        raw_df = load_default_data()
    else:
        uploaded = st.file_uploader(
            "Tải file bán hàng (.csv, .xlsx)",
            type=["csv", "xlsx", "xls"],
            help="Hỗ trợ cả cột tiếng Việt ('Ngày đặt hàng', 'Doanh thu') và tiếng Anh ('Order Date', 'Sales').",
        )
        sample_df = load_default_data().head(50)
        csv_sample = sample_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải file CSV mẫu",
            data=csv_sample,
            file_name="mau_du_lieu_ban_hang.csv",
            mime="text/csv",
            width="stretch",
        )
        if uploaded is not None:
            raw_df = read_uploaded_file(uploaded)

    st.divider()
    st.markdown("**2. Cấu hình Dự báo AI**")
    forecast_months = st.slider(
        "Số tháng dự báo tiếp theo:",
        min_value=1,
        max_value=12,
        value=6,
        step=1,
    )

    retrain_on_upload = False
    if data_source == "Tải lên file CSV / Excel":
        retrain_on_upload = st.checkbox(
            "Huấn luyện lại mô hình trên file mới",
            value=False,
            help="Dùng khi file tải lên có từ 18 tháng dữ liệu trở lên.",
        )

    st.divider()
    st.caption("GVHD: TS. Phạm Anh Phương • Khoa Toán - Tin")

if raw_df is None:
    st.markdown('<h2 class="page-title">Tổng quan Kinh doanh (Business overview)</h2>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Hệ thống Phân tích và Dự đoán Doanh thu Bán hàng</p>', unsafe_allow_html=True)
    st.write("")
    st.info("👈 Vui lòng bấm **Browse files** ở thanh bên trái để tải file dữ liệu lên, hoặc chọn **Dữ liệu chuẩn (Superstore)**.")
    st.stop()

# Chuẩn hóa tên cột tiếng Việt -> chuẩn hệ thống
raw_df.columns = [str(c).strip() for c in raw_df.columns]
vn_to_en_cols = {
    "Mã đơn hàng": "Order ID",
    "Ngày đặt hàng": "Order Date",
    "Danh mục": "Category",
    "Danh mục con": "Sub-Category",
    "Tên sản phẩm": "Product Name",
    "Khu vực": "Region",
    "Số lượng": "Quantity",
    "Doanh thu": "Sales",
}
raw_df = raw_df.rename(columns={k: v for k, v in vn_to_en_cols.items() if k in raw_df.columns})

if not {"Order Date", "Sales"}.issubset(set(raw_df.columns)):
    st.error(f"❌ File thiếu cột bắt buộc (`Ngày đặt hàng`/`Order Date` và `Doanh thu`/`Sales`). Các cột hiện có: {list(raw_df.columns)}")
    st.stop()

df_clean = clean_and_prepare_data(raw_df.copy())
if df_clean.empty:
    st.warning("⚠️ File chưa có dòng dữ liệu hợp lệ (Ngày đặt hàng và Doanh thu đang trống hoặc <= 0).")
    st.stop()

daily_sales, monthly_sales = aggregate_sales(df_clean)

# ==================== TOP HEADER BAR ====================
years_avail = sorted(df_clean["Year"].unique().tolist())
latest_year = years_avail[-1]
prev_year = years_avail[-2] if len(years_avail) >= 2 else latest_year

top_left, top_mid, top_right = st.columns([2.4, 1.1, 0.85])
with top_left:
    st.markdown('<h1 class="page-title">Tổng quan Kinh doanh (Business overview)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Phân tích doanh thu, đơn hàng, cơ cấu sản phẩm và dự báo xu hướng bằng Học máy</p>', unsafe_allow_html=True)

with top_mid:
    year_options = ["Tất cả các năm"] + [str(y) for y in reversed(years_avail)]
    selected_year_opt = st.selectbox("Kỳ phân tích:", year_options, index=0, label_visibility="collapsed")

with top_right:
    export_csv = df_clean.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇ Export báo cáo",
        data=export_csv,
        file_name="bao_cao_doanh_thu_sach.csv",
        mime="text/csv",
        width="stretch",
    )

st.write("")

if selected_year_opt == "Tất cả các năm":
    view_df = df_clean.copy()
    compare_label = f"năm {latest_year} vs {prev_year}"
else:
    sel_y = int(selected_year_opt)
    view_df = df_clean[df_clean["Year"] == sel_y].copy()
    prev_y = sel_y - 1 if (sel_y - 1) in years_avail else sel_y
    compare_label = f"so với năm {prev_y}"

# Bảng màu biểu đồ động theo chế độ Sáng / Tối
dark_mode = is_dark_theme()
chart_text_color = "#F1F5F9" if dark_mode else "#0F172A"
chart_muted_color = "#94A3B8" if dark_mode else "#64748B"
chart_grid_color = "#1E293B" if dark_mode else "#F1F5F9"
chart_prev_line = "#475569" if dark_mode else "#CBD5E1"
donut_edge_color = "#1E293B" if dark_mode else "#FFFFFF"

# ==================== 3 TABS VỚI MATERIAL VECTOR ICONS ====================
tab_dash, tab_forecast, tab_data = st.tabs([
    ":material/space_dashboard: Tổng quan & Phân tích",
    ":material/auto_graph: Dự đoán Doanh thu Tương lai",
    ":material/table_view: Dữ liệu & Tiền xử lý",
])

# ==============================================================================
# TAB 1: DASHBOARD CHÍNH
# ==============================================================================
with tab_dash:
    df_latest_y = df_clean[df_clean["Year"] == (latest_year if selected_year_opt == "Tất cả các năm" else int(selected_year_opt))]
    ref_year = prev_year if selected_year_opt == "Tất cả các năm" else (int(selected_year_opt) - 1 if (int(selected_year_opt) - 1) in years_avail else int(selected_year_opt))
    df_prev_y = df_clean[df_clean["Year"] == ref_year]

    def calc_growth(curr_val, prev_val):
        if prev_val <= 0:
            return "+0.0%", True
        pct = (curr_val - prev_val) / prev_val * 100.0
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%", pct >= 0

    tot_rev = view_df["Sales"].sum()
    rev_growth, rev_pos = calc_growth(df_latest_y["Sales"].sum(), df_prev_y["Sales"].sum())

    tot_orders = view_df["Order ID"].nunique() if "Order ID" in view_df.columns else len(view_df)
    ord_curr = df_latest_y["Order ID"].nunique() if "Order ID" in df_latest_y.columns else len(df_latest_y)
    ord_prev = df_prev_y["Order ID"].nunique() if "Order ID" in df_prev_y.columns else len(df_prev_y)
    ord_growth, ord_pos = calc_growth(ord_curr, ord_prev)

    tot_qty = int(view_df["Quantity"].sum()) if "Quantity" in view_df.columns else len(view_df)
    qty_curr = df_latest_y["Quantity"].sum() if "Quantity" in df_latest_y.columns else len(df_latest_y)
    qty_prev = df_prev_y["Quantity"].sum() if "Quantity" in df_prev_y.columns else len(df_prev_y)
    qty_growth, qty_pos = calc_growth(qty_curr, qty_prev)

    aov = tot_rev / max(tot_orders, 1)
    aov_curr = df_latest_y["Sales"].sum() / max(ord_curr, 1)
    aov_prev = df_prev_y["Sales"].sum() / max(ord_prev, 1)
    aov_growth, aov_pos = calc_growth(aov_curr, aov_prev)

    m_grp = view_df.groupby("YearMonth").agg(
        sales=("Sales", "sum"),
        orders=("Order ID", "nunique") if "Order ID" in view_df.columns else ("Sales", "count"),
        qty=("Quantity", "sum") if "Quantity" in view_df.columns else ("Sales", "count"),
    ).reset_index()
    m_grp["aov"] = m_grp["sales"] / np.maximum(m_grp["orders"], 1)

    # ---------- HÀNG 1: 4 THẺ KPI SPARKLINE ----------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_saas_kpi(
            "Tổng Doanh thu",
            f"${tot_rev:,.0f}",
            rev_growth,
            compare_label,
            "💳",
            "rgba(99, 91, 255, 0.16)",
            m_grp["sales"].tail(10).tolist(),
            INDIGO,
            rev_pos,
        )
    with k2:
        render_saas_kpi(
            "Đơn hàng hoàn tất",
            f"{tot_orders:,}",
            ord_growth,
            compare_label,
            "🧾",
            "rgba(20, 184, 166, 0.16)",
            m_grp["orders"].tail(10).tolist(),
            TEAL,
            ord_pos,
        )
    with k3:
        render_saas_kpi(
            "Sản lượng bán ra",
            f"{tot_qty:,}",
            qty_growth,
            compare_label,
            "📦",
            "rgba(245, 158, 11, 0.16)",
            m_grp["qty"].tail(10).tolist(),
            AMBER,
            qty_pos,
        )
    with k4:
        render_saas_kpi(
            "Giá trị TB / Đơn",
            f"${aov:,.2f}",
            aov_growth,
            compare_label,
            "📈",
            "rgba(16, 185, 129, 0.16)",
            m_grp["aov"].tail(10).tolist(),
            "#10B981",
            aov_pos,
        )

    # ---------- HÀNG 2: REVENUE TREND (TRÁI) + DONUT CHART (PHẢI) ----------
    col_trend, col_donut = st.columns([1.68, 1])

    with col_trend:
        with st.container(border=True):
            st.markdown('<div class="card-title">Xu hướng Doanh thu (Revenue trend)</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">So sánh biến động doanh thu 12 tháng năm gần nhất với cùng kỳ năm trước</div>', unsafe_allow_html=True)

            target_y = latest_year if selected_year_opt == "Tất cả các năm" else int(selected_year_opt)
            comp_y = target_y - 1 if (target_y - 1) in years_avail else None

            months_12 = list(range(1, 13))
            month_labels = [f"T{m}" for m in months_12]

            curr_m = df_clean[df_clean["Year"] == target_y].groupby("Month")["Sales"].sum().reindex(months_12, fill_value=0).values
            fig_tr, ax_tr = plt.subplots(figsize=(8.6, 3.35))
            fig_tr.patch.set_alpha(0.0)
            ax_tr.patch.set_alpha(0.0)

            if comp_y is not None:
                prev_m = df_clean[df_clean["Year"] == comp_y].groupby("Month")["Sales"].sum().reindex(months_12, fill_value=0).values
                xp, yp = smooth_series(prev_m)
                ax_tr.plot(xp, yp, color=chart_prev_line, linewidth=2.0, linestyle="-", label=f"Năm {comp_y}")

            xc, yc = smooth_series(curr_m)
            ax_tr.plot(xc, yc, color=INDIGO, linewidth=2.6, label=f"Năm {target_y}")
            ax_tr.fill_between(xc, yc, color=INDIGO, alpha=0.14)

            last_x = len(months_12) - 1
            last_y = curr_m[-1]
            ax_tr.scatter([last_x], [last_y], color=INDIGO, s=44, zorder=5, edgecolor="#FFFFFF", linewidth=1.5)
            ax_tr.annotate(
                f"T12 • ${last_y/1000:,.1f}k",
                xy=(last_x, last_y),
                xytext=(last_x - 1.45, last_y + max(curr_m) * 0.12),
                fontsize=8.5,
                fontweight="bold",
                color="#FFFFFF",
                bbox=dict(boxstyle="round,pad=0.35", fc=INDIGO, ec="none"),
            )

            ax_tr.set_xticks(range(12))
            ax_tr.set_xticklabels(month_labels, fontsize=9, color=chart_muted_color)
            ax_tr.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
            ax_tr.tick_params(axis="y", labelsize=8.5, colors=chart_muted_color)
            ax_tr.set_ylim(0, max(curr_m.max(), 1) * 1.3)
            ax_tr.grid(axis="x", visible=False)
            ax_tr.grid(axis="y", linestyle="-", color=chart_grid_color, linewidth=1)
            sns.despine(ax=ax_tr, left=True, bottom=False)
            ax_tr.spines["bottom"].set_color(chart_grid_color)
            leg = ax_tr.legend(loc="upper left", frameon=False, fontsize=9)
            for txt in leg.get_texts():
                txt.set_color(chart_text_color)
            plt.tight_layout()
            st.pyplot(fig_tr)
            plt.close(fig_tr)

    with col_donut:
        with st.container(border=True):
            st.markdown('<div class="card-title">Cơ cấu Doanh thu (Revenue by category)</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Tỷ trọng đóng góp theo từng danh mục sản phẩm</div>', unsafe_allow_html=True)

            if "Category" in view_df.columns:
                cat_series = view_df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            else:
                cat_series = pd.Series({"Tổng doanh thu": tot_rev})

            donut_colors = [INDIGO, TEAL, SKY, AMBER, "#A855F7"][: len(cat_series)]

            d_left, d_right = st.columns([1.05, 1])
            with d_left:
                fig_dn, ax_dn = plt.subplots(figsize=(3.4, 3.35))
                fig_dn.patch.set_alpha(0.0)
                ax_dn.patch.set_alpha(0.0)
                ax_dn.pie(
                    cat_series.values,
                    colors=donut_colors,
                    startangle=90,
                    counterclock=False,
                    wedgeprops=dict(width=0.33, edgecolor=donut_edge_color, linewidth=2.2),
                )
                ax_dn.text(0, 0.08, f"${tot_rev/1000:,.0f}k", ha="center", va="center", fontsize=15, fontweight="bold", color=chart_text_color)
                ax_dn.text(0, -0.18, "Doanh thu", ha="center", va="center", fontsize=9, color=chart_muted_color)
                ax_dn.axis("equal")
                plt.tight_layout()
                st.pyplot(fig_dn)
                plt.close(fig_dn)

            with d_right:
                st.write("")
                for (cat_name, val), col_hex in zip(cat_series.items(), donut_colors):
                    pct = val / max(tot_rev, 1) * 100
                    st.markdown(
                        f'<div style="display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid rgba(128,128,128,0.16);">'
                        f'<div style="display:flex; align-items:center; gap:8px;">'
                        f'<span style="width:9px; height:9px; border-radius:3px; background:{col_hex}; display:inline-block;"></span>'
                        f'<span style="font-size:13px; color:var(--text-color); opacity:0.88; font-weight:500;">{cat_name}</span>'
                        f'</div>'
                        f'<span style="font-size:13px; font-weight:700; color:var(--text-color);">{pct:.0f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; align-items:center; padding-top:10px; margin-top:4px;">'
                    f'<span style="font-size:12.5px; color:var(--text-color); opacity:0.68;">Tổng cộng</span>'
                    f'<span style="font-size:13.5px; font-weight:800; color:var(--text-color);">${tot_rev:,.0f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---------- HÀNG 3: 3 THẺ PHÂN TÍCH CHI TIẾT ----------
    b1, b2, b3 = st.columns([1, 1.15, 1.15])

    with b1:
        with st.container(border=True):
            st.markdown('<div class="card-title">Doanh thu theo Quý</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">So sánh Q1 – Q4 giữa 2 năm</div>', unsafe_allow_html=True)

            target_y = latest_year if selected_year_opt == "Tất cả các năm" else int(selected_year_opt)
            comp_y = target_y - 1 if (target_y - 1) in years_avail else None
            quarters = [1, 2, 3, 4]
            q_curr = df_clean[df_clean["Year"] == target_y].groupby("Quarter")["Sales"].sum().reindex(quarters, fill_value=0).values

            fig_q, ax_q = plt.subplots(figsize=(4.4, 3.1))
            fig_q.patch.set_alpha(0.0)
            ax_q.patch.set_alpha(0.0)
            x_q = np.arange(4)
            width = 0.34

            if comp_y is not None:
                q_prev = df_clean[df_clean["Year"] == comp_y].groupby("Quarter")["Sales"].sum().reindex(quarters, fill_value=0).values
                ax_q.bar(x_q - width / 2, q_prev, width=width, color=INDIGO_SOFT, alpha=0.55, label=str(comp_y))
                bars_curr = ax_q.bar(x_q + width / 2, q_curr, width=width, color=INDIGO, label=str(target_y))
            else:
                bars_curr = ax_q.bar(x_q, q_curr, width=0.45, color=INDIGO, label=str(target_y))

            for bar in bars_curr:
                h = bar.get_height()
                ax_q.text(bar.get_x() + bar.get_width() / 2, h * 1.03, f"${h/1000:,.0f}k", ha="center", fontsize=8, fontweight="bold", color=chart_text_color)

            ax_q.set_xticks(x_q)
            ax_q.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=9, color=chart_muted_color)
            ax_q.set_yticks([])
            ax_q.set_ylim(0, max(q_curr.max(), 1) * 1.25)
            sns.despine(ax=ax_q, left=True, bottom=False)
            ax_q.spines["bottom"].set_color(chart_grid_color)
            leg_q = ax_q.legend(loc="upper left", frameon=False, fontsize=8.5)
            for txt in leg_q.get_texts():
                txt.set_color(chart_text_color)
            plt.tight_layout()
            st.pyplot(fig_q)
            plt.close(fig_q)

    with b2:
        with st.container(border=True):
            st.markdown('<div class="card-title">Hiệu suất theo Khu vực</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Tỷ lệ đóng góp doanh thu từng vùng thị trường</div>', unsafe_allow_html=True)

            if "Region" in view_df.columns:
                reg_s = view_df.groupby("Region")["Sales"].sum().sort_values(ascending=False).head(4)
                bar_palette = [INDIGO, TEAL, AMBER, SKY]
                for (r_name, r_val), b_col in zip(reg_s.items(), bar_palette):
                    pct = r_val / max(tot_rev, 1) * 100
                    st.markdown(
                        f'<div class="prog-row">'
                        f'<div class="prog-top">'
                        f'<span class="prog-name">{r_name}</span>'
                        f'<span><span class="prog-stats">${r_val:,.0f} / ${tot_rev:,.0f}</span><span class="prog-pct">{pct:.0f}%</span></span>'
                        f'</div>'
                        f'<div class="prog-track"><div class="prog-fill" style="width:{min(pct*2.2, 100):.1f}%; background:{b_col};"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Dữ liệu không có cột Khu vực (Region).")

    with b3:
        with st.container(border=True):
            st.markdown('<div class="card-title">Nguồn Doanh thu Chính</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-sub">Top mặt hàng có doanh số cao nhất</div>', unsafe_allow_html=True)

            group_col = "Product Name" if "Product Name" in view_df.columns else ("Sub-Category" if "Sub-Category" in view_df.columns else None)
            if group_col is not None:
                top_items = view_df.groupby(group_col)["Sales"].sum().sort_values(ascending=False).head(4)
                badge_styles = [
                    ("rgba(99, 91, 255, 0.16)", INDIGO, "💎"),
                    ("rgba(20, 184, 166, 0.16)", TEAL, "⚡"),
                    ("rgba(245, 158, 11, 0.16)", AMBER, "🔥"),
                    ("rgba(56, 189, 248, 0.16)", SKY, "⭐"),
                ]
                for (item_name, item_val), (bg_c, fg_c, ico) in zip(top_items.items(), badge_styles):
                    share = item_val / max(tot_rev, 1) * 100
                    short_lbl = item_name[:26] + "..." if len(str(item_name)) > 26 else str(item_name)
                    st.markdown(
                        f'<div class="source-row">'
                        f'<div class="source-left">'
                        f'<div class="source-badge" style="background:{bg_c}; color:{fg_c};">{ico}</div>'
                        f'<div class="source-name" title="{item_name}">{short_lbl}</div>'
                        f'</div>'
                        f'<div class="source-right">'
                        f'<span class="source-val">${item_val:,.0f}</span>'
                        f'<span class="source-share">{share:.1f}%</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Dữ liệu không có cột Tên sản phẩm.")

# ==============================================================================
# TAB 2: DỰ ĐOÁN DOANH THU TƯƠNG LAI
# ==============================================================================
with tab_forecast:
    artifact = load_saved_model()
    if artifact is None:
        st.error("❌ Chưa tìm thấy file mô hình `models/best_model.joblib`. Vui lòng chạy `python src/forecaster.py` trước.")
        st.stop()

    if len(monthly_sales) < 12:
        st.error(f"❌ Dữ liệu hiện tại chỉ có {len(monthly_sales)} tháng. Mô hình cần tối thiểu 12 tháng lịch sử.")
        st.stop()

    model = artifact["model"]
    model_name = artifact["model_name"]
    metrics_table = artifact["metrics_table"]

    if retrain_on_upload:
        if len(monthly_sales) >= 18:
            sup_df = create_supervised_dataset(monthly_sales)
            model.fit(sup_df[FEATURE_COLS], sup_df["Sales"].values)
            st.success("✅ Đã huấn luyện lại mô hình trực tiếp trên bộ dữ liệu mới tải lên!")
        else:
            st.warning("⚠️ Dữ liệu tải lên có ít hơn 18 tháng nên hệ thống giữ nguyên mô hình đã huấn luyện sẵn.")

    best_row = metrics_table[metrics_table["Mô hình"] == model_name]
    model_mae = float(best_row["MAE"].iloc[0]) if not best_row.empty else 0.0
    model_mape = float(best_row["MAPE (%)"].iloc[0]) if not best_row.empty else 0.0

    forecast_df = recursive_forecast(model, monthly_sales, steps=forecast_months)
    total_forecast = forecast_df["Predicted_Sales"].sum()
    avg_forecast = forecast_df["Predicted_Sales"].mean()

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        render_saas_kpi("Mô hình tối ưu", model_name, f"MAPE {model_mape:.2f}%", "độ lệch chuẩn", "🤖", "rgba(99, 91, 255, 0.16)", [18, 16, 15, 14, 13], INDIGO, True)
    with f2:
        render_saas_kpi(f"Dự báo {forecast_months} tháng", f"${total_forecast:,.0f}", f"+{forecast_months} kỳ", "tổng kỳ vọng", "🔮", "rgba(20, 184, 166, 0.16)", forecast_df["Predicted_Sales"].tolist(), TEAL, True)
    with f3:
        render_saas_kpi("Trung bình / Tháng", f"${avg_forecast:,.0f}", "Định mức TB", "mỗi tháng tới", "📊", "rgba(245, 158, 11, 0.16)", monthly_sales["Sales"].tail(8).tolist(), AMBER, True)
    with f4:
        render_saas_kpi("Biên độ sai số MAE", f"±${model_mae:,.0f}", "Ngưỡng ±MAE", "tập kiểm thử", "🎯", "rgba(244, 63, 94, 0.16)", [12, 11, 13, 10, 11], ROSE, False)

    with st.container(border=True):
        st.markdown(f'<div class="card-title">Biểu đồ Lịch sử & Dự báo Doanh thu {forecast_months} Tháng Tiếp Theo</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Đường liền nét màu tím Indigo là doanh thu thực tế; đường nét đứt màu đỏ hồng là dự báo đệ quy kèm dải sai số ±MAE</div>', unsafe_allow_html=True)

        hist_recent = monthly_sales.tail(24).copy()
        fig_fc, ax_fc = plt.subplots(figsize=(13, 4.4))
        fig_fc.patch.set_alpha(0.0)
        ax_fc.patch.set_alpha(0.0)

        ax_fc.plot(
            hist_recent["YearMonth"],
            hist_recent["Sales"],
            marker="o",
            markersize=5,
            linewidth=2.4,
            color=INDIGO,
            label="Doanh thu Lịch sử",
        )
        ax_fc.fill_between(hist_recent["YearMonth"], hist_recent["Sales"], color=INDIGO, alpha=0.10)

        bridge_x = [hist_recent["YearMonth"].iloc[-1]] + forecast_df["YearMonth"].tolist()
        bridge_y = np.array([hist_recent["Sales"].iloc[-1]] + forecast_df["Predicted_Sales"].tolist())

        ax_fc.plot(
            bridge_x,
            bridge_y,
            marker="s",
            markersize=6,
            linestyle="--",
            linewidth=2.4,
            color=ROSE,
            label=f"Dự báo {forecast_months} tháng tới ({model_name})",
        )

        if model_mae > 0:
            lower_b = np.maximum(0, bridge_y - model_mae)
            upper_b = bridge_y + model_mae
            lower_b[0] = bridge_y[0]
            upper_b[0] = bridge_y[0]
            ax_fc.fill_between(bridge_x, lower_b, upper_b, color=ROSE, alpha=0.14, label="Vùng sai số (±MAE)")

        ax_fc.axvline(x=hist_recent["YearMonth"].iloc[-1], color=chart_muted_color, linestyle=":", linewidth=1.3)
        ax_fc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${int(v/1000)}k"))
        ax_fc.tick_params(axis="y", colors=chart_muted_color, labelsize=8.5)
        ax_fc.grid(axis="x", visible=False)
        ax_fc.grid(axis="y", linestyle="-", color=chart_grid_color)
        sns.despine(ax=ax_fc, left=True)
        ax_fc.spines["bottom"].set_color(chart_grid_color)
        plt.xticks(rotation=45, ha="right", fontsize=8.5, color=chart_muted_color)
        leg_fc = ax_fc.legend(frameon=False, fontsize=9)
        for txt in leg_fc.get_texts():
            txt.set_color(chart_text_color)
        plt.tight_layout()
        st.pyplot(fig_fc)
        plt.close(fig_fc)

    c_left, c_right = st.columns([1.1, 1])
    with c_left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Chi tiết Doanh thu Dự báo từng Tháng</div>', unsafe_allow_html=True)
            display_fc = forecast_df.copy()
            display_fc.columns = ["Tháng dự báo", "Doanh thu dự báo (USD)"]
            st.dataframe(display_fc.style.format({"Doanh thu dự báo (USD)": "${:,.2f}"}), width="stretch", height=240)
            st.download_button(
                label="⬇ Tải bảng dự báo (.csv)",
                data=display_fc.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"du_bao_{forecast_months}_thang.csv",
                mime="text/csv",
                width="stretch",
            )

    with c_right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Bảng So sánh Sai số Các Mô hình</div>', unsafe_allow_html=True)
            st.dataframe(
                metrics_table.style.format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE (%)": "{:.2f}%"}),
                width="stretch",
                height=240,
            )

# ==============================================================================
# TAB 3: DỮ LIỆU & TIỀN XỬ LÝ
# ==============================================================================
with tab_data:
    with st.container(border=True):
        st.markdown('<div class="card-title">Dữ liệu Giao dịch sau Làm sạch & Chuẩn hóa</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-sub">Tổng số {len(df_clean):,} dòng giao dịch hợp lệ • Từ {df_clean["Order Date"].min().date()} đến {df_clean["Order Date"].max().date()}</div>', unsafe_allow_html=True)
        preview_df = df_clean.head(30).copy()
        preview_df["Order Date"] = preview_df["Order Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(preview_df, width="stretch", height=320)

    with st.container(border=True):
        st.markdown('<div class="card-title">Thống kê Mô tả các Biến định lượng</div>', unsafe_allow_html=True)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        st.dataframe(df_clean[num_cols].describe().round(2), width="stretch")