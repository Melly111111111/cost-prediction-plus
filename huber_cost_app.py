from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "huber_only_model.joblib"
DB_PATH = APP_DIR / "cost_feedback.db"


PAGE_CSS = """
<style>
    .stApp { background-color: #f7f9fb; }
    h1 {
        color: #0f3d5e;
        border-bottom: 2px solid #0f3d5e;
        padding-bottom: 10px;
        text-align: center;
        letter-spacing: 0;
    }
    .section-header {
        background-color: #eaf2f8;
        color: #0f3d5e;
        font-weight: 700;
        padding: 8px 12px;
        border-left: 5px solid #0f3d5e;
        margin: 12px 0 16px 0;
        text-align: center;
    }
    .price-box {
        background-color: white;
        border: 2px solid #0f3d5e;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 0;
    }
    .price-box h2 {
        color: #0f3d5e;
        margin: 8px 0 0 0;
        font-size: 34px;
        letter-spacing: 0;
    }
    .judgment-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 16px;
        text-align: center;
        font-weight: 700;
    }
    .judgment-success {
        background-color: #dff3e4;
        color: #145a32;
        border: 1px solid #b7dfc3;
    }
    .judgment-warning {
        background-color: #fff2cf;
        color: #7a5200;
        border: 1px solid #f5d889;
    }
    .judgment-danger {
        background-color: #fde2df;
        color: #8f1d14;
        border: 1px solid #efb4ae;
    }
    .history-band {
        background: #edf4fb;
        border-left: 6px solid #0f4c8a;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 8px 0 18px 0;
        font-weight: 800;
        color: #0f4c8a;
        font-size: 20px;
        letter-spacing: 0;
    }
    .history-band .history-emoji {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        font-size: 24px;
        line-height: 1;
    }
    .analysis-card {
        background: white;
        border: 1px solid #dde4eb;
        border-radius: 14px;
        padding: 22px 22px 18px 22px;
        min-height: 520px;
        box-shadow: 0 1px 4px rgba(15, 61, 94, 0.04);
    }
    .analysis-card h3 {
        margin: 0 0 14px 0;
        color: #0f4c8a;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0;
    }
    .analysis-card p, .analysis-card li {
        color: #495057;
        font-size: 16px;
        line-height: 1.8;
    }
    .analysis-card ul {
        margin: 10px 0 0 20px;
    }
    .analysis-divider {
        border-top: 1px solid #e2e8ef;
        margin: 18px 0 16px 0;
    }
    .analysis-rating-title {
        color: #0f4c8a;
        font-size: 24px;
        font-weight: 800;
        margin: 0 0 12px 0;
    }
    .analysis-rating {
        font-size: 18px;
        line-height: 1.8;
    }
    .rating-tag {
        font-weight: 800;
    }
    .rating-low {
        color: #1f9d55;
    }
    .rating-mid {
        color: #d97706;
    }
    .rating-high {
        color: #dc2626;
    }
    div.stButton > button[kind="primary"] {
        background-color: #f07c22 !important;
        border-color: #f07c22 !important;
        color: white !important;
        width: 100%;
        height: 48px;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 12px;
        border: 1px solid #e1e7ed;
        border-radius: 8px;
    }
</style>
"""


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            input_json TEXT NOT NULL,
            derived_json TEXT NOT NULL,
            prediction REAL NOT NULL,
            log_prediction REAL NOT NULL,
            avg_cost REAL,
            median_cost REAL,
            source TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            predicted_cost REAL NOT NULL,
            actual_cost REAL NOT NULL,
            error REAL NOT NULL,
            error_rate REAL NOT NULL,
            reviewer TEXT,
            note TEXT,
            input_json TEXT NOT NULL,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id)
        )
        """
    )
    conn.commit()
    return conn


def save_prediction(
    input_data: dict[str, object],
    derived_data: dict[str, object],
    prediction: float,
    log_prediction: float,
    avg_cost: float,
    median_cost: float,
    source: str,
) -> str:
    prediction_id = str(uuid.uuid4())
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (id, created_at, input_json, derived_json, prediction, log_prediction, avg_cost, median_cost, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now().isoformat(timespec="seconds"),
                json.dumps(input_data, ensure_ascii=False),
                json.dumps(derived_data, ensure_ascii=False),
                prediction,
                log_prediction,
                avg_cost,
                median_cost,
                source,
            ),
        )
        conn.commit()
    return prediction_id


def save_feedback(
    prediction_id: str,
    predicted_cost: float,
    actual_cost: float,
    reviewer: str,
    note: str,
    input_data: dict[str, object],
) -> None:
    error = actual_cost - predicted_cost
    error_rate = error / max(abs(actual_cost), 1e-12)
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO feedback
            (prediction_id, created_at, predicted_cost, actual_cost, error, error_rate, reviewer, note, input_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                datetime.now().isoformat(timespec="seconds"),
                predicted_cost,
                actual_cost,
                error,
                error_rate,
                reviewer,
                note,
                json.dumps(input_data, ensure_ascii=False),
            ),
        )
        conn.commit()


def read_feedback(limit: int = 200) -> pd.DataFrame:
    with connect_db() as conn:
        return pd.read_sql_query(
            """
            SELECT created_at AS 录入时间,
                   prediction_id AS 预测ID,
                   predicted_cost AS 预测成本,
                   actual_cost AS 人工确认成本,
                   error AS 误差,
                   error_rate AS 误差率,
                   reviewer AS 确认人,
                   note AS 备注
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


def read_predictions(limit: int = 200) -> pd.DataFrame:
    with connect_db() as conn:
        return pd.read_sql_query(
            """
            SELECT created_at AS 预测时间,
                   id AS 预测ID,
                   prediction AS 预测成本,
                   avg_cost AS 历史均值,
                   median_cost AS 历史中位数,
                   source AS 来源
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )


@st.cache_resource
def load_model_bundle() -> dict[str, object]:
    candidate_names = [
        os.environ.get("HUBER_MODEL_FILE"),
        "huber_only_model.joblib",
        "huber_model.joblib",
        "无灭菌纱布AI报价数据_成本预测模型.joblib",
        "成本预测模型.joblib",
    ]
    candidates: list[Path] = []
    for name in candidate_names:
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = APP_DIR / path
        candidates.append(path)
    candidates.extend(sorted(APP_DIR.rglob("*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True))

    seen = set()
    load_errors: list[str] = []
    for path in candidates:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            continue
        seen.add(normalized)
        if not path.exists():
            load_errors.append(f"{path.name}: 文件不存在")
            continue
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            load_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(bundle, dict) and "formula_pipeline" in bundle:
            bundle["_model_path"] = str(path)
            return bundle
        load_errors.append(f"{path.name}: 文件已读取，但不是包含 formula_pipeline 的Huber模型包")

    available = [str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*.joblib")]
    st.error("未找到可用的 Huber joblib 模型文件。")
    st.write("云端当前能看到的 joblib 文件：", available or "没有找到任何 .joblib 文件")
    st.write("加载尝试记录：")
    st.code("\n".join(load_errors) if load_errors else "没有加载记录")
    st.stop()


@st.cache_data
def load_history(input_file: str, target: str) -> pd.DataFrame:
    path = APP_DIR / input_file
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
    if target not in df.columns:
        raise ValueError(f"历史数据中找不到目标列：{target}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def get_feature_options(history: pd.DataFrame, col: str, fallback: list[str]) -> list[str]:
    if col not in history.columns:
        return fallback
    values = [str(v) for v in history[col].dropna().unique().tolist()]
    return sorted(values) or fallback


def default_value(history: pd.DataFrame, col: str, fallback: float) -> float:
    if col not in history.columns:
        return fallback
    series = pd.to_numeric(history[col], errors="coerce").dropna()
    if series.empty:
        return fallback
    return float(series.median())


def numeric_input(
    label: str,
    history: pd.DataFrame,
    col: str,
    fallback: float,
    min_value: float | None = None,
    step: float = 0.01,
    fmt: str = "%.2f",
) -> float:
    value = default_value(history, col, fallback)
    kwargs = {"label": label, "value": value, "step": step, "format": fmt}
    if min_value is not None:
        kwargs["min_value"] = min_value
    return float(st.number_input(**kwargs))


def build_input_row(input_data: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    row = dict(input_data)
    row["产品面积cm2"] = float(row["长（cm）"]) * float(row["宽（cm）"])
    row["用料指数"] = (
        float(row["长（cm）"])
        * float(row["宽（cm）"])
        * float(row["层数"])
        * float(row["克重g/㎡"])
    )
    row["MOQ_log"] = float(np.log1p(float(row["MOQ"])))
    row["纸箱面积"] = float(row["纸箱（长）"]) * float(row["纸箱（宽）"])
    row["纸箱体积"] = row["纸箱面积"] * float(row["纸箱（高）"])
    row["是否有纸箱"] = int(
        float(row["纸箱（长）"]) + float(row["纸箱（宽）"]) + float(row["纸箱（高）"]) > 0
    )
    people = float(row["折叠岗位人数"])
    row["折叠人均产能"] = (
        float(row["折叠产能（PCS)"]) / people if people not in (0.0, -0.0) else np.nan
    )
    derived = {
        "产品面积cm2": row["产品面积cm2"],
        "用料指数": row["用料指数"],
        "MOQ_log": row["MOQ_log"],
        "纸箱面积": row["纸箱面积"],
        "纸箱体积": row["纸箱体积"],
        "是否有纸箱": row["是否有纸箱"],
        "折叠人均产能": row["折叠人均产能"],
    }
    return pd.DataFrame([row]), derived


def predict_huber(bundle: dict[str, object], input_df: pd.DataFrame) -> tuple[float, float]:
    model = bundle["formula_pipeline"]
    expected_numeric = bundle["numeric_cols"]
    expected_categorical = bundle["categorical_cols"]
    expected_cols = list(expected_numeric) + list(expected_categorical)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[expected_cols]
    log_prediction = float(model.predict(input_df)[0])
    return float(np.exp(log_prediction)), log_prediction


def coefficient_analysis(bundle: dict[str, object], input_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    model = bundle["formula_pipeline"]
    preprocessor = model.named_steps["preprocess"]
    regressor = model.named_steps["model"]
    numeric_cols = list(bundle["numeric_cols"])
    categorical_cols = list(bundle["categorical_cols"])

    transformed = preprocessor.transform(input_df)
    coefs = np.asarray(regressor.coef_, dtype=float)
    feature_names = list(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(onehot.get_feature_names_out(categorical_cols)))

    contributions = np.asarray(transformed[0], dtype=float) * coefs
    result = pd.DataFrame(
        {
            "特征": feature_names,
            "模型输入值": np.asarray(transformed[0], dtype=float),
            "系数": coefs,
            "成本贡献": contributions,
            "影响方向": np.where(contributions >= 0, "推高成本", "降低成本"),
        }
    )
    result["绝对影响"] = result["成本贡献"].abs()
    return result.sort_values("绝对影响", ascending=False).head(top_n)


def find_history_match(history: pd.DataFrame, input_data: dict[str, object], target: str) -> pd.DataFrame:
    match_cols = [
        "长（cm）",
        "宽（cm）",
        "层数",
        "克重g/㎡",
        "粘胶配比%",
        "涤纶配比%",
        "机型（cm）",
        "开料（cm）",
        "断料（cm）",
        "灭菌方式",
        "MOQ",
        "纸袋（高）",
        "内包装材质",
        "材质色数（内）",
        "纸箱（长）",
        "纸箱（宽）",
        "纸箱（高）",
        "外箱材质",
        "材质色数（外）",
        "每箱数量（PCS)",
        "折叠产能（PCS)",
        "折叠岗位人数",
    ]
    mask = pd.Series(True, index=history.index)
    for col in match_cols:
        if col not in history.columns:
            continue
        if col in ["灭菌方式", "内包装材质", "外箱材质"]:
            mask &= history[col].astype(str) == str(input_data[col])
        else:
            left = pd.to_numeric(history[col], errors="coerce")
            mask &= np.isclose(left, float(input_data[col]), rtol=0, atol=1e-9, equal_nan=False)
    return history.loc[mask].dropna(subset=[target])


def render_distribution(history: pd.DataFrame, target: str, predicted_cost: float) -> go.Figure:
    data = history[target].dropna()
    data = data[data > 0]
    fig = go.Figure()
    if len(data) > 1:
        try:
            from scipy.stats import gaussian_kde

            x_min = float(data.min())
            x_max = float(data.max())
            padding = max((x_max - x_min) * 0.1, x_max * 0.05, 1e-6)
            x_range = np.linspace(x_min - padding, x_max + padding, 280)
            kde = gaussian_kde(data)
            y_kde = kde(x_range)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    fill="tozeroy",
                    line={"color": "#9ebbd1", "width": 4},
                    fillcolor="rgba(170, 192, 214, 0.48)",
                    name="历史成本密度",
                    hovertemplate="成本：%{x:.6f}<br>密度：%{y:.3f}<extra></extra>",
                )
            )
        except Exception:
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=40,
                    histnorm="probability density",
                    marker_color="#9fb8cc",
                    opacity=0.72,
                    name="历史成本分布",
                )
            )
    else:
        fig.add_trace(go.Scatter(x=data, y=np.ones(len(data)), mode="lines", line={"color": "#9ebbd1", "width": 4}))

    fig.add_vline(x=predicted_cost, line_width=3, line_dash="dash", line_color="#e5554f")
    peak_y = float(np.nanmax(fig.data[0].y)) if len(fig.data) and getattr(fig.data[0], "y", None) is not None else 1.0
    fig.add_annotation(
        x=predicted_cost,
        y=peak_y * 0.95 if peak_y > 0 else 0.95,
        text="当前核价位置",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#e5554f",
        font={"color": "#e5554f", "size": 14},
        bgcolor="white",
        bordercolor="#e5554f",
        borderpad=4,
    )
    fig.update_layout(
        height=520,
        title={"text": "内部成本分布密度图", "x": 0.42, "y": 0.94},
        xaxis_title="成本单价（元/PCS）",
        yaxis_title="出现频率（密度）",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#667085", "size": 14},
        margin={"l": 35, "r": 20, "t": 75, "b": 50},
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor="#dfe5ec",
        zeroline=False,
        tickformat=".3f",
        title_font={"size": 18, "color": "#667085"},
    )
    fig.update_yaxes(
        gridcolor="#dfe5ec",
        zeroline=False,
        title_font={"size": 18, "color": "#667085"},
    )
    return fig


def distribution_rating(predicted_cost: float, history: pd.Series) -> tuple[str, str, str]:
    if history.empty:
        return "中位区", "#d97706", "历史数据不足，当前结果仅供参考。"
    data = history.dropna()
    data = data[data > 0]
    if data.empty:
        return "中位区", "#d97706", "历史数据不足，当前结果仅供参考。"
    pct = float((data <= predicted_cost).mean() * 100)
    if pct <= 25:
        return "低价位", "#1f9d55", "极具市场竞争力。"
    if pct <= 75:
        return "中价位", "#d97706", "处于常规区间，适合正常报价。"
    if pct <= 90:
        return "偏高位", "#f59e0b", "建议核对工艺、包装或用料参数。"
    return "高价位", "#dc2626", "明显高于常规水平，建议重点复核成本构成。"


def history_percentile(predicted_cost: float, history: pd.Series) -> float | None:
    data = pd.to_numeric(history, errors="coerce").dropna()
    data = data[data > 0]
    if data.empty:
        return None
    return float((data <= predicted_cost).mean() * 100)


def prediction_judgment(predicted_cost: float, history: pd.Series) -> tuple[str, str]:
    pct = history_percentile(predicted_cost, history)
    if pct is None:
        return "judgment-warning", "历史数据不足，当前结果仅供参考。"
    if pct < 5:
        return "judgment-warning", f"当前预测低于历史5%分位（约{pct:.1f}%分位），建议复核是否存在漏填或异常低值。"
    if pct > 95:
        return "judgment-danger", f"当前预测高于历史95%分位（约{pct:.1f}%分位），建议重点复核成本构成。"
    return "judgment-success", f"当前预测位于历史常规区间（约{pct:.1f}%分位），可作为常规报价参考。"


def render_history_analysis_section(history: pd.DataFrame, target: str, predicted_cost: float) -> None:
    data = history[target].dropna()
    data = data[data > 0]
    if data.empty:
        st.info("历史数据不足，无法绘制分布分析图。")
        return

    label, color, text = distribution_rating(predicted_cost, data)
    pct = float((data <= predicted_cost).mean() * 100)

    st.markdown(
        """
        <div class="history-band">
            <span class="history-emoji">📊</span>
            历史基准对照分析
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.45, 1.0], gap="large")
    with left:
        st.plotly_chart(render_distribution(history, target, predicted_cost), use_container_width=True)
    with right:
        right_html = f"""
        <div class="analysis-card">
            <h3>📈 图像说明</h3>
            <p>该图展示了当前报价在公司历史报价库中的相对位置。</p>
            <ul>
                <li><b>蓝色阴影：</b>代表历史订单的成本分布。</li>
                <li><b>红色虚线：</b>代表您当前的核价结果。</li>
            </ul>
            <div class="analysis-divider"></div>
            <div class="analysis-rating-title">💡 评价：</div>
            <div class="analysis-rating">
                <span class="rating-tag" style="color:{color};">[{label}]</span>
                {text}
            </div>
            <div style="margin-top: 16px; color: #667085; font-size: 15px; line-height: 1.8;">
                当前成本位于历史数据的约 <b>{pct:.1f}%</b> 分位。
            </div>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)


st.set_page_config(page_title="纱布成本预测与人工校准系统", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

bundle = load_model_bundle()
target_col = str(bundle["target"])
history_df = load_history(str(bundle["input_file"]), target_col)
avg_cost = float(history_df[target_col].mean())
median_cost = float(history_df[target_col].median())

st.title("纱布产品成本智能预测系统")
st.info("预测结果会自动存储，人工确认成本也会保存，便于后续复盘和再训练。")

with st.expander("录入产品与包装参数", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        length = numeric_input("长（cm）", history_df, "长（cm）", 7.5)
        width = numeric_input("宽（cm）", history_df, "宽（cm）", 7.5)
        layers = numeric_input("层数", history_df, "层数", 4, min_value=1.0, step=1.0, fmt="%.0f")
        gram = numeric_input("克重g/㎡", history_df, "克重g/㎡", 30, min_value=0.0, step=1.0, fmt="%.0f")
        viscose = numeric_input("粘胶配比%", history_df, "粘胶配比%", 0.5, min_value=0.0, step=0.01, fmt="%.2f")
        polyester = numeric_input("涤纶配比%", history_df, "涤纶配比%", 0.5, min_value=0.0, step=0.01, fmt="%.2f")
    with col2:
        machine = numeric_input("机型（cm）", history_df, "机型（cm）", 7.3)
        cutting = numeric_input("开料（cm）", history_df, "开料（cm）", 14.0)
        breaking = numeric_input("断料（cm）", history_df, "断料（cm）", 14.0)
        sterilization = st.selectbox(
            "灭菌方式",
            get_feature_options(history_df, "灭菌方式", ["不灭菌", "EO预处理"]),
        )
        moq = numeric_input("MOQ", history_df, "MOQ", 1_920_000, min_value=0.0, step=1000.0, fmt="%.0f")
        paper_bag_h = numeric_input("纸袋（高）", history_df, "纸袋（高）", 8.5)
    with col3:
        inner_material = st.selectbox(
            "内包装材质",
            get_feature_options(history_df, "内包装材质", ["50g/㎡白鸡皮纸"]),
        )
        inner_colors = numeric_input("材质色数（内）", history_df, "材质色数（内）", 1, min_value=0.0, step=1.0, fmt="%.0f")
        carton_l = numeric_input("纸箱（长）", history_df, "纸箱（长）", 52)
        carton_w = numeric_input("纸箱（宽）", history_df, "纸箱（宽）", 32)
        carton_h = numeric_input("纸箱（高）", history_df, "纸箱（高）", 44)
        outer_material = st.selectbox(
            "外箱材质",
            get_feature_options(history_df, "外箱材质", ["单瓦", "双瓦"]),
        )
    with col4:
        outer_colors = numeric_input("材质色数（外）", history_df, "材质色数（外）", 1, min_value=0.0, step=1.0, fmt="%.0f")
        box_qty = numeric_input("每箱数量（PCS)", history_df, "每箱数量（PCS)", 10000, min_value=1.0, step=100.0, fmt="%.0f")
        fold_capacity = numeric_input("折叠产能（PCS)", history_df, "折叠产能（PCS)", 69231, min_value=0.0, step=100.0, fmt="%.0f")
        fold_people = numeric_input("折叠岗位人数", history_df, "折叠岗位人数", 5.0, min_value=0.01, step=0.01, fmt="%.2f")
        st.write("")
        predict_clicked = st.button("开始预测", type="primary")

input_payload = {
    "长（cm）": length,
    "宽（cm）": width,
    "层数": layers,
    "克重g/㎡": gram,
    "粘胶配比%": viscose,
    "涤纶配比%": polyester,
    "机型（cm）": machine,
    "开料（cm）": cutting,
    "断料（cm）": breaking,
    "灭菌方式": sterilization,
    "MOQ": moq,
    "纸袋（高）": paper_bag_h,
    "内包装材质": inner_material,
    "材质色数（内）": inner_colors,
    "纸箱（长）": carton_l,
    "纸箱（宽）": carton_w,
    "纸箱（高）": carton_h,
    "外箱材质": outer_material,
    "材质色数（外）": outer_colors,
    "每箱数量（PCS)": box_qty,
    "折叠产能（PCS)": fold_capacity,
    "折叠岗位人数": fold_people,
}

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if predict_clicked:
    input_df, derived_payload = build_input_row(input_payload)
    history_match = find_history_match(history_df, input_payload, target_col)
    if not history_match.empty:
        predicted_cost = float(history_match[target_col].iloc[0])
        log_prediction = float(np.log(predicted_cost))
        source = "历史完全匹配"
        st.success("匹配到历史完全一致记录，本次结果直接采用历史实际成本。")
    else:
        predicted_cost, log_prediction = predict_huber(bundle, input_df)
        source = "AI预测"

    prediction_id = save_prediction(
        input_payload,
        derived_payload,
        predicted_cost,
        log_prediction,
        avg_cost,
        median_cost,
        source,
    )
    analysis = coefficient_analysis(bundle, input_df)
    st.session_state.last_prediction = {
        "prediction_id": prediction_id,
        "predicted_cost": predicted_cost,
        "log_prediction": log_prediction,
        "source": source,
        "input_payload": input_payload,
        "derived_payload": derived_payload,
        "analysis": analysis,
    }

last = st.session_state.last_prediction
if last:
    predicted_cost = float(last["predicted_cost"])

    st.markdown('<div class="section-header">成本预测结果</div>', unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(
            f"""
            <div class="price-box">
                <p style="color:#596b7a; margin:0; font-size:14px;">预测成本单价（PCS）</p>
                <h2>{predicted_cost:.6f}</h2>
                <p style="color:#596b7a; margin:6px 0 0 0;">{last["source"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        klass, text = prediction_judgment(predicted_cost, history_df[target_col])
        st.markdown(f'<div class="judgment-box {klass}">{text}</div>', unsafe_allow_html=True)

        st.markdown("#### 人工确认成本")
        with st.form("feedback_form", clear_on_submit=False):
            actual_cost = st.number_input(
                "人工确认后的准确成本",
                min_value=0.0,
                value=predicted_cost,
                step=0.0001,
                format="%.6f",
            )
            reviewer = st.text_input("确认人", value="")
            note = st.text_area("备注", value="", height=80)
            submitted = st.form_submit_button("保存人工确认结果")
            if submitted:
                save_feedback(
                    str(last["prediction_id"]),
                    predicted_cost,
                    float(actual_cost),
                    reviewer.strip(),
                    note.strip(),
                    dict(last["input_payload"]),
                )
                st.success("人工确认成本已保存")

    with res_col2:
        analysis_df = last["analysis"]
        fig = px.bar(
            analysis_df.sort_values("成本贡献"),
            x="成本贡献",
            y="特征",
            orientation="h",
            color="成本贡献",
            color_continuous_scale="RdYlGn_r",
            title="各工艺参数对成本的影响权重",
        )
        fig.update_layout(
            height=430,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            coloraxis_showscale=False,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            analysis_df[["特征", "模型输入值", "系数", "成本贡献", "影响方向"]],
            use_container_width=True,
            hide_index=True,
        )

    render_history_analysis_section(history_df, target_col, predicted_cost)

with st.expander("数据库记录", expanded=False):
    tab1, tab2 = st.tabs(["人工确认记录", "预测记录"])
    with tab1:
        feedback_df = read_feedback()
        if feedback_df.empty:
            st.info("还没有人工确认记录。")
        else:
            feedback_df["误差率"] = feedback_df["误差率"].map(lambda x: f"{x * 100:+.2f}%")
            st.dataframe(feedback_df, use_container_width=True, hide_index=True)
            st.download_button(
                "下载人工确认记录 CSV",
                data=feedback_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="cost_feedback_records.csv",
                mime="text/csv",
            )
    with tab2:
        predictions_df = read_predictions()
        if predictions_df.empty:
            st.info("还没有预测记录。")
        else:
            st.dataframe(predictions_df, use_container_width=True, hide_index=True)
