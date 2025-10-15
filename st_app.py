"""
E-Commerce ML Analytics Suite — Optimized (Fast & Light)

What’s inside
- DB-first (HANA) with CSV fallback only (no synthetic data)
- Reviews removed (no review table, no review prediction/KPIs)
- Server-side SQL to compute features & project only needed cols
- Row limit + training sample controls
- Fast models: HistGradientBoostingClassifier
- Careful CSV loading (usecols + safe fallbacks), no truth-value DataFrame bugs
- Lightweight charts and optional PDF report

Run: `streamlit run app.py`
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier

# Optional deps
try:
    from hdbcli import dbapi
except Exception:
    dbapi = None

try:
    from fpdf import FPDF
except Exception:
    FPDF = None

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="E-Commerce ML Suite (Optimized)", layout="wide")

# -----------------------------
# Config
# -----------------------------

def _cfg():
    s = st.secrets.get("hana", {})
    return {
        "address": s.get("address") or os.getenv("HANA_ADDRESS"),
        "port": int(s.get("port", os.getenv("HANA_PORT", 443))),
        "user": s.get("user") or os.getenv("HANA_USER", "DBADMIN"),
        "password": s.get("password") or os.getenv("HANA_PASSWORD"),
        "schema": s.get("schema") or os.getenv("HANA_SCHEMA", "ECOMM_BRAZIL"),
        "encrypt": bool(s.get("encrypt", True)),
        "sslValidateCertificate": bool(s.get("sslValidateCertificate", False)),
    }

CFG = _cfg()
SCHEMA = CFG["schema"]

# -----------------------------
# DB helpers
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_connection():
    if dbapi is None or CFG["address"] is None:
        raise RuntimeError("HANA client or address not available")
    return dbapi.connect(
        address=CFG["address"],
        port=CFG["port"],
        user=CFG["user"],
        password=CFG["password"],
        encrypt=CFG["encrypt"],
        sslValidateCertificate=CFG["sslValidateCertificate"],
        timeout=10,
    )

@st.cache_data(show_spinner=False, ttl=600)
def fetch_df(query: str, params: tuple = ()):  # server-side compute + projection only
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0].lower() for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)

# -----------------------------
# Data Loading (DB-first, CSV fallback)
# -----------------------------

NEEDED_COLS = [
    "order_id",
    "payment_value",
    "payment_installments",
    "product_photos_qty",
    "product_description_lenght",
    "product_weight_g",
    "purchase_dayofweek",
    "late_delivery",
    "churn",
    "order_status",
]

FEATURE_COLS = [
    "payment_value",
    "payment_installments",
    "product_photos_qty",
    "product_description_lenght",
    "product_weight_g",
    "purchase_dayofweek",
]

@st.cache_data(show_spinner=False, ttl=600)
def load_from_hana(limit: int = 200_000):
    # Reviews removed; only required columns pulled
    q = f'''
    SELECT
        o."ORDER_ID" as order_id,
        COALESCE(p."PAYMENT_VALUE", 0)        as payment_value,
        COALESCE(p."PAYMENT_INSTALLMENTS", 0) as payment_installments,
        COALESCE(pr."PRODUCT_PHOTOS_QTY", 0)  as product_photos_qty,
        COALESCE(pr."PRODUCT_DESCRIPTION_LENGHT", 0) as product_description_lenght,
        COALESCE(pr."PRODUCT_WEIGHT_G", 0)    as product_weight_g,
        TO_INTEGER(DAYOFWEEK(o."ORDER_PURCHASE_TIMESTAMP")) - 1 as purchase_dayofweek,
        CASE WHEN o."ORDER_DELIVERED_CUSTOMER_DATE" > o."ORDER_ESTIMATED_DELIVERY_DATE" THEN 1 ELSE 0 END as late_delivery,
        CASE WHEN o."ORDER_STATUS" IN ('canceled','unavailable') THEN 1 ELSE 0 END as churn,
        o."ORDER_STATUS" as order_status
    FROM "{SCHEMA}"."ORDERS" o
    LEFT JOIN "{SCHEMA}"."ORDER_ITEMS" oi ON oi."ORDER_ID" = o."ORDER_ID"
    LEFT JOIN "{SCHEMA}"."ORDER_PAYMENTS" p ON p."ORDER_ID" = o."ORDER_ID"
    LEFT JOIN "{SCHEMA}"."PRODUCTS" pr ON pr."PRODUCT_ID" = oi."PRODUCT_ID"
    WHERE o."ORDER_PURCHASE_TIMESTAMP" IS NOT NULL
    LIMIT ?
    '''
    df = fetch_df(q, (limit,))
    # Downcast to reduce memory footprint
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce").fillna(0).astype("int16")
    for c in ["payment_value", "product_description_lenght", "product_weight_g", "product_photos_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    df["purchase_dayofweek"] = pd.to_numeric(df["purchase_dayofweek"], errors="coerce").fillna(0).astype("int8")
    df["late_delivery"] = df["late_delivery"].astype("int8")
    df["churn"] = df["churn"].astype("int8")
    return df

EXTS = [".csv", ".csv.gz", ".parquet"]

@st.cache_data(show_spinner=False, ttl=600)
def _try_read_alias(base, usecols=None):
    for ext in EXTS:
        p = os.path.join(".", base + ext)
        if os.path.exists(p):
            if p.endswith(".parquet"):
                return pd.read_parquet(p, columns=usecols)
            return pd.read_csv(p, usecols=usecols)
    return None

@st.cache_data(show_spinner=False, ttl=600)
def load_from_csv(limit_rows: int = 300_000):
    # Safe CSV loading without ambiguous truth-value checks
    orders = _try_read_alias("olist_orders_dataset", usecols=[
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_estimated_delivery_date", "order_delivered_customer_date",
    ])
    if orders is None:
        raise RuntimeError("olist_orders_dataset not found")

    items = _try_read_alias("olist_order_items_dataset", usecols=["order_id", "product_id"])
    if items is None:
        items = pd.DataFrame(columns=["order_id", "product_id"])  # safe empty

    pays = _try_read_alias("olist_order_payments_dataset", usecols=["order_id", "payment_installments", "payment_value"])
    if pays is None:
        pays = pd.DataFrame(columns=["order_id", "payment_installments", "payment_value"])  # safe empty

    prods = _try_read_alias("olist_products_dataset", usecols=[
        "product_id", "product_photos_qty", "product_description_lenght", "product_weight_g",
    ])
    if prods is None:
        prods = pd.DataFrame(columns=["product_id", "product_photos_qty", "product_description_lenght", "product_weight_g"])  # safe empty

    # Sample early to avoid massive merges
    if len(orders) > limit_rows:
        orders = orders.sample(n=limit_rows, random_state=42)

    df = orders.merge(items, on="order_id", how="left")
    if not pays.empty:
        df = df.merge(pays, on="order_id", how="left")
    if not prods.empty:
        df = df.merge(prods, on="product_id", how="left")

    dt = pd.to_datetime
    df["order_purchase_timestamp"] = dt(df["order_purchase_timestamp"], errors="coerce")
    df["order_estimated_delivery_date"] = dt(df["order_estimated_delivery_date"], errors="coerce")
    df["order_delivered_customer_date"] = dt(df["order_delivered_customer_date"], errors="coerce")

    # Feature engineering
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek.fillna(0).astype("int8")
    df["late_delivery"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).fillna(False).astype("int8")
    df["churn"] = df["order_status"].isin(["canceled", "unavailable"]).astype("int8")

    # Fill + downcast
    for c in ["payment_value", "product_description_lenght", "product_weight_g", "product_photos_qty"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype("float32")
    df["payment_installments"] = pd.to_numeric(df.get("payment_installments", 0), errors="coerce").fillna(0).astype("int16")

    return df[NEEDED_COLS]

# -----------------------------
# Feature Split
# -----------------------------

def to_Xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y_late = df["late_delivery"].astype("int8")
    y_churn = df["churn"].astype("int8")
    return X, y_late, y_churn

# -----------------------------
# Models (fast)
# -----------------------------

@st.cache_resource(show_spinner=False)
def train_models(X, y_late, y_churn):
    clf_late = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200, l2_regularization=0.01)
    clf_churn = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=200, l2_regularization=0.01)
    clf_late.fit(X, y_late)
    clf_churn.fit(X, y_churn)
    return clf_late, clf_churn

# -----------------------------
# PDF Export (lazy)
# -----------------------------

def export_pdf(pred: dict, file_name: str = "report.pdf"):
    if FPDF is None:
        raise RuntimeError("FPDF not installed")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="E-Commerce Prediction Report", ln=True, align="C")
    pdf.ln(8)
    for k, v in pred.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
    pdf.output(file_name)
    return file_name

# -----------------------------
# App
# -----------------------------

st.markdown("<h1 style='text-align:center;color:#4CAF50;'>⚡ E-Commerce ML Suite — Optimized</h1>", unsafe_allow_html=True)

# Controls
c0, c1, c2 = st.columns([2, 2, 2])
with c0:
    data_source = st.selectbox("Data Source Preference", ["HANA (fast SQL)", "CSV (fallback)"], index=0)
with c1:
    row_limit = st.number_input("Row Limit (pull fewer rows = faster)", min_value=5_000, max_value=1_000_000, value=150_000, step=5_000)
with c2:
    sample_for_training = st.slider("Train on % of pulled rows", min_value=10, max_value=100, value=50, step=10)

st.caption("Tip: reduce Row Limit and Train % for instant responsiveness.")

# Load data (DB -> CSV)
load_start = time.time()
if data_source.startswith("HANA"):
    try:
        df = load_from_hana(limit=int(row_limit))
        source = "SAP HANA Cloud"
    except Exception:
        df = load_from_csv(limit_rows=int(row_limit))
        source = "CSV (Olist)"
elif data_source.startswith("CSV"):
    df = load_from_csv(limit_rows=int(row_limit))
    source = "CSV (Olist)"
else:
    raise RuntimeError("Unsupported data source selection")
load_secs = time.time() - load_start

st.caption(f"Data Source: {source} • Rows: {len(df):,} • Loaded in {load_secs:.2f}s")

# Compute features/targets once
X, y_late, y_churn = to_Xy(df)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Prediction", "📈 Feature Importance"]) 

# Dashboard
with tab1:
    st.subheader("KPI Overview (fast)")
    total_orders = len(df)
    revenue = float(df["payment_value"].sum())
    churn_rate = float((df["churn"] == 1).mean() * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Orders", f"{total_orders:,}")
    c2.metric("Revenue (R$)", f"{revenue:,.2f}")
    c3.metric("Churn %", f"{churn_rate:0.2f}")

    if "order_status" in df.columns:
        vc = df["order_status"].value_counts()
        st.bar_chart(vc)

# Prediction
with tab2:
    # Subsample to train quickly
    if len(X) > 0:
        frac = max(0.1, min(1.0, sample_for_training / 100.0))
        X_train = X.sample(frac=frac, random_state=42)
        idx = X_train.index
        y_lat_tr = y_late.loc[idx]
        y_chu_tr = y_churn.loc[idx]
    else:
        X_train, y_lat_tr, y_chu_tr = X, y_late, y_churn

    with st.spinner("Training compact models..."):
        clf_late, clf_churn = train_models(X_train, y_lat_tr, y_chu_tr)

    st.subheader("Enter Order Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        payment_value = st.number_input("Payment Value", 0.0, 50000.0, 200.0)
    with c2:
        payment_installments = st.number_input("Installments", 0, 60, 4)
    with c3:
        product_photos_qty = st.number_input("Photos Qty", 0, 50, 5)
    c4, c5, c6 = st.columns(3)
    with c4:
        product_description_lenght = st.number_input("Description Length", 0, 10000, 1000)
    with c5:
        product_weight_g = st.number_input("Weight (g)", 0, 100000, 1000)
    with c6:
        purchase_dayofweek = st.selectbox("Day of Week (0=Mon)", list(range(7)), index=0)

    input_df = pd.DataFrame([[payment_value, payment_installments, product_photos_qty,
                              product_description_lenght, product_weight_g, purchase_dayofweek]],
                            columns=FEATURE_COLS)

    if st.button("Predict", use_container_width=True):
        pred_late = int(clf_late.predict(input_df)[0])
        pred_churn = int(clf_churn.predict(input_df)[0])
        st.session_state["prediction"] = {
            "is_late": pred_late,
            "will_churn": pred_churn,
        }

    if "prediction" in st.session_state:
        r = st.session_state["prediction"]
        col1, col2 = st.columns(2)
        col1.metric("🚚 Delivery", "Late" if r["is_late"] else "On Time")
        col2.metric("📉 Churn", "Yes" if r["will_churn"] else "No")

    st.divider()
    if "prediction" in st.session_state:
        r = st.session_state["prediction"]
        if FPDF is None:
            st.info("Install FPDF to enable PDF export: pip install fpdf2")
        disabled = FPDF is None
        if st.button("Generate PDF Report", use_container_width=True, disabled=disabled):
            try:
                path = export_pdf({
                    "Delivery": "Late" if r["is_late"] else "On Time",
                    "Churn": "Yes" if r["will_churn"] else "No",
                })
                with open(path, "rb") as f:
                    st.download_button("Download Report", data=f, file_name=path)
            except Exception as e:
                st.error(f"PDF export failed: {e}")

# Feature Importance
with tab3:
    st.subheader("Feature Importance (approx.)")
    try:
        # Quick model on a subset to compute importances
        n = min(10000, len(df))
        if n > 10:
            samp = df.sample(n=n, random_state=42)
            Xs, ys = samp[FEATURE_COLS], samp["churn"].astype("int8")
            tmp_clf = HistGradientBoostingClassifier().fit(Xs, ys)
            fi_vals = getattr(tmp_clf, "feature_importances_", None)
            if fi_vals is not None:
                fi = pd.Series(fi_vals, index=FEATURE_COLS).sort_values(ascending=False)
            else:
                raise AttributeError("No feature_importances_")
        else:
            fi = pd.Series({c: 0.0 for c in FEATURE_COLS})
    except Exception:
        # Fallback: absolute correlation with churn
        def _corr(a, b):
            try:
                return float(np.corrcoef(a, b)[0, 1])
            except Exception:
                return 0.0
        fi = pd.Series({c: abs(_corr(df[c].values, df["churn"].values)) if df[c].std() > 0 else 0.0
                        for c in FEATURE_COLS}).fillna(0).sort_values(ascending=False)

    st.bar_chart(fi)
