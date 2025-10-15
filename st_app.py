# st_app.py — HANA + CSV fallback (CSV files in project root)

import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from fpdf import FPDF
from hdbcli import dbapi

# ============== PAGE CONFIG ==============
st.set_page_config(page_title="E-Commerce ML Dashboard", layout="centered")

# ============== CONFIG / SECRETS ==============
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
SCHEMA_NAME = CFG["schema"]

# CSVs are in the project root (same folder as this file)
CSV_DIR = "."
CSV_FILES = {
    "customers": "customers.csv",
    "geolocation": "geolocation.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "order_payments.csv",
    "reviews": "order_reviews.csv",
    "products": "products.csv",
    "sellers": "sellers.csv",
    "categories": "category_translation.csv",
}

# ============== CONNECTION HELPERS ==============
@st.cache_resource(show_spinner=False)
def get_connection():
    """Connect to HANA with a short timeout and perform a quick health check."""
    conn = dbapi.connect(
        address=CFG["address"],
        port=CFG["port"],
        user=CFG["user"],
        password=CFG["password"],
        encrypt=CFG["encrypt"],
        sslValidateCertificate=CFG["sslValidateCertificate"],
        timeout=10,
    )
    cur = conn.cursor()
    cur.execute("SELECT 'OK' AS STATUS FROM DUMMY")
    _ = cur.fetchall()
    return conn

def _read_csv_bundle():
    """Load CSVs from the project root; return dict or None if any file is missing."""
    bundle = {}
    for key, fname in CSV_FILES.items():
        path = os.path.join(CSV_DIR, fname)
        if not os.path.exists(path):
            return None
        bundle[key] = pd.read_csv(path)
    return bundle

@st.cache_data(show_spinner=False)
def load_from_hana():
    """Load all required tables from HANA (columns lower-cased)."""
    conn = get_connection()
    cur = conn.cursor()

    def fetch_table(table):
        cur.execute(f'SELECT * FROM "{SCHEMA_NAME}"."{table}"')
        cols = [c[0].lower() for c in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)

    return {
        "customers":   fetch_table("CUSTOMERS"),
        "geolocation": fetch_table("GEOLOCATION"),
        "orders":      fetch_table("ORDERS"),
        "order_items": fetch_table("ORDER_ITEMS"),
        "payments":    fetch_table("ORDER_PAYMENTS"),
        "reviews":     fetch_table("ORDER_REVIEWS"),
        "products":    fetch_table("PRODUCTS"),
        "sellers":     fetch_table("SELLERS"),
        "categories":  fetch_table("CATEGORY_TRANSLATION"),
    }

@st.cache_data(show_spinner=False)
def load_from_csv():
    """Load all required tables from CSVs in the project root."""
    bundle = _read_csv_bundle()
    if bundle is None:
        raise RuntimeError(
            "CSV files not found in project root. "
            "Place the required CSVs next to st_app.py."
        )
    return bundle

def load_data(mode: str):
    """
    mode ∈ {"Auto (HANA→CSV)", "Force HANA", "Force CSV"}
    Returns (data_dict, source_label)
    """
    if mode == "Force HANA":
        data = load_from_hana()
        return data, "SAP HANA Cloud"

    if mode == "Force CSV":
        data = load_from_csv()
        return data, "CSV Demo"

    # Auto: try HANA first, then CSV
    try:
        data = load_from_hana()
        return data, "SAP HANA Cloud"
    except Exception:
        data = load_from_csv()
        return data, "CSV Demo"

# ============== FEATURES / MODELS ==============
def prepare_features(d):
    df = (
        d["orders"]
        .merge(d["order_items"], on="order_id", how="left")
        .merge(d["payments"], on="order_id", how="left")
        .merge(d["reviews"], on="order_id", how="left")
        .merge(d["customers"], on="customer_id", how="left")
        .merge(d["products"], on="product_id", how="left")
        .merge(d["sellers"], on="seller_id", how="left")
        .merge(d["categories"], on="product_category_name", how="left")
    )

    dt = pd.to_datetime
    num = pd.to_numeric

    df["order_purchase_timestamp"] = dt(df["order_purchase_timestamp"], errors="coerce")
    df["order_delivered_customer_date"] = dt(df["order_delivered_customer_date"], errors="coerce")
    df["order_estimated_delivery_date"] = dt(df["order_estimated_delivery_date"], errors="coerce")

    df["review_score"] = num(df["review_score"], errors="coerce")
    df["payment_value"] = num(df["payment_value"], errors="coerce")
    df["payment_installments"] = num(df["payment_installments"], errors="coerce")
    df["product_photos_qty"] = num(df.get("product_photos_qty", 0), errors="coerce").fillna(0)
    # Keep original misspelling as per many public datasets
    df["product_description_lenght"] = num(df.get("product_description_lenght", 0), errors="coerce").fillna(0)
    df["product_weight_g"] = num(df.get("product_weight_g", 0), errors="coerce").fillna(0)

    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["late_delivery"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)
    df["churn"] = df["order_status"].isin(["canceled", "unavailable"]).astype(int)

    df = df.dropna(subset=["review_score", "payment_value"])

    X = df[
        [
            "payment_value",
            "payment_installments",
            "product_photos_qty",
            "product_description_lenght",
            "product_weight_g",
            "purchase_dayofweek",
        ]
    ]
    y_review = df["review_score"]
    y_late = df["late_delivery"]
    y_churn = df["churn"]
    return X, y_review, y_late, y_churn

def get_or_train_model(path, model_type, X, y):
    if os.path.exists(path):
        return joblib.load(path)
    model = model_type(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, path)
    return model

def export_to_pdf(predictions: dict, file_name="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="E-Commerce Prediction Report", ln=True, align="C")
    pdf.ln(10)
    for label, value in predictions.items():
        pdf.cell(200, 10, txt=f"{label}: {value}", ln=True)
    pdf.output(file_name)

# ============== UI ==============
def main():
    st.title("📦 Intelligent E-Commerce Prediction Engine")

    # Data source selector
    source_mode = st.sidebar.radio(
        "🔌 Data Source Mode",
        ("Auto (HANA→CSV)", "Force HANA", "Force CSV"),
        index=0,
        help="Auto tries HANA first then falls back to CSV in the project root.",
    )

    # Load data
    try:
        data, source = load_data(source_mode)
        st.caption(f"Data Source: **{source}**")
        if source == "CSV Demo":
            st.warning("Using CSV demo data (HANA not reachable or 'Force CSV' selected).")
    except Exception as e:
        st.error(f"❌ Unable to load data: {e}")
        st.stop()

    menu = st.sidebar.selectbox("📋 Menu", ["📊 View Sample Data", "📈 Predict Customer Behavior"])

    if menu == "📊 View Sample Data":
        st.subheader("🔍 Explore Tables")
        table_name = st.selectbox("Select a table", list(data.keys()))
        st.dataframe(data[table_name].head(20), use_container_width=True)

    else:
        st.subheader("🧠 Train & Predict")

        with st.spinner("🔄 Preparing features and training models..."):
            X, y_review, y_late, y_churn = prepare_features(data)
            review_model = get_or_train_model("review_model.pkl", RandomForestRegressor, X, y_review)
            late_model = get_or_train_model("late_model.pkl", RandomForestClassifier, X, y_late)
            churn_model = get_or_train_model("churn_model.pkl", RandomForestClassifier, X, y_churn)

        st.subheader("📝 Enter Order Details")
        payment_value = st.slider("Payment Value (R$)", 0.0, 5000.0, 200.0)
        payment_installments = st.slider("Installments", 1, 24, 4)
        product_photos_qty = st.slider("Product Photos Qty", 0, 20, 5)
        product_description_lenght = st.slider("Description Length", 0, 4000, 1000)
        product_weight_g = st.slider("Product Weight (g)", 0, 10000, 1000)
        purchase_dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))

        input_df = pd.DataFrame(
            [[
                payment_value,
                payment_installments,
                product_photos_qty,
                product_description_lenght,
                product_weight_g,
                purchase_dayofweek,
            ]],
            columns=X.columns,
        )

        if st.button("🔍 Predict"):
            st.session_state["prediction_result"] = {
                "review_score": round(float(review_model.predict(input_df)[0]), 2),
                "is_late": int(late_model.predict(input_df)[0]),
                "will_churn": int(churn_model.predict(input_df)[0]),
            }

        if "prediction_result" in st.session_state:
            result = st.session_state["prediction_result"]
            st.success(f"⭐ Predicted Review Score: {result['review_score']}")
            st.info(f"🚚 Delivery: {'Late' if result['is_late'] else 'On Time'}")
            st.warning(f"📉 Churn Risk: {'Yes' if result['will_churn'] else 'No'}")

            if st.button("📄 Download PDF Report"):
                export_to_pdf(
                    {
                        "Predicted Review Score": result["review_score"],
                        "Delivery Status": "Late" if result["is_late"] else "On Time",
                        "Churn Risk": "Yes" if result["will_churn"] else "No",
                    }
                )
                st.success("✅ PDF saved as report.pdf in current directory")


if __name__ == "__main__":
    main()
