import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import joblib
from fpdf import FPDF
from hdbcli import dbapi

st.set_page_config(page_title="E-Commerce ML Suite",layout="wide")

def _cfg():
    s=st.secrets.get("hana",{})
    return {
        "address":s.get("address")or os.getenv("HANA_ADDRESS"),
        "port":int(s.get("port",os.getenv("HANA_PORT",443))),
        "user":s.get("user")or os.getenv("HANA_USER","DBADMIN"),
        "password":s.get("password")or os.getenv("HANA_PASSWORD"),
        "schema":s.get("schema")or os.getenv("HANA_SCHEMA","ECOMM_BRAZIL"),
        "encrypt":bool(s.get("encrypt",True)),
        "sslValidateCertificate":bool(s.get("sslValidateCertificate",False))
    }

CFG=_cfg()
SCHEMA_NAME=CFG["schema"]
EXTS=[".csv",".csv.gz",".parquet"]
FILE_ALIASES={
    "customers":["olist_customers_dataset"],
    "geolocation":["olist_geolocation_dataset","compressed_data"],
    "orders":["olist_orders_dataset"],
    "order_items":["olist_order_items_dataset"],
    "payments":["olist_order_payments_dataset"],
    "reviews":["olist_order_reviews_dataset"],
    "products":["olist_products_dataset"],
    "sellers":["olist_sellers_dataset"],
    "categories":["product_category_name_translation"]
}

@st.cache_resource(show_spinner=False)
def get_connection():
    conn=dbapi.connect(address=CFG["address"],port=CFG["port"],user=CFG["user"],password=CFG["password"],encrypt=CFG["encrypt"],sslValidateCertificate=CFG["sslValidateCertificate"],timeout=10)
    cur=conn.cursor()
    cur.execute("SELECT 'OK' FROM DUMMY")
    _=cur.fetchall()
    return conn

@st.cache_data(show_spinner=False)
def load_from_hana():
    conn=get_connection()
    cur=conn.cursor()
    def fetch(table):
        cur.execute(f'SELECT * FROM "{SCHEMA_NAME}"."{table}"')
        cols=[d[0].lower() for d in cur.description]
        rows=cur.fetchall()
        return pd.DataFrame(rows,columns=cols)
    return {
        "customers":fetch("CUSTOMERS"),
        "geolocation":fetch("GEOLOCATION"),
        "orders":fetch("ORDERS"),
        "order_items":fetch("ORDER_ITEMS"),
        "payments":fetch("ORDER_PAYMENTS"),
        "reviews":fetch("ORDER_REVIEWS"),
        "products":fetch("PRODUCTS"),
        "sellers":fetch("SELLERS"),
        "categories":fetch("CATEGORY_TRANSLATION")
    }

def _try_read_exact(path):
    if not os.path.exists(path):
        return None
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _try_read_aliases(basenames):
    for base in basenames:
        for ext in EXTS:
            p=os.path.join(".",base+ext)
            df=_try_read_exact(p)
            if df is not None:
                return df
    return None

@st.cache_data(show_spinner=False)
def load_from_csv_olist():
    data={}
    for logical,aliases in FILE_ALIASES.items():
        df=_try_read_aliases(aliases)
        if df is None:
            data[logical]=pd.DataFrame()
        else:
            data[logical]=df
    return data

@st.cache_data(show_spinner=False)
def load_synthetic_demo():
    return {
        "customers":pd.DataFrame({"customer_id":["c1","c2"],"customer_unique_id":["u1","u2"],"customer_state":["SP","RJ"]}),
        "geolocation":pd.DataFrame({}),
        "orders":pd.DataFrame({"order_id":["o1","o2"],"customer_id":["c1","c2"],"order_status":["delivered","canceled"],"order_purchase_timestamp":["2017-01-10","2017-01-11"],"order_estimated_delivery_date":["2017-01-15","2017-01-20"],"order_delivered_customer_date":["2017-01-14","2017-01-25"]}),
        "order_items":pd.DataFrame({"order_id":["o1","o2"],"order_item_id":[1,1],"product_id":["p1","p2"],"seller_id":["s1","s2"]}),
        "payments":pd.DataFrame({"order_id":["o1","o2"],"payment_sequential":[1,1],"payment_type":["credit_card","boleto"],"payment_installments":[3,2],"payment_value":[200.0,120.5]}),
        "reviews":pd.DataFrame({"review_id":["r1","r2"],"order_id":["o1","o2"],"review_score":[5,2]}),
        "products":pd.DataFrame({"product_id":["p1","p2"],"product_category_name":["cat_a","cat_b"],"product_weight_g":[500,800],"product_photos_qty":[3,1],"product_description_lenght":[900,400]}),
        "sellers":pd.DataFrame({"seller_id":["s1","s2"],"seller_state":["SP","RJ"]}),
        "categories":pd.DataFrame({"product_category_name":["cat_a","cat_b"],"product_category_name_english":["cat_a","cat_b"]})
    }

def load_data():
    try:
        return load_from_hana(),"SAP HANA Cloud"
    except Exception:
        try:
            return load_from_csv_olist(),"CSV (Olist)"
        except Exception:
            return load_synthetic_demo(),"Synthetic Demo"

def prepare_features(d):
    df=d["orders"].merge(d["order_items"],on="order_id",how="left").merge(d["payments"],on="order_id",how="left").merge(d["reviews"],on="order_id",how="left").merge(d["customers"],on="customer_id",how="left").merge(d["products"],on="product_id",how="left").merge(d["sellers"],on="seller_id",how="left").merge(d["categories"],on="product_category_name",how="left")
    dt=pd.to_datetime;num=pd.to_numeric
    for c in ["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"]:
        if c in df.columns:df[c]=dt(df[c],errors="coerce")
    for c in ["review_score","payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g"]:
        if c in df.columns:df[c]=num(df[c],errors="coerce")
    df["product_photos_qty"]=df.get("product_photos_qty",0).fillna(0)
    df["product_description_lenght"]=df.get("product_description_lenght",0).fillna(0)
    df["product_weight_g"]=df.get("product_weight_g",0).fillna(0)
    if "order_purchase_timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"]):
        df["purchase_dayofweek"]=df["order_purchase_timestamp"].dt.dayofweek
    else:df["purchase_dayofweek"]=0
    if {"order_delivered_customer_date","order_estimated_delivery_date"}.issubset(df.columns):
        df["late_delivery"]=(df["order_delivered_customer_date"]>df["order_estimated_delivery_date"]).astype(int)
    else:df["late_delivery"]=0
    if "order_status" in df.columns:df["churn"]=df["order_status"].isin(["canceled","unavailable"]).astype(int)
    else:df["churn"]=0
    if {"review_score","payment_value"}.issubset(df.columns):df=df.dropna(subset=["review_score","payment_value"])
    X=df[["payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g","purchase_dayofweek"]]
    y_review=df["review_score"];y_late=df["late_delivery"];y_churn=df["churn"]
    return X,y_review,y_late,y_churn

def get_or_train_model(path,model_type,X,y):
    if os.path.exists(path):return joblib.load(path)
    model=model_type(n_estimators=100,random_state=42);model.fit(X,y);joblib.dump(model,path);return model

def export_to_pdf(predictions:dict,file_name="report.pdf"):
    pdf=FPDF();pdf.add_page();pdf.set_font("Arial",size=12)
    pdf.cell(200,10,txt="E-Commerce Prediction Report",ln=True,align="C");pdf.ln(10)
    for k,v in predictions.items():pdf.cell(200,10,txt=f"{k}: {v}",ln=True)
    pdf.output(file_name)

def main():
    st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📦 Intelligent E-Commerce ML Suite</h1>",unsafe_allow_html=True)
    data,source=load_data()
    st.caption(f"Data Source: {source}")
    tab1,tab2=st.tabs(["📊 Dashboard","🤖 Prediction Engine"])
    with tab1:
        st.subheader("Explore Tables")
        table=st.selectbox("Select a table",list(data.keys()))
        st.dataframe(data[table].head(20),use_container_width=True)
    with tab2:
        X,y_review,y_late,y_churn=prepare_features(data)
        review_model=get_or_train_model("review_model.pkl",RandomForestRegressor,X,y_review)
        late_model=get_or_train_model("late_model.pkl",RandomForestClassifier,X,y_late)
        churn_model=get_or_train_model("churn_model.pkl",RandomForestClassifier,X,y_churn)
        st.subheader("Enter Order Details")
        c1,c2,c3=st.columns(3)
        with c1:payment_value=st.number_input("Payment Value (R$)",0.0,5000.0,200.0)
        with c2:payment_installments=st.number_input("Installments",1,24,4)
        with c3:product_photos_qty=st.number_input("Product Photos Qty",0,20,5)
        c4,c5,c6=st.columns(3)
        with c4:product_description_lenght=st.number_input("Description Length",0,4000,1000)
        with c5:product_weight_g=st.number_input("Product Weight (g)",0,10000,1000)
        with c6:purchase_dayofweek=st.selectbox("Day of Week (0=Mon,6=Sun)",list(range(7)))
        input_df=pd.DataFrame([[payment_value,payment_installments,product_photos_qty,product_description_lenght,product_weight_g,purchase_dayofweek]],columns=X.columns)
        if st.button("🔍 Predict",use_container_width=True):
            st.session_state["prediction_result"]={"review_score":round(float(review_model.predict(input_df)[0]),2),"is_late":int(late_model.predict(input_df)[0]),"will_churn":int(churn_model.predict(input_df)[0])}
        if "prediction_result" in st.session_state:
            res=st.session_state["prediction_result"]
            col1,col2,col3=st.columns(3)
            col1.metric("⭐ Review Score",res["review_score"])
            col2.metric("🚚 Delivery","Late" if res["is_late"] else "On Time")
            col3.metric("📉 Churn Risk","Yes" if res["will_churn"] else "No")
            if st.button("📄 Download PDF Report",use_container_width=True):
                export_to_pdf({"Predicted Review Score":res["review_score"],"Delivery Status":"Late" if res["is_late"] else "On Time","Churn Risk":"Yes" if res["will_churn"] else "No"})
                st.success("✅ PDF saved as report.pdf in current directory")

if __name__=="__main__":
    main()
