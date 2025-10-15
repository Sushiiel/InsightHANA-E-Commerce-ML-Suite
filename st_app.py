import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
from fpdf import FPDF
from hdbcli import dbapi
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce ML Suite", layout="wide")

def _cfg():
    s=st.secrets.get("hana",{})
    return{
        "address":s.get("address")or os.getenv("HANA_ADDRESS"),
        "port":int(s.get("port",os.getenv("HANA_PORT",443))),
        "user":s.get("user")or os.getenv("HANA_USER","DBADMIN"),
        "password":s.get("password")or os.getenv("HANA_PASSWORD"),
        "schema":s.get("schema")or os.getenv("HANA_SCHEMA","ECOMM_BRAZIL"),
        "encrypt":bool(s.get("encrypt",True)),
        "sslValidateCertificate":bool(s.get("sslValidateCertificate",False)),
    }

CFG=_cfg()
SCHEMA_NAME=CFG["schema"]

FILE_ALIASES={
    "customers":["olist_customers_dataset"],
    "geolocation":["olist_geolocation_dataset","compressed_data"],
    "orders":["olist_orders_dataset"],
    "order_items":["olist_order_items_dataset"],
    "payments":["olist_order_payments_dataset"],
    "reviews":["olist_order_reviews_dataset"],
    "products":["olist_products_dataset"],
    "sellers":["olist_sellers_dataset"],
    "categories":["product_category_name_translation"],
}

@st.cache_resource(show_spinner=False)
def get_connection():
    conn=dbapi.connect(
        address=CFG["address"],
        port=CFG["port"],
        user=CFG["user"],
        password=CFG["password"],
        encrypt=CFG["encrypt"],
        sslValidateCertificate=CFG["sslValidateCertificate"],
        timeout=10,
    )
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
    return{
        "customers":fetch("CUSTOMERS"),
        "geolocation":fetch("GEOLOCATION"),
        "orders":fetch("ORDERS"),
        "order_items":fetch("ORDER_ITEMS"),
        "payments":fetch("ORDER_PAYMENTS"),
        "reviews":fetch("ORDER_REVIEWS"),
        "products":fetch("PRODUCTS"),
        "sellers":fetch("SELLERS"),
        "categories":fetch("CATEGORY_TRANSLATION"),
    }

def _try_read_exact(path):
    if not os.path.exists(path):return None
    if path.endswith(".parquet"):return pd.read_parquet(path)
    return pd.read_csv(path)

def _try_read_aliases(bases):
    for base in bases:
        for ext in [".csv",".csv.gz",".parquet"]:
            p=os.path.join(".",base+ext)
            df=_try_read_exact(p)
            if df is not None:return df
    return None

@st.cache_data(show_spinner=False)
def load_from_csv():
    data={}
    for logical,aliases in FILE_ALIASES.items():
        df=_try_read_aliases(aliases)
        if df is None:
            if logical=="geolocation":data[logical]=pd.DataFrame()
            else:raise FileNotFoundError(f"Missing CSV for {logical}")
        else:data[logical]=df
    return data

@st.cache_data(show_spinner=False)
def load_synthetic():
    return{
        "customers":pd.DataFrame({"customer_id":["c1","c2"],"customer_state":["SP","RJ"]}),
        "geolocation":pd.DataFrame({}),
        "orders":pd.DataFrame({
            "order_id":["o1","o2"],
            "customer_id":["c1","c2"],
            "order_status":["delivered","canceled"],
            "order_purchase_timestamp":["2017-01-10","2017-01-11"],
            "order_estimated_delivery_date":["2017-01-15","2017-01-20"],
            "order_delivered_customer_date":["2017-01-14","2017-01-25"]
        }),
        "order_items":pd.DataFrame({"order_id":["o1","o2"],"product_id":["p1","p2"],"seller_id":["s1","s2"]}),
        "payments":pd.DataFrame({"order_id":["o1","o2"],"payment_installments":[3,2],"payment_value":[200.0,120.5]}),
        "reviews":pd.DataFrame({"order_id":["o1","o2"],"review_score":[5,2]}),
        "products":pd.DataFrame({"product_id":["p1","p2"],"product_category_name":["cat_a","cat_b"],"product_weight_g":[500,800],"product_photos_qty":[3,1],"product_description_lenght":[900,400]}),
        "sellers":pd.DataFrame({"seller_id":["s1","s2"],"seller_state":["SP","RJ"]}),
        "categories":pd.DataFrame({"product_category_name":["cat_a","cat_b"],"product_category_name_english":["cat_a","cat_b"]})
    }

def load_data():
    try:return load_from_hana(),"SAP HANA Cloud"
    except: 
        try:return load_from_csv(),"CSV (Olist)"
        except:return load_synthetic(),"Synthetic Demo"

def prepare_features(d):
    df=(d["orders"]
        .merge(d["order_items"],on="order_id",how="left")
        .merge(d["payments"],on="order_id",how="left")
        .merge(d["reviews"],on="order_id",how="left")
        .merge(d["customers"],on="customer_id",how="left")
        .merge(d["products"],on="product_id",how="left")
        .merge(d["sellers"],on="seller_id",how="left")
        .merge(d["categories"],on="product_category_name",how="left"))
    df["order_purchase_timestamp"]=pd.to_datetime(df.get("order_purchase_timestamp"),errors="coerce")
    df["order_delivered_customer_date"]=pd.to_datetime(df.get("order_delivered_customer_date"),errors="coerce")
    df["order_estimated_delivery_date"]=pd.to_datetime(df.get("order_estimated_delivery_date"),errors="coerce")
    df["review_score"]=pd.to_numeric(df.get("review_score"),errors="coerce")
    df["payment_value"]=pd.to_numeric(df.get("payment_value"),errors="coerce")
    df["payment_installments"]=pd.to_numeric(df.get("payment_installments"),errors="coerce")
    df["product_photos_qty"]=pd.to_numeric(df.get("product_photos_qty"),errors="coerce").fillna(0)
    df["product_description_lenght"]=pd.to_numeric(df.get("product_description_lenght"),errors="coerce").fillna(0)
    df["product_weight_g"]=pd.to_numeric(df.get("product_weight_g"),errors="coerce").fillna(0)
    df["purchase_dayofweek"]=df["order_purchase_timestamp"].dt.dayofweek.fillna(0)
    df["late_delivery"]=(df["order_delivered_customer_date"]>df["order_estimated_delivery_date"]).astype(int)
    df["churn"]=df["order_status"].isin(["canceled","unavailable"]).astype(int)
    df=df.dropna(subset=["review_score","payment_value"])
    X=df[["payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g","purchase_dayofweek"]]
    return X,df["review_score"],df["late_delivery"],df["churn"]

def get_or_train_model(path,cls,X,y):
    if os.path.exists(path):return joblib.load(path)
    m=cls(n_estimators=100,random_state=42)
    m.fit(X,y)
    joblib.dump(m,path)
    return m

def export_to_pdf(preds,file="report.pdf"):
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    pdf.cell(200,10,txt="E-Commerce Prediction Report",ln=True,align="C")
    pdf.ln(10)
    for k,v in preds.items():pdf.cell(200,10,txt=f"{k}: {v}",ln=True)
    pdf.output(file)

def explore_dataset(data):
    st.subheader("📂 Dataset Explorer")
    t=st.selectbox("Choose Table",list(data.keys()))
    df=data[t]
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(20),use_container_width=True)
    st.markdown("### 🔎 Summary Statistics")
    st.write(df.describe(include="all").transpose())
    st.markdown("### 📉 Missing Values")
    st.bar_chart(df.isnull().sum())
    num=df.select_dtypes(include=["int64","float64"]).columns.tolist()
    if num:
        st.markdown("### 📊 Numeric Column Distribution")
        col=st.selectbox("Select column",num)
        fig,ax=plt.subplots()
        sns.histplot(df[col].dropna(),kde=True,ax=ax)
        st.pyplot(fig)
    if len(num)>1:
        st.markdown("### 🔗 Correlation Heatmap")
        fig,ax=plt.subplots(figsize=(8,5))
        sns.heatmap(df[num].corr(),annot=True,cmap="coolwarm",ax=ax)
        st.pyplot(fig)

def main():
    st.title("📦 E-Commerce ML Suite")
    data,source=load_data()
    st.caption(f"Data Source: **{source}**")
    menu=st.sidebar.radio("Menu",["📊 View Sample Data","📊 Explore Dataset","📈 Predict Customer Behavior"])
    if menu=="📊 View Sample Data":
        t=st.selectbox("Select Table",list(data.keys()))
        st.dataframe(data[t].head(20),use_container_width=True)
    elif menu=="📊 Explore Dataset":
        explore_dataset(data)
    else:
        X,y_review,y_late,y_churn=prepare_features(data)
        review=get_or_train_model("review.pkl",RandomForestRegressor,X,y_review)
        late=get_or_train_model("late.pkl",RandomForestClassifier,X,y_late)
        churn=get_or_train_model("churn.pkl",RandomForestClassifier,X,y_churn)
        st.subheader("📝 Enter Order Details")
        pv=st.slider("Payment Value (R$)",0.0,5000.0,200.0)
        pi=st.slider("Installments",1,24,4)
        pq=st.slider("Product Photos Qty",0,20,5)
        dl=st.slider("Description Length",0,4000,1000)
        wg=st.slider("Product Weight (g)",0,10000,1000)
        dow=st.selectbox("Day of Week",list(range(7)))
        inp=pd.DataFrame([[pv,pi,pq,dl,wg,dow]],columns=X.columns)
        if st.button("🔍 Predict"):
            st.session_state["res"]={
                "Review Score":round(float(review.predict(inp)[0]),2),
                "Delivery":("Late" if late.predict(inp)[0] else "On Time"),
                "Churn Risk":("Yes" if churn.predict(inp)[0] else "No")
            }
        if "res" in st.session_state:
            r=st.session_state["res"]
            st.success(f"⭐ Review Score: {r['Review Score']}")
            st.info(f"🚚 Delivery: {r['Delivery']}")
            st.warning(f"📉 Churn Risk: {r['Churn Risk']}")
            if st.button("📄 Download PDF Report"):
                export_to_pdf(r)
                st.success("✅ Saved as report.pdf")

if __name__=="__main__":
    main()
