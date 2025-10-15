import os, time
import pandas as pd, numpy as np, streamlit as st
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from hdbcli import dbapi
except:
    dbapi=None

try:
    from fpdf import FPDF
except:
    FPDF=None

st.set_page_config(page_title="E-Commerce ML Suite", layout="wide")

def get_config():
    s=st.secrets.get("hana",{})
    return {
        "address":s.get("address") or os.getenv("HANA_ADDRESS"),
        "port":int(s.get("port",os.getenv("HANA_PORT",443))),
        "user":s.get("user") or os.getenv("HANA_USER","DBADMIN"),
        "password":s.get("password") or os.getenv("HANA_PASSWORD"),
        "schema":s.get("schema") or os.getenv("HANA_SCHEMA","ECOMM_BRAZIL"),
        "encrypt":bool(s.get("encrypt",True)),
        "sslValidateCertificate":bool(s.get("sslValidateCertificate",False))
    }

DB_CONFIG=get_config()
DB_SCHEMA=DB_CONFIG["schema"]

@st.cache_resource
def get_conn():
    if dbapi is None or DB_CONFIG["address"] is None:
        raise RuntimeError("HANA not available")
    return dbapi.connect(
        address=DB_CONFIG["address"],port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],password=DB_CONFIG["password"],
        encrypt=DB_CONFIG["encrypt"],
        sslValidateCertificate=DB_CONFIG["sslValidateCertificate"],
        timeout=10
    )

@st.cache_data(ttl=600)
def fetch_df(sql,params=()):
    cur=get_conn().cursor()
    cur.execute(sql,params)
    cols=[d[0].lower() for d in cur.description]
    return pd.DataFrame(cur.fetchall(),columns=cols)

FEATURES=[
    "payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g",
    "purchase_dayofweek","num_items","avg_product_weight_g","avg_product_photos_qty","avg_product_description_lenght",
    "value_per_item","price_per_installment","est_delivery_days","actual_delivery_days","delay_days","is_weekend",
    "purchase_month","purchase_hour"
]

@st.cache_data(ttl=600)
def load_from_hana(limit=200000):
    q=f'''
    SELECT
        o."ORDER_ID" AS order_id,
        COALESCE(p_sum.sum_payment_value,0) AS payment_value,
        COALESCE(p_sum.max_installments,0) AS payment_installments,
        COALESCE(i_cnt.num_items,0) AS num_items,
        COALESCE(pr_agg.avg_product_photos_qty,0) AS avg_product_photos_qty,
        COALESCE(pr_agg.avg_product_description_lenght,0) AS avg_product_description_lenght,
        COALESCE(pr_agg.avg_product_weight_g,0) AS avg_product_weight_g,
        TO_INTEGER(DAYOFWEEK(o."ORDER_PURCHASE_TIMESTAMP"))-1 AS purchase_dayofweek,
        EXTRACT(MONTH FROM o."ORDER_PURCHASE_TIMESTAMP")-1 AS purchase_month,
        EXTRACT(HOUR FROM o."ORDER_PURCHASE_TIMESTAMP") AS purchase_hour,
        DAYS_BETWEEN(o."ORDER_PURCHASE_TIMESTAMP",o."ORDER_ESTIMATED_DELIVERY_DATE") AS est_delivery_days,
        DAYS_BETWEEN(o."ORDER_PURCHASE_TIMESTAMP",o."ORDER_DELIVERED_CUSTOMER_DATE") AS actual_delivery_days,
        CASE WHEN o."ORDER_DELIVERED_CUSTOMER_DATE">o."ORDER_ESTIMATED_DELIVERY_DATE"
             THEN DAYS_BETWEEN(o."ORDER_ESTIMATED_DELIVERY_DATE",o."ORDER_DELIVERED_CUSTOMER_DATE") ELSE 0 END AS delay_days,
        CASE WHEN (TO_INTEGER(DAYOFWEEK(o."ORDER_PURCHASE_TIMESTAMP"))-1) IN (5,6) THEN 1 ELSE 0 END AS is_weekend,
        CASE WHEN o."ORDER_DELIVERED_CUSTOMER_DATE">o."ORDER_ESTIMATED_DELIVERY_DATE" THEN 1 ELSE 0 END AS late_delivery,
        CASE WHEN o."ORDER_STATUS" IN ('canceled','unavailable') THEN 1 ELSE 0 END AS churn,
        o."ORDER_STATUS" AS order_status
    FROM "{DB_SCHEMA}"."ORDERS" o
    LEFT JOIN (SELECT "ORDER_ID",SUM("PAYMENT_VALUE") AS sum_payment_value,
                      MAX("PAYMENT_INSTALLMENTS") AS max_installments
               FROM "{DB_SCHEMA}"."ORDER_PAYMENTS" GROUP BY "ORDER_ID") p_sum
      ON p_sum."ORDER_ID"=o."ORDER_ID"
    LEFT JOIN (SELECT "ORDER_ID",COUNT(*) AS num_items
               FROM "{DB_SCHEMA}"."ORDER_ITEMS" GROUP BY "ORDER_ID") i_cnt
      ON i_cnt."ORDER_ID"=o."ORDER_ID"
    LEFT JOIN (SELECT oi."ORDER_ID",
                      AVG(COALESCE(pr."PRODUCT_PHOTOS_QTY",0)) AS avg_product_photos_qty,
                      AVG(COALESCE(pr."PRODUCT_DESCRIPTION_LENGHT",0)) AS avg_product_description_lenght,
                      AVG(COALESCE(pr."PRODUCT_WEIGHT_G",0)) AS avg_product_weight_g
               FROM "{DB_SCHEMA}"."ORDER_ITEMS" oi
               LEFT JOIN "{DB_SCHEMA}"."PRODUCTS" pr ON pr."PRODUCT_ID"=oi."PRODUCT_ID"
               GROUP BY oi."ORDER_ID") pr_agg
      ON pr_agg."ORDER_ID"=o."ORDER_ID"
    WHERE o."ORDER_PURCHASE_TIMESTAMP" IS NOT NULL
    LIMIT ?'''
    df=fetch_df(q,(limit,))
    for c in ["payment_value","avg_product_photos_qty","avg_product_description_lenght","avg_product_weight_g"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype("float32")
    for c in ["payment_installments","num_items","purchase_dayofweek","purchase_month","purchase_hour",
              "est_delivery_days","actual_delivery_days","delay_days","is_weekend"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype("int32")
    df["product_photos_qty"]=df["avg_product_photos_qty"].astype("float32")
    df["product_description_lenght"]=df["avg_product_description_lenght"].astype("float32")
    df["product_weight_g"]=df["avg_product_weight_g"].astype("float32")
    df["value_per_item"]=(df["payment_value"]/df["num_items"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    df["price_per_installment"]=(df["payment_value"]/df["payment_installments"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    df["late_delivery"]=df["late_delivery"].astype("int8")
    df["churn"]=df["churn"].astype("int8")
    return df

EXTS=[".csv",".csv.gz",".parquet"]

@st.cache_data(ttl=600)
def try_read(base,usecols=None):
    for ext in EXTS:
        p=os.path.join(".",base+ext)
        if os.path.exists(p):
            return pd.read_parquet(p,columns=usecols) if p.endswith(".parquet") else pd.read_csv(p,usecols=usecols)
    return None

@st.cache_data(ttl=600)
def load_from_csv(limit=300000):
    orders=try_read("olist_orders_dataset",usecols=["order_id","customer_id","order_status",
                                                    "order_purchase_timestamp","order_estimated_delivery_date",
                                                    "order_delivered_customer_date"])
    if orders is None: raise RuntimeError("orders file missing")
    items=try_read("olist_order_items_dataset",usecols=["order_id","product_id"]) or pd.DataFrame(columns=["order_id","product_id"])
    pays=try_read("olist_order_payments_dataset",usecols=["order_id","payment_installments","payment_value"]) or pd.DataFrame(columns=["order_id","payment_installments","payment_value"])
    prods=try_read("olist_products_dataset",usecols=["product_id","product_photos_qty","product_description_lenght","product_weight_g"]) or pd.DataFrame(columns=["product_id","product_photos_qty","product_description_lenght","product_weight_g"])
    items_cnt=items.groupby("order_id").size().rename("num_items").reset_index()
    pr_agg=items.merge(prods,on="product_id",how="left").groupby("order_id").agg(
        avg_product_photos_qty=("product_photos_qty","mean"),
        avg_product_description_lenght=("product_description_lenght","mean"),
        avg_product_weight_g=("product_weight_g","mean")).reset_index()
    pay_agg=pays.groupby("order_id").agg(payment_value=("payment_value","sum"),
                                         payment_installments=("payment_installments","max")).reset_index()
    if len(orders)>limit: orders=orders.sample(n=limit,random_state=42)
    df=orders.merge(items_cnt,on="order_id",how="left").merge(pr_agg,on="order_id",how="left").merge(pay_agg,on="order_id",how="left")
    dt=pd.to_datetime
    df["order_purchase_timestamp"]=dt(df["order_purchase_timestamp"],errors="coerce")
    df["order_estimated_delivery_date"]=dt(df["order_estimated_delivery_date"],errors="coerce")
    df["order_delivered_customer_date"]=dt(df["order_delivered_customer_date"],errors="coerce")
    df["purchase_dayofweek"]=df["order_purchase_timestamp"].dt.dayofweek.fillna(0).astype("int16")
    df["purchase_month"]=df["order_purchase_timestamp"].dt.month.fillna(1).astype("int16")-1
    df["purchase_hour"]=df["order_purchase_timestamp"].dt.hour.fillna(0).astype("int16")
    df["est_delivery_days"]=(df["order_estimated_delivery_date"]-df["order_purchase_timestamp"]).dt.days.fillna(0).astype("int32")
    df["actual_delivery_days"]=(df["order_delivered_customer_date"]-df["order_purchase_timestamp"]).dt.days.fillna(0).astype("int32")
    df["delay_days"]=(df["actual_delivery_days"]-df["est_delivery_days"]).clip(lower=0).astype("int32")
    df["is_weekend"]=df["purchase_dayofweek"].isin([5,6]).astype("int8")
    df["late_delivery"]=(df["order_delivered_customer_date"]>df["order_estimated_delivery_date"]).fillna(False).astype("int8")
    df["churn"]=df["order_status"].isin(["canceled","unavailable"]).astype("int8")
    for c in ["payment_value","avg_product_description_lenght","avg_product_weight_g","avg_product_photos_qty"]:
        df[c]=pd.to_numeric(df.get(c,0),errors="coerce").fillna(0).astype("float32")
    df["payment_installments"]=pd.to_numeric(df.get("payment_installments",0),errors="coerce").fillna(0).astype("int16")
    df["num_items"]=pd.to_numeric(df.get("num_items",0),errors="coerce").fillna(0).astype("int16")
    df["product_photos_qty"]=df["avg_product_photos_qty"].astype("float32")
    df["product_description_lenght"]=df["avg_product_description_lenght"].astype("float32")
    df["product_weight_g"]=df["avg_product_weight_g"].astype("float32")
    df["value_per_item"]=(df["payment_value"]/df["num_items"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    df["price_per_installment"]=(df["payment_value"]/df["payment_installments"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    return df

def split_xy(df):
    return df[FEATURES],df["late_delivery"].astype("int8"),df["churn"].astype("int8")

@st.cache_resource
def train_models(X,y1,y2):
    m1=HistGradientBoostingClassifier(max_depth=6,learning_rate=0.08,max_iter=200,l2_regularization=0.01)
    m2=HistGradientBoostingClassifier(max_depth=6,learning_rate=0.08,max_iter=200,l2_regularization=0.01)
    m1.fit(X,y1); m2.fit(X,y2)
    return m1,m2

def make_pdf(pred,f="report.pdf"):
    if FPDF is None: raise RuntimeError("FPDF not installed")
    pdf=FPDF(); pdf.add_page(); pdf.set_font("Arial",size=12)
    pdf.cell(200,10,"E-Commerce Prediction Report",ln=True,align="C"); pdf.ln(8)
    for k,v in pred.items(): pdf.cell(200,10,f"{k}: {v}",ln=True)
    pdf.output(f); return f

# --- UI ---
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>⚡ E-Commerce ML Suite</h1>",unsafe_allow_html=True)
c0,c1,c2=st.columns([2,2,2])
with c0: src=st.selectbox("Data Source",["HANA","CSV"],index=0)
with c1: limit=st.number_input("Row Limit",5000,1000000,150000,5000)
with c2: frac=st.slider("Train %",10,100,50,10)
load_start=time.time()
if src=="HANA":
    try: df=load_from_hana(int(limit)); source="HANA"
    except: df=load_from_csv(int(limit)); source="CSV"
else: df=load_from_csv(int(limit)); source="CSV"
st.caption(f"Source: {source} • Rows: {len(df):,} • {time.time()-load_start:.2f}s")
X,y_late,y_churn=split_xy(df)
tab1,tab2,tab3=st.tabs(["Dashboard","Prediction","Importance"])

with tab1:
    total=len(df); rev=float(df["payment_value"].sum())
    churn=(df["churn"]==1).mean()*100
    avg_items=df["num_items"].replace(0,np.nan).mean() or 0
    avg_days=df["actual_delivery_days"].replace(0,np.nan).mean() or 0
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Orders",f"{total:,}")
    c2.metric("Revenue",f"{rev:,.2f}")
    c3.metric("Churn %",f"{churn:.2f}")
    c4.metric("Avg Items",f"{avg_items:.2f}")
    c5.metric("Avg Delivery Days",f"{avg_days:.1f}")
    st.bar_chart(df["order_status"].value_counts())

with tab2:
    if len(X)>0:
        sub=X.sample(frac=frac/100,random_state=42); idx=sub.index
        y1,y2=y_late.loc[idx],y_churn.loc[idx]
    else: sub,y1,y2=X,y_late,y_churn
    with st.spinner("Training..."): m1,m2=train_models(sub,y1,y2)
    st.subheader("Enter Order")
    pv=st.number_input("Payment Value",0.0,50000.0,200.0)
    pi=st.number_input("Installments",0,60,4)
    ppq=st.number_input("Photos Qty",0,50,5)
    pdl=st.number_input("Description Len",0,10000,1000)
    pw=st.number_input("Weight (g)",0,100000,1000)
    pdw=st.selectbox("Day of Week",list(range(7)),0)
    defaults=df[FEATURES].median(numeric_only=True).fillna(0).to_dict() if not df.empty else {c:0 for c in FEATURES}
    row={c:defaults.get(c,0) for c in FEATURES}
    row.update({"payment_value":float(pv),"payment_installments":int(pi),
                "product_photos_qty":float(ppq),"product_description_lenght":float(pdl),
                "product_weight_g":float(pw),"purchase_dayofweek":int(pdw)})
    row["value_per_item"]=row["payment_value"]/row.get("num_items",1)
    row["price_per_installment"]=row["payment_value"]/max(row.get("payment_installments",1),1)
    inp=pd.DataFrame([[row[c] for c in FEATURES]],columns=FEATURES)
    if st.button("Predict"):
        st.session_state["pred"]={"is_late":int(m1.predict(inp)[0]),
                                  "will_churn":int(m2.predict(inp)[0])}
    if "pred" in st.session_state:
        r=st.session_state["pred"]
        c1,c2=st.columns(2)
        c1.metric("Delivery","Late" if r["is_late"] else "On Time")
        c2.metric("Churn","Yes" if r["will_churn"] else "No")
        if st.button("PDF",disabled=FPDF is None):
            f=make_pdf({"Delivery":"Late" if r["is_late"] else "On Time",
                        "Churn":"Yes" if r["will_churn"] else "No"})
            with open(f,"rb") as fobj:
                st.download_button("Download",data=fobj,file_name=f)

with tab3:
    try:
        n=min(10000,len(df))
        samp=df.sample(n=n,random_state=42) if n>10 else df
        fi=HistGradientBoostingClassifier().fit(samp[FEATURES],samp["churn"]).feature_importances_
        st.bar_chart(pd.Series(fi,index=FEATURES).sort_values(ascending=False))
    except: pass
