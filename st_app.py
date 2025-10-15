import os
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    from hdbcli import dbapi
except Exception:
    dbapi=None
try:
    from fpdf import FPDF
except Exception:
    FPDF=None
st.set_page_config(page_title="E-Commerce ML Suite (Optimized)",layout="wide")

def _cfg():
    s=st.secrets.get("hana",{})
    return{"address":s.get("address")or os.getenv("HANA_ADDRESS"),"port":int(s.get("port",os.getenv("HANA_PORT",443))),"user":s.get("user")or os.getenv("HANA_USER","DBADMIN"),"password":s.get("password")or os.getenv("HANA_PASSWORD"),"schema":s.get("schema")or os.getenv("HANA_SCHEMA","ECOMM_BRAZIL"),"encrypt":bool(s.get("encrypt",True)),"sslValidateCertificate":bool(s.get("sslValidateCertificate",False))}
CFG=_cfg()
SCHEMA=CFG["schema"]
@st.cache_resource(show_spinner=False)
def get_connection():
    if dbapi is None or CFG["address"] is None: raise RuntimeError("HANA client or address not available")
    return dbapi.connect(address=CFG["address"],port=CFG["port"],user=CFG["user"],password=CFG["password"],encrypt=CFG["encrypt"],sslValidateCertificate=CFG["sslValidateCertificate"],timeout=10)
@st.cache_data(show_spinner=False,ttl=600)
def fetch_df(query:str,params:tuple=()):
    conn=get_connection();cur=conn.cursor();cur.execute(query,params)
    cols=[d[0].lower() for d in cur.description];rows=cur.fetchall()
    return pd.DataFrame(rows,columns=cols)
NEEDED_COLS=["order_id","payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g","purchase_dayofweek","late_delivery","churn","order_status"]
FEATURE_COLS=["payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g","purchase_dayofweek","num_items","avg_product_weight_g","avg_product_photos_qty","avg_product_description_lenght","value_per_item","price_per_installment","est_delivery_days","actual_delivery_days","delay_days","is_weekend","purchase_month","purchase_hour"]
@st.cache_data(show_spinner=False,ttl=600)
def load_from_hana(limit:int=200_000):
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
        CASE WHEN o."ORDER_DELIVERED_CUSTOMER_DATE">o."ORDER_ESTIMATED_DELIVERY_DATE" THEN DAYS_BETWEEN(o."ORDER_ESTIMATED_DELIVERY_DATE",o."ORDER_DELIVERED_CUSTOMER_DATE") ELSE 0 END AS delay_days,
        CASE WHEN (TO_INTEGER(DAYOFWEEK(o."ORDER_PURCHASE_TIMESTAMP"))-1) IN (5,6) THEN 1 ELSE 0 END AS is_weekend,
        CASE WHEN o."ORDER_DELIVERED_CUSTOMER_DATE">o."ORDER_ESTIMATED_DELIVERY_DATE" THEN 1 ELSE 0 END AS late_delivery,
        CASE WHEN o."ORDER_STATUS" IN ('canceled','unavailable') THEN 1 ELSE 0 END AS churn,
        o."ORDER_STATUS" AS order_status
    FROM "{SCHEMA}"."ORDERS" o
    LEFT JOIN(
        SELECT "ORDER_ID",SUM("PAYMENT_VALUE") AS sum_payment_value,MAX("PAYMENT_INSTALLMENTS") AS max_installments
        FROM "{SCHEMA}"."ORDER_PAYMENTS" GROUP BY "ORDER_ID"
    ) p_sum ON p_sum."ORDER_ID"=o."ORDER_ID"
    LEFT JOIN(
        SELECT "ORDER_ID",COUNT(*) AS num_items FROM "{SCHEMA}"."ORDER_ITEMS" GROUP BY "ORDER_ID"
    ) i_cnt ON i_cnt."ORDER_ID"=o."ORDER_ID"
    LEFT JOIN(
        SELECT oi."ORDER_ID",AVG(COALESCE(pr."PRODUCT_PHOTOS_QTY",0)) AS avg_product_photos_qty,AVG(COALESCE(pr."PRODUCT_DESCRIPTION_LENGHT",0)) AS avg_product_description_lenght,AVG(COALESCE(pr."PRODUCT_WEIGHT_G",0)) AS avg_product_weight_g
        FROM "{SCHEMA}"."ORDER_ITEMS" oi LEFT JOIN "{SCHEMA}"."PRODUCTS" pr ON pr."PRODUCT_ID"=oi."PRODUCT_ID" GROUP BY oi."ORDER_ID"
    ) pr_agg ON pr_agg."ORDER_ID"=o."ORDER_ID"
    WHERE o."ORDER_PURCHASE_TIMESTAMP" IS NOT NULL LIMIT ?
    '''
    df=fetch_df(q,(limit,))
    for c in ["payment_value","avg_product_photos_qty","avg_product_description_lenght","avg_product_weight_g"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype("float32")
    for c in ["payment_installments","num_items","purchase_dayofweek","purchase_month","purchase_hour","est_delivery_days","actual_delivery_days","delay_days","is_weekend"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0).astype("int32")
    df["product_photos_qty"]=df["avg_product_photos_qty"].astype("float32")
    df["product_description_lenght"]=df["avg_product_description_lenght"].astype("float32")
    df["product_weight_g"]=df["avg_product_weight_g"].astype("float32")
    df["value_per_item"]=(df["payment_value"]/df["num_items"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    df["price_per_installment"]=(df["payment_value"]/df["payment_installments"].replace(0,np.nan)).fillna(df["payment_value"]).astype("float32")
    df["late_delivery"]=pd.to_numeric(df["late_delivery"],errors="coerce").fillna(0).astype("int8")
    df["churn"]=pd.to_numeric(df["churn"],errors="coerce").fillna(0).astype("int8")
    return df
EXTS=[".csv",".csv.gz",".parquet"]
@st.cache_data(show_spinner=False,ttl=600)
def _try_read_alias(base,usecols=None):
    for ext in EXTS:
        p=os.path.join(".",base+ext)
        if os.path.exists(p):
            if p.endswith(".parquet"): return pd.read_parquet(p,columns=usecols)
            return pd.read_csv(p,usecols=usecols)
    return None
@st.cache_data(show_spinner=False,ttl=600)
def load_from_csv(limit_rows:int=300_000):
    orders=_try_read_alias("olist_orders_dataset",usecols=["order_id","customer_id","order_status","order_purchase_timestamp","order_estimated_delivery_date","order_delivered_customer_date"])
    if orders is None: raise RuntimeError("olist_orders_dataset not found")
    items=_try_read_alias("olist_order_items_dataset",usecols=["order_id","product_id"]) or pd.DataFrame(columns=["order_id","product_id"])
    pays=_try_read_alias("olist_order_payments_dataset",usecols=["order_id","payment_installments","payment_value"]) or pd.DataFrame(columns=["order_id","payment_installments","payment_value"])
    prods=_try_read_alias("olist_products_dataset",usecols=["product_id","product_photos_qty","product_description_lenght","product_weight_g"]) or pd.DataFrame(columns=["product_id","product_photos_qty","product_description_lenght","product_weight_g"])
    items_cnt=items.groupby("order_id").size().rename("num_items").reset_index()
    pr_agg=items.merge(prods,on="product_id",how="left").groupby("order_id").agg(avg_product_photos_qty=("product_photos_qty","mean"),avg_product_description_lenght=("product_description_lenght","mean"),avg_product_weight_g=("product_weight_g","mean")).reset_index()
    pay_agg=pays.groupby("order_id").agg(payment_value=("payment_value","sum"),payment_installments=("payment_installments","max")).reset_index()
    if len(orders)>limit_rows: orders=orders.sample(n=limit_rows,random_state=42)
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
    base=["order_id","payment_value","payment_installments","product_photos_qty","product_description_lenght","product_weight_g","purchase_dayofweek","late_delivery","churn","order_status"]
    extras=["num_items","avg_product_weight_g","avg_product_photos_qty","avg_product_description_lenght","value_per_item","price_per_installment","est_delivery_days","actual_delivery_days","delay_days","is_weekend","purchase_month","purchase_hour"]
    cols=base+extras
    for c in extras:
        if c not in df.columns: df[c]=0
    return df[cols]

def to_Xy(df:pd.DataFrame):
    X=df[FEATURE_COLS]
    y_late=df["late_delivery"].astype("int8")
    y_churn=df["churn"].astype("int8")
    return X,y_late,y_churn
@st.cache_resource(show_spinner=False)
def train_models(X,y_late,y_churn):
    clf_late=HistGradientBoostingClassifier(max_depth=6,learning_rate=0.08,max_iter=200,l2_regularization=0.01)
    clf_churn=HistGradientBoostingClassifier(max_depth=6,learning_rate=0.08,max_iter=200,l2_regularization=0.01)
    clf_late.fit(X,y_late);clf_churn.fit(X,y_churn)
    return clf_late,clf_churn

def export_pdf(pred:dict,file_name:str="report.pdf"):
    if FPDF is None: raise RuntimeError("FPDF not installed")
    pdf=FPDF();pdf.add_page();pdf.set_font("Arial",size=12)
    pdf.cell(200,10,txt="E-Commerce Prediction Report",ln=True,align="C");pdf.ln(8)
    for k,v in pred.items(): pdf.cell(200,10,txt=f"{k}: {v}",ln=True)
    pdf.output(file_name);return file_name
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>⚡ E-Commerce ML Suite — Optimized</h1>",unsafe_allow_html=True)
c0,c1,c2=st.columns([2,2,2])
with c0:
    data_source=st.selectbox("Data Source Preference",["HANA (fast SQL)","CSV (fallback)"],index=0)
with c1:
    row_limit=st.number_input("Row Limit (pull fewer rows = faster)",min_value=5_000,max_value=1_000_000,value=150_000,step=5_000)
with c2:
    sample_for_training=st.slider("Train on % of pulled rows",min_value=10,max_value=100,value=50,step=10)
st.caption("Tip: reduce Row Limit and Train % for instant responsiveness.")
load_start=time.time()
if data_source.startswith("HANA"):
    try:
        df=load_from_hana(limit=int(row_limit));source="SAP HANA Cloud"
    except Exception:
        df=load_from_csv(limit_rows=int(row_limit));source="CSV (Olist)"
elif data_source.startswith("CSV"):
    df=load_from_csv(limit_rows=int(row_limit));source="CSV (Olist)"
else:
    raise RuntimeError("Unsupported data source selection")
load_secs=time.time()-load_start
st.caption(f"Data Source: {source} • Rows: {len(df):,} • Loaded in {load_secs:.2f}s")
X,y_late,y_churn=to_Xy(df)
tab1,tab2,tab3=st.tabs(["📊 Dashboard","🤖 Prediction","📈 Feature Importance"])
with tab1:
    st.subheader("KPI Overview (fast)")
    total_orders=len(df);revenue=float(df["payment_value"].sum());churn_rate=float((df["churn"]==1).mean()*100)
    avg_items=float(df["num_items"].replace(0,np.nan).mean() or 0);avg_delivery_days=float(df["actual_delivery_days"].replace(0,np.nan).mean() or 0)
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Total Orders",f"{total_orders:,}");c2.metric("Revenue (R$)",f"{revenue:,.2f}");c3.metric("Churn %",f"{churn_rate:0.2f}");c4.metric("Avg Items/Order",f"{avg_items:0.2f}");c5.metric("Avg Delivery Days",f"{avg_delivery_days:0.1f}")
    if "order_status" in df.columns:
        vc=df["order_status"].value_counts();st.bar_chart(vc)
with tab2:
    if len(X)>0:
        frac=max(0.1,min(1.0,sample_for_training/100.0));X_train=X.sample(frac=frac,random_state=42)
        idx=X_train.index;y_lat_tr=y_late.loc[idx];y_chu_tr=y_churn.loc[idx]
    else:
        X_train,y_lat_tr,y_chu_tr=X,y_late,y_churn
    with st.spinner("Training compact models..."):
        clf_late,clf_churn=train_models(X_train,y_lat_tr,y_chu_tr)
    st.subheader("Enter Order Details")
    c1,c2,c3=st.columns(3)
    with c1:
        payment_value=st.number_input("Payment Value",0.0,50000.0,200.0)
    with c2:
        payment_installments=st.number_input("Installments",0,60,4)
    with c3:
        product_photos_qty=st.number_input("Photos Qty",0,50,5)
    c4,c5,c6=st.columns(3)
    with c4:
        product_description_lenght=st.number_input("Description Length",0,10000,1000)
    with c5:
        product_weight_g=st.number_input("Weight (g)",0,100000,1000)
    with c6:
        purchase_dayofweek=st.selectbox("Day of Week (0=Mon)",list(range(7)),index=0)
    with st.expander("Advanced features (optional)"):
        a1,a2,a3=st.columns(3)
        with a1:
            num_items=st.number_input("Num items",0,100,1)
            est_delivery_days=st.number_input("Est. delivery days",0,120,5)
            is_weekend=st.selectbox("Purchase on weekend?",[0,1],index=0)
        with a2:
            avg_product_weight_g=st.number_input("Avg product weight (g)",0,200000,int(product_weight_g))
            actual_delivery_days=st.number_input("Actual delivery days",0,120,5)
            purchase_month=st.number_input("Purchase month (0-11)",0,11,0)
        with a3:
            avg_product_photos_qty=st.number_input("Avg product photos qty",0,50,int(product_photos_qty))
            avg_product_description_lenght=st.number_input("Avg description length",0,10000,int(product_description_lenght))
            purchase_hour=st.number_input("Purchase hour (0-23)",0,23,12)
    feature_medians=df[FEATURE_COLS].median(numeric_only=True).fillna(0).to_dict() if not df.empty else {c:0 for c in FEATURE_COLS}
    row={c:feature_medians.get(c,0) for c in FEATURE_COLS}
    row.update({"payment_value":float(payment_value),"payment_installments":int(payment_installments),"product_photos_qty":float(product_photos_qty),"product_description_lenght":float(product_description_lenght),"product_weight_g":float(product_weight_g),"purchase_dayofweek":int(purchase_dayofweek)})
    for k,v in {"num_items":locals().get("num_items"),"avg_product_weight_g":locals().get("avg_product_weight_g"),"avg_product_photos_qty":locals().get("avg_product_photos_qty"),"avg_product_description_lenght":locals().get("avg_product_description_lenght"),"est_delivery_days":locals().get("est_delivery_days"),"actual_delivery_days":locals().get("actual_delivery_days"),"is_weekend":locals().get("is_weekend"),"purchase_month":locals().get("purchase_month"),"purchase_hour":locals().get("purchase_hour")}.items():
        if v is not None: row[k]=float(v)
    if row.get("num_items",0)>0: row["value_per_item"]=float(row["payment_value"])/float(row["num_items"])
    else: row["value_per_item"]=float(row["payment_value"])
    if row.get("payment_installments",0)>0: row["price_per_installment"]=float(row["payment_value"])/float(row["payment_installments"])
    else: row["price_per_installment"]=float(row["payment_value"])
    input_df=pd.DataFrame([[row[c] for c in FEATURE_COLS]],columns=FEATURE_COLS)
    if st.button("Predict",use_container_width=True):
        pred_late=int(clf_late.predict(input_df)[0]);pred_churn=int(clf_churn.predict(input_df)[0])
        st.session_state["prediction"]={"is_late":pred_late,"will_churn":pred_churn}
    if "prediction" in st.session_state:
        r=st.session_state["prediction"];col1,col2=st.columns(2)
        col1.metric("🚚 Delivery","Late" if r["is_late"] else "On Time")
        col2.metric("📉 Churn","Yes" if r["will_churn"] else "No")
    st.divider()
    if "prediction" in st.session_state:
        r=st.session_state["prediction"]
        if FPDF is None: st.info("Install FPDF to enable PDF export: pip install fpdf2")
        disabled=FPDF is None
        if st.button("Generate PDF Report",use_container_width=True,disabled=disabled):
            try:
                path=export_pdf({"Delivery":"Late" if r["is_late"] else "On Time","Churn":"Yes" if r["will_churn"] else "No"})
                with open(path,"rb") as f: st.download_button("Download Report",data=f,file_name=path)
            except Exception as e:
                st.error(f"PDF export failed: {e}")
with tab3:
    st.subheader("Feature Importance (approx.)")
    try:
        n=min(10000,len(df))
        if n>10:
            samp=df.sample(n=n,random_state=42);Xs,ys=samp[FEATURE_COLS],samp["churn"].astype("int8")
            tmp_clf=HistGradientBoostingClassifier().fit(Xs,ys);fi_vals=getattr(tmp_clf,"feature_importances_",None)
            if fi_vals is not None: fi=pd.Series(fi_vals,index=FEATURE_COLS).sort_values(ascending=False)
            else: raise AttributeError("No feature_importances_")
        else:
            fi=pd.Series({c:0.0 for c in FEATURE_COLS})
    except Exception:
        def _corr(a,b):
            try: return float(np.corrcoef(a,b)[0,1])
            except Exception: return 0.0
        fi=pd.Series({c:abs(_corr(df[c].values,df["churn"].values)) if df[c].std()>0 else 0.0 for c in FEATURE_COLS}).fillna(0).sort_values(ascending=False)
    st.bar_chart(fi)
