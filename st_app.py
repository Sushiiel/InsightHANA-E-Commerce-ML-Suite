import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st
import joblib
import os
from fpdf import FPDF
from hdbcli import dbapi

SCHEMA_NAME="ECOMM_BRAZIL"
HANA_CONFIG={
    'address':'5b88e881-d6dd-4f02-8268-0cb145f3e415.hana.trial-us10.hanacloud.ondemand.com',
    'port':443,
    'user':'DBADMIN',
    'password':'595162sushiielA@',
    'encrypt':True,
    'sslValidateCertificate':False
}

@st.cache_resource
def get_connection():
    return dbapi.connect(**HANA_CONFIG)

@st.cache_data
def load_data():
    conn=get_connection()
    cursor=conn.cursor()
    def fetch_table(table_name):
        cursor.execute(f"SELECT * FROM {SCHEMA_NAME}.{table_name}")
        cols=[desc[0].lower() for desc in cursor.description]
        rows=cursor.fetchall()
        return pd.DataFrame(rows,columns=cols)
    return {
        'customers':fetch_table('customers'),
        'geolocation':fetch_table('geolocation'),
        'orders':fetch_table('orders'),
        'order_items':fetch_table('order_items'),
        'payments':fetch_table('order_payments'),
        'reviews':fetch_table('order_reviews'),
        'products':fetch_table('products'),
        'sellers':fetch_table('sellers'),
        'categories':fetch_table('category_translation')
    }

def prepare_features(d):
    df=d['orders']\
        .merge(d['order_items'],on='order_id',how='left')\
        .merge(d['payments'],on='order_id',how='left')\
        .merge(d['reviews'],on='order_id',how='left')\
        .merge(d['customers'],on='customer_id',how='left')\
        .merge(d['products'],on='product_id',how='left')\
        .merge(d['sellers'],on='seller_id',how='left')\
        .merge(d['categories'],on='product_category_name',how='left')
    df['order_purchase_timestamp']=pd.to_datetime(df['order_purchase_timestamp'],errors='coerce')
    df['order_delivered_customer_date']=pd.to_datetime(df['order_delivered_customer_date'],errors='coerce')
    df['order_estimated_delivery_date']=pd.to_datetime(df['order_estimated_delivery_date'],errors='coerce')
    df['review_score']=pd.to_numeric(df['review_score'],errors='coerce')
    df['payment_value']=pd.to_numeric(df['payment_value'],errors='coerce')
    df['payment_installments']=pd.to_numeric(df['payment_installments'],errors='coerce')
    df['product_photos_qty']=pd.to_numeric(df['product_photos_qty'],errors='coerce').fillna(0)
    df['product_description_lenght']=pd.to_numeric(df['product_description_lenght'],errors='coerce').fillna(0)
    df['product_weight_g']=pd.to_numeric(df['product_weight_g'],errors='coerce').fillna(0)
    df['purchase_dayofweek']=df['order_purchase_timestamp'].dt.dayofweek
    df['late_delivery']=(df['order_delivered_customer_date']>df['order_estimated_delivery_date']).astype(int)
    df['churn']=df['order_status'].isin(['canceled','unavailable']).astype(int)
    df=df.dropna(subset=['review_score','payment_value'])
    X=df[[
        'payment_value','payment_installments','product_photos_qty',
        'product_description_lenght','product_weight_g','purchase_dayofweek'
    ]]
    y_review=df['review_score']
    y_late=df['late_delivery']
    y_churn=df['churn']
    return X,y_review,y_late,y_churn

def get_or_train_model(path,model_type,X,y):
    if os.path.exists(path):
        return joblib.load(path)
    model=model_type(n_estimators=100,random_state=42)
    model.fit(X,y)
    joblib.dump(model,path)
    return model

def export_to_pdf(predictions:dict,file_name="report.pdf"):
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial",size=12)
    pdf.cell(200,10,txt="E-Commerce Prediction Report",ln=True,align="C")
    pdf.ln(10)
    for label,value in predictions.items():
        pdf.cell(200,10,txt=f"{label}: {value}",ln=True)
    pdf.output(file_name)

def main():
    st.set_page_config(page_title="E-Commerce ML Dashboard",layout="centered")
    st.title("📦 Intelligent E-Commerce Prediction Engine")
    menu=st.sidebar.selectbox("📋 Menu",["📊 View Sample Data","📈 Predict Customer Behavior"])

    data=load_data()

    if menu=="📊 View Sample Data":
        st.subheader("🔍 Explore Tables from SAP HANA")
        table_name=st.selectbox("Select a table",list(data.keys()))
        st.dataframe(data[table_name].head(10))

    elif menu=="📈 Predict Customer Behavior":
        with st.spinner("🔄 Training models..."):
            X,y_review,y_late,y_churn=prepare_features(data)
            review_model=get_or_train_model("review_model.pkl",RandomForestRegressor,X,y_review)
            late_model=get_or_train_model("late_model.pkl",RandomForestClassifier,X,y_late)
            churn_model=get_or_train_model("churn_model.pkl",RandomForestClassifier,X,y_churn)
        st.subheader("📝 Enter Order Details")
        payment_value=st.slider('Payment Value (R$)',0.0,5000.0,200.0)
        payment_installments=st.slider('Installments',1,24,4)
        product_photos_qty=st.slider('Product Photos Qty',0,20,5)
        product_description_lenght=st.slider('Description Length',0,4000,1000)
        product_weight_g=st.slider('Product Weight (g)',0,10000,1000)
        purchase_dayofweek=st.selectbox('Day of Week (0=Mon, 6=Sun)',list(range(7)))
        input_df=pd.DataFrame([[
            payment_value,payment_installments,product_photos_qty,
            product_description_lenght,product_weight_g,purchase_dayofweek
        ]],columns=X.columns)
        if st.button("🔍 Predict"):
          st.session_state.prediction_result = {
             "review_score": round(review_model.predict(input_df)[0], 2),
             "is_late": late_model.predict(input_df)[0],
             "will_churn": churn_model.predict(input_df)[0]
    }

        if "prediction_result" in st.session_state:
          result = st.session_state.prediction_result
          st.success(f"⭐ Predicted Review Score: {result['review_score']}")
          st.info(f"🚚 Delivery: {'Late' if result['is_late'] else 'On Time'}")
          st.warning(f"📉 Churn Risk: {'Yes' if result['will_churn'] else 'No'}")

          if st.button("📄 Download PDF Report"):
             export_to_pdf({
                 "Predicted Review Score": result["review_score"],
                 "Delivery Status": "Late" if result["is_late"] else "On Time",
                 "Churn Risk": "Yes" if result["will_churn"] else "No"
        })
             st.success("✅ PDF saved as report.pdf in current directory")

if __name__=="__main__":
    main()
