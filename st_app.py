import streamlit as st
import pandas as pd
from hdbcli import dbapi

# Load secrets
hana = st.secrets["hana"]

HANA_CONFIG = {
    "address": hana["address"],
    "port": int(hana["port"]),
    "user": hana["user"],
    "password": hana["password"],
    "encrypt": hana["encrypt"],
    "sslValidateCertificate": hana["sslValidateCertificate"]
}
SCHEMA_NAME = hana["schema"]

@st.cache_resource
def get_connection():
    return dbapi.connect(**HANA_CONFIG)

@st.cache_data
def load_data():
    conn = get_connection()
    cursor = conn.cursor()
    
    def fetch_table(table_name):
        cursor.execute(f'SELECT * FROM "{SCHEMA_NAME}"."{table_name}" LIMIT 10')
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=cols)
    
    return {
        "customers": fetch_table("CUSTOMERS"),
        "orders": fetch_table("ORDERS"),
        "order_items": fetch_table("ORDER_ITEMS"),
        "payments": fetch_table("ORDER_PAYMENTS"),
        "reviews": fetch_table("ORDER_REVIEWS"),
        "products": fetch_table("PRODUCTS"),
        "sellers": fetch_table("SELLERS"),
        "categories": fetch_table("CATEGORY_TRANSLATION")
    }
st.sidebar.write("🔗 Testing Connection")
try:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT CURRENT_USER, CURRENT_SCHEMA FROM DUMMY")
    st.success(f"Connected as {cursor.fetchall()}")
except Exception as e:
    st.error(f"Connection failed: {e}")
