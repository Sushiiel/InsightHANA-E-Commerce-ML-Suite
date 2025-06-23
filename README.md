# 🛍️ Intelligent E-Commerce ML Dashboard

🚀 **Live Demo**: [Try the App](https://geo-aware-logistics-planner-using-sap-hana-cloud.streamlit.app/)  

A Streamlit-powered machine learning dashboard that connects to SAP HANA Cloud to provide actionable insights and predictions on customer behavior in an e-commerce platform.

## 📌 Overview

This project integrates real-time data from SAP HANA and applies machine learning techniques to:
- Predict customer **review scores**
- Forecast **delivery delays**
- Detect potential **customer churn**
- Explore sample data directly from the SAP HANA schema

## 📊 Key Features

- 🔌 **Live SAP HANA Connection**: Fetches tables directly from a cloud SAP HANA instance (`ECOMM_BRAZIL` schema)
- 🤖 **ML-Based Prediction**:
  - Predicts review score (regression)
  - Predicts if delivery will be late (classification)
  - Predicts customer churn (classification)
- 📄 **Downloadable PDF Report**: Summarizes predictions in a neatly formatted PDF
- 📋 **View Sample Data**: Explore the contents of each table before predictions
- 🧠 **Model Persistence**: Trained models are saved and reused to avoid retraining on every run

## 📚 Dataset

Data used in this project is based on the **Brazilian E-Commerce Public Dataset** available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Tables used:
- `customers`
- `geolocation`
- `orders`
- `order_items`
- `order_payments`
- `order_reviews`
- `products`
- `sellers`
- `category_translation`

## 🛠️ Tech Stack

| Component         | Tool/Library                  |
|------------------|-------------------------------|
| UI Framework     | Streamlit                     |
| ML Models        | scikit-learn (Random Forest)  |
| Data Handling    | pandas                        |
| SAP HANA Access  | hdbcli (Python HANA driver)   |
| Model Storage    | joblib                        |
| PDF Generation   | fpdf                          |
| Deployment       | Localhost (can be moved to Cloud or SAP BTP) |

## 🧠 ML Features Used

The model is trained on the following engineered features:
- `payment_value`
- `payment_installments`
- `product_photos_qty`
- `product_description_length`
- `product_weight_g`
- `purchase_dayofweek`

These features were selected based on their impact on user experience and customer satisfaction metrics.

## 🖥️ How to Run

1. Clone the repo
```bash
git clone https://github.com/yourusername/ecomm-ml-dashboard.git
cd ecomm-ml-dashboard
