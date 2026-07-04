# 🏦 Bank Customer Intelligence Platform

## Overview
This project is an advanced Machine Learning pipeline designed for the banking sector. It utilizes **K-Means Clustering** to segment bank customers based on their financial profiles and behavioral engagement metrics. The goal is to optimize telemarketing campaigns, improve conversion rates, and allocate banking resources efficiently.

The project transforms raw, unstructured customer data into a highly visual, interactive **Streamlit Dashboard** that relationship managers can use in real-time to predict a customer's engagement level.

## 🌟 Key Features
- **Unsupervised Machine Learning**: K-Means clustering algorithm trained on 45,211 customers.
- **Automated Data Pipeline**: Handles outlier capping, standardization, and missing value management.
- **Native USD Architecture**: Models trained natively on USD financial metrics.
- **Real-Time Interactive Dashboard**: Built with Streamlit, featuring a modern UI, real-time predictions, and user history tracking.
- **Business Strategy Engine**: Provides actionable recommendations (e.g., "Invest Heavily", "Automate & Monitor") based on predictive clustering.

## 📂 Project Structure
```text
├── data/
│   └── raw/               # Raw datasets (bank-full_USD.csv)
├── models/                # Serialized ML models (scaler.pkl, kmeans_model.pkl)
├── notebooks/             # Jupyter notebooks for EDA and model training
├── frontend/              # Streamlit application (premium_app.py)
├── reports/               # Markdown interpretation and evaluation reports
├── figures/               # Generated charts and visualizations
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🚀 How to Run the Dashboard
1. Ensure Python 3.9+ is installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   cd frontend
   streamlit run premium_app.py
   ```

## 📊 The 3 Customer Segments
1. **High-Engagement**: High balances (~$4,700), high patience on calls (nearly 10 mins). These are premium targets.
2. **Moderate-Engagement**: Loyal customers (~$1,400) who require nurturing to graduate to premium status.
3. **Low-Engagement**: The majority segment (~$700) with short attention spans. Best suited for automated digital marketing.

## 🎓 Academic Context
This project was developed as a Bachelor's Degree Graduation Project to demonstrate end-to-end Machine Learning deployment, from raw data engineering to a production-ready frontend interface.
