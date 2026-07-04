import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os

# Set page config
st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🏦", layout="wide")

# Map clusters to human-readable labels (Based on our AI Profiling)
CLUSTER_LABELS = {
    0: "🌟 High Engagement (Wealthy, Long Calls)",
    1: "⚠️ Low Engagement (Low Balance, Short Calls)",
    2: "🔄 Moderate Engagement (Returning/Loyal Customers)"
}

# --- Load Models ---
@st.cache_resource
def load_models():
    # Streamlit runs from the frontend folder, so models are in ../models
    scaler_path = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
    kmeans_path = os.path.join(os.path.dirname(__file__), '../models/kmeans_model.pkl')
    
    try:
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        return scaler, kmeans
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

scaler, kmeans = load_models()

# --- Sidebar: User Input ---
st.sidebar.header("🔍 Customer Predictor")
st.sidebar.markdown("Enter customer details below to predict their segment using the AI model.")

age = st.sidebar.slider("Age", 18, 100, 35)
balance = st.sidebar.number_input("Yearly Balance (EUR)", -5000, 100000, 1000)
duration = st.sidebar.slider("Last Call Duration (seconds)", 0, 4000, 200)
campaign = st.sidebar.slider("Calls During This Campaign", 1, 50, 2)
previous = st.sidebar.slider("Calls Before This Campaign", 0, 50, 0)

predict_btn = st.sidebar.button("Predict Segment", type="primary")

# --- Main Dashboard ---
st.title("🏦 Bank Customer Segmentation Dashboard")
st.markdown("Welcome to the final interactive dashboard for the thesis project. Use the sidebar to predict a new customer's segment, or explore the analytics below.")

tab1, tab2 = st.tabs(["🔮 Live Prediction", "📊 Cluster Analytics"])

with tab1:
    st.subheader("Customer Prediction Result")
    
    if predict_btn:
        if scaler and kmeans:
            # 1. Create input DataFrame
            input_data = pd.DataFrame({
                'age': [age],
                'balance': [balance],
                'duration': [duration],
                'campaign': [campaign],
                'previous': [previous]
            })
            
            # 2. Scale the input
            input_scaled = scaler.transform(input_data)
            
            # 3. Predict the cluster
            cluster_id = kmeans.predict(input_scaled)[0]
            label = CLUSTER_LABELS.get(cluster_id, "Unknown")
            
            # 4. Display visually
            if cluster_id == 0:
                st.success(f"### Predicted Segment: {label}")
                st.markdown("This customer is **highly engaged**. They tend to have high balances and stay on the phone for a long time. **Action:** Offer them premium financial products.")
            elif cluster_id == 1:
                st.error(f"### Predicted Segment: {label}")
                st.markdown("This customer has **low engagement**. They have lower balances and hang up quickly. **Action:** Minimize contact to reduce operational costs.")
            else:
                st.warning(f"### Predicted Segment: {label}")
                st.markdown("This customer has **moderate engagement**. They have been contacted frequently in the past. **Action:** Focus on relationship retention.")
                
            st.markdown("---")
            st.markdown("#### Input Data Sent to Model:")
            st.dataframe(input_data)
        else:
            st.error("Models not found! Make sure you ran all notebooks.")
    else:
        st.info("Adjust the sliders in the sidebar and click 'Predict Segment'.")

with tab2:
    st.subheader("Artificial Intelligence Cluster Analysis")
    st.markdown("These charts prove the mathematical validity of our 3 customer segments.")
    
    col1, col2 = st.columns(2)
    
    # Load images dynamically
    fig_dir = os.path.join(os.path.dirname(__file__), '../figures')
    
    with col1:
        st.markdown("#### 1. Cluster Personas (Radar Chart)")
        radar_path = os.path.join(fig_dir, 'cluster_radar_chart.png')
        if os.path.exists(radar_path):
            st.image(Image.open(radar_path), use_column_width=True)
            
        st.markdown("#### 3. Mathematical Proof (Elbow Curve)")
        elbow_path = os.path.join(fig_dir, 'elbow_curve.png')
        if os.path.exists(elbow_path):
            st.image(Image.open(elbow_path), use_column_width=True)

    with col2:
        st.markdown("#### 2. Business Proof (Conversion Rate)")
        conv_path = os.path.join(fig_dir, 'conversion_rate_by_cluster.png')
        if os.path.exists(conv_path):
            st.image(Image.open(conv_path), use_column_width=True)
            
        st.markdown("#### 4. 2D Visual Map (PCA Scatter)")
        pca_path = os.path.join(fig_dir, 'pca_scatter_plot.png')
        if os.path.exists(pca_path):
            st.image(Image.open(pca_path), use_column_width=True)
