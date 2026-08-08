import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import os
import urllib.request


# --- Page Layout & Styling Configuration ---
st.set_page_config(page_title="Multi-Disease Chest X-Ray Diagnostic AI", layout="wide", page_icon="🫁")

# Initialize Session State Variables to save Patient Logs without losing data on click
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []

# Custom Professional UI Injection (Gives it a modern dark clinical aesthetic)
st.markdown("""
    <style>
        .main { background-color: #0f172a; color: #f8fafc; }
        .stButton>button { 
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); 
            color: white; border: none; font-weight: bold; border-radius: 8px; width: 100%; transition: 0.3s;
        }
        .stButton>button:hover { background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%); transform: translateY(-2px); }
        .reportview-container .main .block-container{ max-width: 1200px; }
        h1, h2, h3 { color: #38bdf8 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 28px !important; }
        .stAlert { border-radius: 12px !important; border: 1px solid rgba(56, 189, 248, 0.2); }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Multi-Disease Chest X-Ray Diagnostic AI")
st.caption("AI-powered chest X-ray screening for COVID-19, Pneumonia, and Tuberculosis")
st.markdown("---")

# --- Load the multi-disease model ---
@st.cache_resource
def load_multi_disease_model():
    model_path = "models/multi_disease_cnn_final.h5"
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception:
        return None

model = load_multi_disease_model()

# --- Sidebar UI Dashboard Menu ---
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>Clinical Control</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigation Menu:", ["🏥 Dashboard Overview", "🩻 X-Ray Neural Diagnostics"])
st.sidebar.markdown("---")

# Metric Counters visible in sidebar tracking active logs
st.sidebar.markdown("### 📊 Presentation Metrics")
st.sidebar.metric(label="Total Patients Scanned", value=len(st.session_state.patient_records))
st.sidebar.markdown("---")
st.sidebar.info("🤖 **System Status**: Engine online.\n\n✨ **AI Model**: DenseNet121 (Transfer Learning), 8 Epochs, 4-way classification.")

# --- PAGE 1: OVERVIEW & ANALYTICS ---
if app_mode == "🏥 Dashboard Overview":
    st.header("📋 Clinical Intelligence Dashboard")
    st.write("AI-assisted chest X-ray screening across four diagnostic classes: Normal, Pneumonia, COVID-19, and Tuberculosis.")

    m1, m2, m3 = st.columns(3)
    with m1: st.metric(label="Model Architecture", value="DenseNet121", delta="Transfer Learning")
    with m2: st.metric(label="Training", value="8 Epochs", delta="4 Diagnostic Classes")
    with m3: st.metric(label="Session Total Scanned Logs", value=f"{len(st.session_state.patient_records)} Patients")

    st.markdown("### 📋 Current Session Patient Registration Log")
    if len(st.session_state.patient_records) > 0:
        log_df = pd.DataFrame(st.session_state.patient_records)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("💡 **No Active Logs**: No patients have been scanned yet during this presentation segment. Run a scan to add data entries.")

# --- PAGE 2: X-RAY MULTI-DISEASE DIAGNOSTICS ---
elif app_mode == "🩻 X-Ray Neural Diagnostics":
    st.header("🩻 Computer Vision Diagnostics (Multi-Disease Chest X-Ray)")
    st.write("Upload a chest X-ray to detect **Normal, Pneumonia, COVID-19, or Tuberculosis**.")

    if model is None:
        st.warning(
            "⚠️ **Model not loaded.** The multi-disease model file "
            "(`models/multi_disease_cnn_final.h5`) hasn't been added to this deployment, "
            "or failed to load. Add the trained `.h5` file to the `models/` folder in the "
            "GitHub repo and redeploy."
        )
    else:
        class_names = ['COVID', 'Normal', 'Pneumonia', 'Tuberculosis']   # Check your train_gen.class_indices order!

        patient_name = st.text_input("Patient Full Name (مریض کا نام)", "Guest Patient")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 👤 Patient Profile")
            gender = st.selectbox("Patient Gender (جنس)", ["Male", "Female"])
            age = st.slider("Patient Age (عمر)", 1, 100, 40)

        ui_left, ui_right = st.columns(2)

        with ui_left:
            uploaded_file = st.file_uploader("Select Radiograph Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Chest X-Ray", use_column_width=True)

        with ui_right:
            if uploaded_file is not None:
                st.markdown("#### Neural Analysis Controls")
                if st.button("🧬 RUN MULTI-DISEASE DEEP SCAN"):

                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    with st.spinner("Analyzing X-Ray with Multi-Disease CNN..."):
                        preds = model.predict(img_array)[0]
                        pred_idx = np.argmax(preds)
                        confidence = preds[pred_idx] * 100
                        predicted_class = class_names[pred_idx]

                        st.markdown("### 🎯 Diagnosis Result")
                        st.progress(float(confidence) / 100)

                        if predicted_class == "Normal":
                            st.success(f"✅ **DIAGNOSIS: {predicted_class.upper()} LUNGS** (Confidence: {confidence:.2f}%)")
                        else:
                            st.error(f"🚨 **DIAGNOSIS: {predicted_class.upper()} DETECTED** (Confidence: {confidence:.2f}%)")

                        st.markdown("#### 📊 Class Probability Breakdown")
                        prob_df = pd.DataFrame({
                            "Disease": class_names,
                            "Probability (%)": [round(float(p) * 100, 2) for p in preds]
                        }).set_index("Disease")
                        st.bar_chart(prob_df, color="#38bdf8")

                        # Save to session log
                        st.session_state.patient_records.append({
                            "Patient Name": patient_name,
                            "Age": age,
                            "Gender": gender,
                            "Diagnostic Module": "Multi-Disease X-Ray",
                            "System Conclusion Result": f"{predicted_class} ({confidence:.1f}%)"
                        })
                        st.toast(f"Record added for {patient_name}!")
            else:
                st.info("💡 **Awaiting Input**: Please upload a chest X-ray in the left panel to run the multi-disease scan.")
