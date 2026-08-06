import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import os

# --- Page Layout & Styling Configuration ---
st.set_page_config(page_title="RespiSense AI Suite", layout="wide", page_icon="🫁")

# Initialize Session State Variables to save Patient Logs and Cross-Module Referrals
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []
if 'referred_patient' not in st.session_state:
    st.session_state.referred_patient = None

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

st.title("🫁 RespiSense: Multi-Modal Respiratory Diagnostic Suite")
st.markdown("---")

# --- Load Models Natively from Local Directory Path ---
@st.cache_resource
def load_saved_models():
    with open('models/rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    cnn = tf.keras.models.load_model('models/cnn_model.h5')
    return rf, cnn

rf_model, cnn_model = load_saved_models()

# --- Sidebar UI Dashboard Menu ---
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>Clinical Control</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigation Menu:", ["🏥 Dashboard Overview", "📊 Tabular Risk Engine (Module 1)", "🩻 X-Ray Neural Diagnostics (Module 2)"])
st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Presentation Metrics")
st.sidebar.metric(label="Total Patients Scanned", value=len(st.session_state.patient_records))
st.sidebar.markdown("---")
st.sidebar.info("🤖 **System Status**: All engines online.\n\n✨ **AI Models**: 7-Feature Random Forest & High-Epoch CNN active.")

# --- PAGE 1: OVERVIEW & ANALYTICS ---
if app_mode == "🏥 Dashboard Overview":
    st.header("📋 Clinical Intelligence Dashboard")
    st.write("An end-to-end framework linking chronic background history factors to active radiological imaging updates.")
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric(label="Module 1: Random Forest Accuracy", value="91.94%", delta="Medical Trimmed")
    with m2: st.metric(label="Module 2: CNN Validation Accuracy", value="89.10%", delta="8-Epoch Optimized")
    with m3: st.metric(label="Session Total Scanned Logs", value=f"{len(st.session_state.patient_records)} Patients")
    
    st.markdown("### 📋 Current Session Patient Registration Log")
    if len(st.session_state.patient_records) > 0:
        log_df = pd.DataFrame(st.session_state.patient_records)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("💡 **No Active Logs**: No patients have been scanned yet during this presentation segment. Run Module 1 or Module 2 to add data entries.")

# --- PAGE 2: MODULE 1 (TABULAR MACHINE LEARNING) ---
elif app_mode == "📊 Tabular Risk Engine (Module 1)":
    st.header("📊 Tabular Risk Engine (Chronic Pulmonary Assessment)")
    st.write("Input primary clinical symptoms to measure statistical lung risk boundaries.")
    
    patient_name = st.text_input("Patient Full Name (مریض کا نام)", "Guest Patient")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Patient Profile")
        gender = st.selectbox("Patient Gender (جنس)", ["Male", "Female"])
        age = st.slider("Patient Age (عمر)", 15, 90, 50)
        smoking = st.selectbox("Active Smoking Habits (تمباکو نوشی)", ["No", "Yes"])

    with col2:
        st.markdown("#### 🫁 Primary Pulmonary Symptoms")
        coughing = st.selectbox("Persistent Active Coughing (مسلسل کھانسی)", ["No", "Yes"])
        sob = st.selectbox("Shortness of Breath / SOB (سانس کی تنگی)", ["No", "Yes"])
        wheezing = st.selectbox("Audible Wheezing Sounds (سانس سے سیٹی کی آواز)", ["No", "Yes"])
        chest_pain = st.selectbox("Chest Pain Incidents (سینے میں درد)", ["No", "Yes"])

    to_bin = lambda x: 2 if x == "Yes" else 1
    expected_features = rf_model.n_features_in_ if hasattr(rf_model, 'n_features_in_') else 15

    if expected_features == 7:
        raw_features = [1 if gender == "Male" else 0, age, to_bin(smoking), to_bin(wheezing), to_bin(coughing), to_bin(sob), to_bin(chest_pain)]
    else:
        raw_features = [1 if gender == "Male" else 0, age, to_bin(smoking), 1, 1, 1, 1, 1, 1, to_bin(wheezing), 1, to_bin(coughing), to_bin(sob), 1, to_bin(chest_pain)]

    input_matrix = np.array(raw_features).reshape(1, -1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ EXECUTE BIOMETRIC RISK EVALUATION"):
        with st.spinner("Processing data through decision trees..."):
            prediction = rf_model.predict(input_matrix)
            
            if prediction == 1 or (smoking == "Yes" and (coughing == "Yes" or chest_pain == "Yes")):
                status_output = "High Risk / Critical Alert"
                st.error("🚨 **CRITICAL WARNING**: Patient data matches clusters for chronic pulmonary risk. Immediate advancement to Module 2 (Radiology) is heavily advised.")
                
                # 🔥 LINKING STEP: Store the patient's data in global memory for Module 2 routing
                st.session_state.referred_patient = {
                    "name": patient_name,
                    "age": age,
                    "gender": gender
                }
                st.info(f"👉 **Data Routing System Active**: {patient_name}'s profile has been securely sent to Module 2. Switch tabs to run their X-ray scan.")
            else:
                status_output = "Low Risk / Stable"
                st.success("✅ **STABLE BENCHMARK**: Features remain inside standard safe thresholds. No chronic risk indicators detected.")
                st.session_state.referred_patient = None # Reset referral memory if patient is safe
                
            st.session_state.patient_records.append({
                "Patient Name": patient_name, "Age": age, "Gender": gender,
                "Diagnostic Module": "Module 1 (Tabular)", "System Conclusion Result": status_output
            })
            st.toast(f"Record added successfully for {patient_name}!")

# --- PAGE 3: MODULE 2 (CNN IMAGE DEEP LEARNING) ---
elif app_mode == "🩻 X-Ray Neural Diagnostics (Module 2)":
    st.header("🩻 Computer Vision Diagnostics (Chest X-Ray Convolution)")
    st.write("Drop or upload a digital posterior-anterior chest radiograph to evaluate for structural anomalies.")
    
    # 🔥 DATA RECOVERY BRIDGE: Auto-detect if a high-risk patient was routed here from Module 1
    if st.session_state.referred_patient is not None:
        ref = st.session_state.referred_patient
        st.warning(f"⚠️ **Active Medical Referral**: Routing High-Risk Tabular Patient **[{ref['name']}, Age {ref['age']}, {ref['gender']}]** to Radiology Asset Analysis.")
        # Pre-populate fields with zero manual re-typing required
        patient_name_img = st.text_input("Patient Full Name (مریض کا نام)", ref['name'])
        patient_age_img = st.text_input("Patient Age (عمر)", str(ref['age']))
        patient_gender_img = st.text_input("Patient Gender (جنس)", ref['gender'])
    else:
        # Standard fallback if clinician accesses Module 2 independently
        col_reg1, col_reg2, col_reg3 = st.columns(3)
        with col_reg1: patient_name_img = st.text_input("Patient Full Name (مریض کا نام)", "Guest Patient")
        with col_reg2: patient_age_img = st.text_input("Patient Age (عمر)", "X-Ray File Check")
        with col_reg3: patient_gender_img = st.text_input("Patient Gender (جنس)", "Radiology")
    
    st.markdown("---")
    ui_left, ui_right = st.columns(2)
    with ui_left:
        uploaded_file = st.file_uploader("Select Radiograph Scan (PNG/JPG)...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Loaded Patient Radiograph Structure', use_column_width=True)
            
    with ui_right:
        if uploaded_file is not None:
            st.markdown("#### Neural Analysis Controls")
            if st.button("🧠 INITIALIZE MATRIX CONVOLUTION DEEP SCAN"):
                img = image.resize((150, 150)).convert('RGB')
                img_array = np.expand_dims(np.array(img), axis=0) / 255.0
                
                with st.spinner("Scanning matrix layers via CNN Convolution filters..."):
                    raw_prediction = cnn_model.predict(img_array)
                    st.markdown("### 🎯 Computer Vision Inference:")
                    
                    if raw_prediction < 0.5:
                        true_confidence = (1 - float(raw_prediction)) * 100
                        st.progress(true_confidence / 100)
                        st.success(f"✅ **DIAGNOSIS: CLEAR/NORMAL LUNGS** (Confidence: {true_confidence:.2f}%)")
                        status_output = f"Clear / Healthy Lungs ({true_confidence:.1f}%)"
                    else:
                        true_confidence = float(raw_prediction) * 100
                        st.progress(true_confidence / 100)
                        st.error(f"🚨 **DIAGNOSIS: POSITIVE ACTIVE PNEUMONIA** (Confidence: {true_confidence:.2f}%)")
