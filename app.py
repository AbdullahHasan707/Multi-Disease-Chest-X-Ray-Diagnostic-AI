import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2

# Optional Qwen (Alibaba DashScope)
try:
    import dashscope
    from dashscope import Generation
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

st.set_page_config(
    page_title="Multi-Disease Chest X-Ray Diagnostic AI",
    layout="wide",
    page_icon="🫁"
)

if "patient_records" not in st.session_state:
    st.session_state.patient_records = []

st.markdown("""
    <style>
        .main { background-color: #0f172a; color: #f8fafc; }
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
            color: white; border: none; font-weight: bold; border-radius: 8px; width: 100%; transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
            transform: translateY(-2px);
        }
        h1, h2, h3 { color: #38bdf8 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 28px !important; }
        .stAlert { border-radius: 12px !important; border: 1px solid rgba(56, 189, 248, 0.2); }
    </style>
""", unsafe_allow_html=True)

st.title("🫁 Multi-Disease Chest X-Ray Diagnostic AI")
st.caption("AI-powered chest X-ray screening for COVID-19, Pneumonia, and Tuberculosis · Grad-CAM + optional Qwen explanation")
st.markdown("---")


def show_image(img, caption=""):
    """Compatible image display across Streamlit versions."""
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(img, caption=caption, use_column_width=True)
        except TypeError:
            st.image(img, caption=caption)


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
CLASS_NAMES = ["COVID", "Normal", "Pneumonia", "Tuberculosis"]


def make_gradcam_heatmap(model, img_array, pred_index):
    """Pure TensorFlow Grad-CAM."""
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            continue

    if last_conv_layer_name is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_pil, heatmap, alpha=0.45):
    img = np.array(original_pil.resize((224, 224)).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * img).astype(np.uint8)
    return overlay


def get_qwen_explanation(disease, confidence, probs_dict):
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["DASHSCOPE_API_KEY"]
        except Exception:
            api_key = None

    if not QWEN_AVAILABLE or not api_key:
        return None

    dashscope.api_key = api_key
    prompt = f"""You are a careful medical AI assistant helping a clinician.
A chest X-ray model predicted: {disease} with {confidence:.1f}% confidence.
Class probabilities: {probs_dict}

Write a short, clear, non-alarming explanation in 3-5 sentences for a doctor.
Mention that this is AI assistance only and final diagnosis requires clinical correlation and expert review.
Do not invent findings not supported by the prediction."""

    try:
        response = Generation.call(model="qwen-turbo", prompt=prompt)
        if response and getattr(response, "output", None):
            out = response.output
            if isinstance(out, dict):
                return out.get("text") or str(out)
            return str(out)
        return str(response)
    except Exception as e:
        return f"Qwen explanation unavailable: {e}"


# Sidebar
st.sidebar.markdown("<h2 style='text-align: center; color: white;'>Clinical Control</h2>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("Navigation Menu:", ["🏥 Dashboard Overview", "🩻 X-Ray Neural Diagnostics"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Presentation Metrics")
st.sidebar.metric(label="Total Patients Scanned", value=len(st.session_state.patient_records))
st.sidebar.markdown("---")
st.sidebar.info(
    "🤖 **System Status**: Engine online.\n\n"
    "✨ **AI Model**: DenseNet121 + Grad-CAM\n\n"
    "📝 **Optional**: Qwen explanation (set DASHSCOPE_API_KEY)"
)

if app_mode == "🏥 Dashboard Overview":
    st.header("📋 Clinical Intelligence Dashboard")
    st.write(
        "AI-assisted chest X-ray screening across four diagnostic classes: "
        "Normal, Pneumonia, COVID-19, and Tuberculosis. "
        "Includes Grad-CAM visual explainability and optional Qwen text explanation."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Model Architecture", value="DenseNet121", delta="Transfer Learning")
    with m2:
        st.metric(label="Explainability", value="Grad-CAM", delta="+ Qwen optional")
    with m3:
        st.metric(label="Session Total Scanned", value=f"{len(st.session_state.patient_records)} Patients")

    st.markdown("### 📋 Current Session Patient Registration Log")
    if len(st.session_state.patient_records) > 0:
        st.dataframe(pd.DataFrame(st.session_state.patient_records), use_container_width=True)
    else:
        st.info("💡 **No Active Logs**: No patients have been scanned yet. Run a scan to add entries.")

elif app_mode == "🩻 X-Ray Neural Diagnostics":
    st.header("🩻 Computer Vision Diagnostics (Multi-Disease Chest X-Ray)")
    st.write("Upload a chest X-ray to detect **Normal, Pneumonia, COVID-19, or Tuberculosis**.")

    if model is None:
        st.warning(
            "⚠️ **Model not loaded.** Add `models/multi_disease_cnn_final.h5` to the repo and redeploy."
        )
    else:
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
                show_image(image, caption="Uploaded Chest X-Ray")

        with ui_right:
            if uploaded_file is not None:
                st.markdown("#### Neural Analysis Controls")
                if st.button("🧬 RUN MULTI-DISEASE DEEP SCAN"):
                    img = image.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

                    with st.spinner("Analyzing X-Ray · Grad-CAM · optional Qwen..."):
                        preds = model.predict(img_array, verbose=0)[0]
                        pred_idx = int(np.argmax(preds))
                        confidence = float(preds[pred_idx] * 100)
                        predicted_class = CLASS_NAMES[pred_idx]
                        probs_dict = {CLASS_NAMES[i]: round(float(preds[i] * 100), 2) for i in range(len(CLASS_NAMES))}

                        st.markdown("### 🎯 Diagnosis Result")
                        st.progress(min(confidence / 100.0, 1.0))

                        if predicted_class == "Normal":
                            st.success(f"✅ **DIAGNOSIS: {predicted_class.upper()} LUNGS** (Confidence: {confidence:.2f}%)")
                        else:
                            st.error(f"🚨 **DIAGNOSIS: {predicted_class.upper()} DETECTED** (Confidence: {confidence:.2f}%)")

                        st.markdown("#### 📊 Class Probability Breakdown")
                        prob_df = pd.DataFrame({
                            "Disease": CLASS_NAMES,
                            "Probability (%)": [round(float(p) * 100, 2) for p in preds]
                        }).set_index("Disease")
                        st.bar_chart(prob_df)

                        # Grad-CAM
                        st.markdown("#### 🔥 Grad-CAM Explainability")
                        try:
                            heatmap = make_gradcam_heatmap(model, img_array, pred_idx)
                            if heatmap is not None:
                                overlay = overlay_gradcam(image, heatmap)
                                c1, c2 = st.columns(2)
                                with c1:
                                    show_image(image.resize((224, 224)), caption="Original X-Ray")
                                with c2:
                                    show_image(overlay, caption="Grad-CAM (model attention)")
                                st.caption("Warmer colors = regions the model focused on for this prediction.")
                            else:
                                st.info("Grad-CAM could not locate a convolutional layer.")
                        except Exception as e:
                            st.warning(f"Grad-CAM failed: {e}")

                        # Qwen explanation
                        st.markdown("#### 📝 AI Explanation (Qwen)")
                        explanation = get_qwen_explanation(predicted_class, confidence, probs_dict)
                        if explanation:
                            st.info(explanation)
                        else:
                            st.caption(
                                "Set environment variable or Streamlit secret `DASHSCOPE_API_KEY` "
                                "to enable Alibaba Qwen explanations."
                            )

                        st.session_state.patient_records.append({
                            "Patient Name": patient_name,
                            "Age": age,
                            "Gender": gender,
                            "Diagnostic Module": "Multi-Disease X-Ray + Grad-CAM",
                            "System Conclusion Result": f"{predicted_class} ({confidence:.1f}%)"
                        })
                        st.toast(f"Record added for {patient_name}!")
            else:
                st.info("💡 **Awaiting Input**: Upload a chest X-ray on the left to run the multi-disease scan.")
