from keras.layers import TFSMLayer
import streamlit as st
from PIL import Image
import numpy as np
from util import classify, set_background


# ---- Setup Halaman ----
st.set_page_config(page_title="Pneumonia Classifier", page_icon="🩺", layout="centered")

# ---- Background ----
set_background('./bgs/bg.jpg')

# ---- Header ----
st.markdown(
    """
    <h1 style='text-align: center; color: #FFFF;'>🩺 Pneumonia Classification App</h1>
    <p style='text-align: center; color: #FFFF;'>
    Upload a chest X-ray image to detect <b>Pneumonia</b> using a trained deep learning model.
    </p>
    """,
    unsafe_allow_html=True
)

# ---- Load model dengan cache ----
@st.cache_resource
def load_pneumonia_model():
    model = TFSMLayer(
        r"D:\Code\Github\Pneunomia Classification\pneumonia_model_tf",
        call_endpoint="serving_default")
    return model

model = load_pneumonia_model()

# ---- Load class names ----
with open('./model/labels.txt', 'r') as f:
    class_names = [a.strip().split(' ')[1] for a in f.readlines()]

# ---- Upload File ----
uploaded_file = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Preview", use_container_width=True)

    with st.spinner("🔍 Classifying... please wait..."):
        # Prediksi
        class_name, conf_score = classify(image, model, class_names)

    # ---- Tampilkan Hasil ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style='background-color:#EBF5FB; padding:12px; border-radius:15px; text-align:center; padding-bottom:0.5px'>
                <p style='color:#117A65; font-size:20px; font-weight:bold;'>
                ✅ Result: {class_name}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style='background-color:#EBF5FB; padding:12px; border-radius:15px; text-align:center; padding-bottom:0.5px'>
                <p style='color:#1A5276; font-size:20px; font-weight:bold;'>
                📊 Confidence: {int(conf_score * 1000) / 10}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


else:
    st.markdown(
    "<p style='color:#FFFF; font-size:18px; text-align:center;'>👆 Please upload an image file (JPG or PNG) to start the classification.</p>",
    unsafe_allow_html=True)

