import os
import requests
import joblib
import tensorflow as tf
import streamlit as st

# ==============================
# AUTO DOWNLOAD MODELS
# ==============================

def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            r = requests.get(url)
            with open(filename, "wb") as f:
                f.write(r.content)

# Your model URLs (upload models to Hugging Face first)
VOICE_MODEL_URL = "https://huggingface.co/spaces/kishore2573/NeuroSense-PRO/resolve/main/voice_model.pkl"
SCALER_URL = "https://huggingface.co/spaces/kishore2573/NeuroSense-PRO/resolve/main/scaler.pkl"
HANDWRITING_MODEL_URL = "https://huggingface.co/spaces/kishore2573/NeuroSense-PRO/resolve/main/handwriting_model.keras"

download_model(VOICE_MODEL_URL, "voice_model.pkl")
download_model(SCALER_URL, "scaler.pkl")
download_model(HANDWRITING_MODEL_URL, "handwriting_model.keras")

# ==============================
# LOAD MODELS
# ==============================

@st.cache_resource
def load_models():
    voice_model = joblib.load("voice_model.pkl")
    scaler = joblib.load("scaler.pkl")
    handwriting_model = tf.keras.models.load_model("handwriting_model.keras")
    return voice_model, scaler, handwriting_model

voice_model, scaler, handwriting_model = load_models()