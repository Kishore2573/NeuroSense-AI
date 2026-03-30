# =============================
# SYSTEM CLEAN START
# =============================
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# =============================
# IMPORTS
# =============================
import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import tensorflow as tf
import cv2
from PIL import Image
import base64
import gdown   # 🔥 NEW

# =============================
# DOWNLOAD MODEL (FIX FOR LARGE FILE)
# =============================
MODEL_PATH = "handwriting_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1QIQn0WiKcTNO7EpystaIedD46ldM_aEu"
    gdown.download(url, MODEL_PATH, quiet=False)

# =============================
# PAGE SETTINGS
# =============================
st.set_page_config(page_title="NeuroSense AI", page_icon="🧠", layout="wide")

# =============================
# BACKGROUND
# =============================
def add_bg(image_file):
    try:
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
    except:
        encoded = ""

    st.markdown(f"""
    <style>
    .stApp {{
        background:
        linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.65)),
        url("data:image/webp;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

add_bg("background.webp")

# =============================
# HEADER
# =============================
st.markdown("""
<h1 style='text-align:center;color:#e11d48;'>🧠 NeuroSense AI Pro</h1>
<p style='text-align:center;'>AI Clinical Parkinson Detection</p>
""", unsafe_allow_html=True)

st.divider()

# =============================
# LOAD MODELS (SAFE PATH)
# =============================
@st.cache_resource
def load_models():
    BASE = os.path.dirname(os.path.abspath(__file__))

    voice_model = joblib.load(os.path.join(BASE, "voice_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))

    handwriting_model = tf.keras.models.load_model(
        os.path.join(BASE, "handwriting_model.keras"),
        compile=False
    )

    return voice_model, scaler, handwriting_model

voice_model, scaler, handwriting_model = load_models()

# =============================
# SESSION STATE
# =============================
if "voice_score" not in st.session_state:
    st.session_state.voice_score = None

if "hand_score" not in st.session_state:
    st.session_state.hand_score = None

# =============================
# LAYOUT
# =============================
col1, col2 = st.columns(2)

# =============================
# VOICE ANALYSIS (FIXED)
# =============================
with col1:
    st.subheader("🎤 Voice Analysis")

    voice_file = st.file_uploader("Upload Voice", type=["wav","mp3","ogg","m4a"])

    def extract_features(upload):
        # 🔥 FIX: SAFE TEMP FILE
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(upload.read())
            temp_path = tmp.name

        y, sr = librosa.load(temp_path, sr=22050)

        features = [
            np.mean(librosa.yin(y,50,300)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.zero_crossing_rate(y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        ]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))

        features = np.array(features)

        if len(features) < 22:
            features = np.pad(features, (0,22-len(features)))

        return features.reshape(1,-1)

    if voice_file:
        st.audio(voice_file)

        if st.button("Analyze Voice"):
            feat = extract_features(voice_file)
            feat = scaler.transform(feat)

            prob = voice_model.predict_proba(feat)[0][1]
            score = prob * 100

            st.session_state.voice_score = score
            st.progress(int(score))
            st.success(f"Voice Score: {score:.2f}%")

# =============================
# HANDWRITING ANALYSIS
# =============================
with col2:
    st.subheader("✍ Handwriting Analysis")

    file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    def preprocess(img):
        img = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        thresh = cv2.adaptiveThreshold(
            blur,255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,11,2
        )

        img = cv2.resize(thresh,(224,224))/255.0
        img = np.expand_dims(img,0)
        img = np.expand_dims(img,-1)

        return img

    if file:
        img = Image.open(file)
        st.image(img, width=200)

        if st.button("Analyze Handwriting"):
            data = preprocess(img)
            prob = float(handwriting_model(data)[0][0])
            score = prob * 100

            st.session_state.hand_score = score
            st.progress(int(score))
            st.success(f"Handwriting Score: {score:.2f}%")

# =============================
# FINAL RESULT
# =============================
st.divider()
st.header("🧠 Final Result")

v = st.session_state.voice_score
h = st.session_state.hand_score

if v is not None and h is not None:
    final = (v+h)/2

    st.progress(int(final))
    st.metric("Risk", f"{final:.2f}%")

    if final > 65:
        st.error("High Parkinson Risk")
    elif final > 40:
        st.warning("Moderate Risk")
    else:
        st.success("Low Risk")

else:
    st.info("Run both tests")

st.divider()
st.caption("NeuroSense AI • Clinical Screening System")
