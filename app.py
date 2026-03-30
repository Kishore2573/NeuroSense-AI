# =============================
# SYSTEM CLEAN START (Fix warnings)
# =============================
import os
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"        # Disable GPU errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # Disable oneDNN logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"         # Hide TensorFlow logs
warnings.filterwarnings("ignore")                # Hide sklearn warnings

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import gdown

MODEL_PATH = "handwriting_model.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/file/d/1QIQn0WiKcTNO7EpystaIedD46ldM_aEu/view?usp=drive_link"
    gdown.download(url, MODEL_PATH, quiet=False)
# =============================
# PAGE SETTINGS
# =============================
st.set_page_config(
    page_title="NeuroSense AI",
    page_icon="🧠",
    layout="wide"
)

# =============================
# BACKGROUND IMAGE
# =============================
def add_bg(image_file):
    try:
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
    except:
        encoded = ""

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.65)),
            url("data:image/webp;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .title-center {{
            text-align:center;
            padding-top:10px;
        }}

        h1 {{
            color:#e11d48;
            font-size:44px;
            font-weight:800;
        }}

        h2,h3 {{
            color:white;
        }}

        p, label {{
            color:#f1f5f9;
        }}

        .block-container {{
            background: rgba(0,0,0,0.35);
            padding: 20px;
            border-radius: 14px;
        }}

        .stButton>button {{
            background-color:#e11d48;
            color:white;
            height:45px;
            border-radius:10px;
            font-weight:600;
            width:100%;
        }}

        .stButton>button:hover {{
            background-color:#be123c;
        }}

        [data-testid="stFileUploader"] {{
            background-color: rgba(0,0,0,0.5);
            border-radius:10px;
            padding:10px;
        }}

        .stProgress > div > div > div > div {{
            background-color:#e11d48;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("background.webp")

# =============================
# HEADER
# =============================
st.markdown(
    """
    <div class="title-center">
        <h1>🧠 NeuroSense AI Pro</h1>
        <p>Clinical AI System for Early Parkinson Detection</p>
        <p>Voice Biomarkers • Motor Pattern Analysis • AI Diagnosis</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =============================
# LOAD MODELS
# =============================
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        voice_model = joblib.load("voice_model.pkl")
        scaler = joblib.load("scaler.pkl")
        handwriting_model = tf.keras.models.load_model(
            "handwriting_model.keras",
            compile=False
        )
        return voice_model, scaler, handwriting_model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

voice_model, scaler, handwriting_model = load_models()

# =============================
# SESSION STATE
# =============================
if "voice_score" not in st.session_state:
    st.session_state.voice_score = None

if "hand_score" not in st.session_state:
    st.session_state.hand_score = None

# =============================
# DASHBOARD
# =============================
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("System Status", "Active")

with c2:
    st.metric("AI Model", "NeuroSense v2")

with c3:
    st.metric("Screening Mode", "Clinical")

st.divider()

# =============================
# MAIN PANELS
# =============================
col1, col2 = st.columns([1, 1], gap="large")

# =============================
# VOICE ANALYSIS
# =============================
with col1:
    st.subheader("🎤 Voice Biomarker Analysis")

    voice_file = st.file_uploader(
        "Upload Voice Sample",
        type=["wav", "mp3", "ogg", "m4a"],
        key="voice_upload"
    )

    def save_temp(upload):
        suffix = upload.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(upload.read())
            return tmp.name

    def extract_features(file):
        y, sr = librosa.load(file, sr=22050)

        features = []

        pitch = librosa.yin(y, fmin=50, fmax=300)
        features.append(np.mean(pitch))

        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(bandwidth))

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.extend(np.mean(mfcc, axis=1))

        features = np.array(features)

        if len(features) < 22:
            features = np.pad(features, (0, 22 - len(features)))
        else:
            features = features[:22]

        return features.reshape(1, -1)

    if voice_file:
        st.audio(voice_file)

        if st.button("Analyze Voice"):
            with st.spinner("Analyzing voice biomarkers..."):
                path = save_temp(voice_file)
                features = extract_features(path)
                features = scaler.transform(features)

                prob = voice_model.predict_proba(features)[0][1]
                score = prob * 100
                st.session_state.voice_score = score

            st.progress(int(score))
            st.metric("Voice Score", f"{score:.2f}%")

# =============================
# HANDWRITING ANALYSIS
# =============================
with col2:
    st.subheader("✍ Handwriting Motor Analysis")

    method = st.radio(
        "Input Method",
        ["Upload Image", "Use Camera"]
    )

    image = None

    if method == "Upload Image":
        file = st.file_uploader(
            "Upload handwriting sample",
            type=["png", "jpg", "jpeg"],
            key="hand_upload"
        )
        if file:
            image = Image.open(file)
            st.image(image, width=260)

    if method == "Use Camera":
        cam = st.camera_input("Capture handwriting")
        if cam:
            image = Image.open(cam)
            st.image(image, width=260)

    IMG_SIZE = 224

    def preprocess(image):
        img = np.array(image.convert("RGB"))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
        processed = processed / 255.0
        processed = np.expand_dims(processed, axis=0)

        return processed

    if image is not None:
        if st.button("Analyze Handwriting"):
            with st.spinner("Analyzing handwriting patterns..."):
                processed = preprocess(image)
                prob = float(handwriting_model(processed, training=False)[0][0])
                score = prob * 100
                st.session_state.hand_score = score

            st.progress(int(score))
            st.metric("Handwriting Score", f"{score:.2f}%")

# =============================
# FINAL RESULT
# =============================
st.divider()
st.header("🧠 Final Neurological Assessment")

voice_score = st.session_state.voice_score
hand_score = st.session_state.hand_score

if voice_score is not None and hand_score is not None:

    final_score = (voice_score + hand_score) / 2

    st.progress(int(final_score))
    st.metric("Overall Parkinson Risk", f"{final_score:.2f}%")

    if final_score >= 65:
        diagnosis = "High Parkinson Risk"
        st.error(diagnosis)

        causes = [
            "Possible dopamine neuron degeneration",
            "Motor control irregularities detected",
            "Voice tremor biomarkers present",
            "Handwriting micrographia patterns observed"
        ]

        prevention = [
            "Consult neurologist",
            "Regular neurological tests",
            "Start physiotherapy",
            "Healthy brain diet",
            "Exercise regularly"
        ]

    elif final_score >= 40:
        diagnosis = "Moderate Risk"
        st.warning(diagnosis)

        causes = [
            "Early motor irregularities",
            "Mild voice instability",
            "Possible neurological stress"
        ]

        prevention = [
            "Monitor symptoms",
            "Hand coordination exercises",
            "Voice practice",
            "Reduce stress",
            "Good sleep cycle"
        ]

    else:
        diagnosis = "Low Risk"
        st.success(diagnosis)

        causes = [
            "Stable neurological signals",
            "No tremor indicators detected"
        ]

        prevention = [
            "Maintain healthy lifestyle",
            "Regular exercise",
            "Brain training activities",
            "Balanced diet"
        ]

    st.subheader("Clinical Insights")
    for c in causes:
        st.write("•", c)

    st.subheader("Prevention Advice")
    for p in prevention:
        st.write("•", p)

else:
    st.info("Run both analyses to generate final report.")

st.divider()
st.caption("NeuroSense AI • Advanced Clinical Screening System")
