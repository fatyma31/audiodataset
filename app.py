import os
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import gdown
from PIL import Image

st.set_page_config(page_title="Audio Classifier", page_icon="🎵", layout="centered")

# ── Config ──────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 4
HOP_LENGTH  = 512
N_FFT       = 2048
N_MELS      = 128
IMG_SIZE    = 128

CLASSES = [
    'Baby cry','Clock tick','Cow','Dog','Fire crackling','Frog',
    'Helicopter','Person sneeze','Pig','Rain','Rooster','Sea waves'
]

MODEL_PATH = "model.tflite"
GDRIVE_ID  = "1gEX_-jHiFX7NPr_Sw63iTnDEPwKi7vlt"

# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
   interpreter.allocate_tensors()
   return interpreter

# ── Feature Extraction ──────────────────────────────────────
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Resize using PIL — no tensorflow needed
    mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))
    mel_img = mel_img.resize((IMG_SIZE, IMG_SIZE))
    mel_r   = np.array(mel_img).astype(np.float32) / 255.0

    cmap = plt.get_cmap('magma')
    rgb  = cmap(mel_r)[:, :, :3]
    rgb  = (rgb * 255).astype(np.float32)
    rgb  = (rgb / 127.5) - 1.0  # MobileNetV2 preprocessing

    return rgb[np.newaxis, ...], audio, mel_db

# ── Predict ──────────────────────────────────────────────────
def predict(interpreter, features):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ── UI ───────────────────────────────────────────────────────
st.title("🎵 Audio Classifier")
st.markdown("**MobileNetV2 Transfer Learning** — 12 Sound Classes")
st.divider()

interpreter = load_model()
st.success("✅ Model loaded!")

st.subheader("Upload Audio File")
uploaded = st.file_uploader("Choose .wav / .mp3 / .ogg / .flac",
                             type=["wav","mp3","ogg","flac"])

if uploaded:
    st.audio(uploaded)
    st.caption(f"📄 {uploaded.name}  ({uploaded.size/1024:.1f} KB)")

    if st.button("🔍 Predict", use_container_width=True, type="primary"):
        with st.spinner("Extracting features..."):
            with tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded.name)[1], delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                x, audio, mel_db = extract_features(tmp_path)
            except Exception as e:
                st.error(f"Feature extraction error: {e}")
                st.stop()

        with st.spinner("Running model..."):
            probs = predict(interpreter, x)

        st.divider()
        st.subheader("Prediction")

        top_idx   = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        top_conf  = float(probs[top_idx]) * 100

        c1, c2 = st.columns(2)
        c1.metric("Predicted Class", top_class)
        c2.metric("Confidence", f"{top_conf:.1f}%")

        top6  = np.argsort(probs)[::-1][:6]
        names = [CLASSES[i] for i in top6]
        confs = probs[top6] * 100

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ['#378ADD' if i == 0 else '#B5D4F4' for i in range(6)]
        ax.barh(names[::-1], confs[::-1], color=colors[::-1])
        for i, (name, val) in enumerate(zip(names[::-1], confs[::-1])):
            ax.text(val + 0.3, i, f"{val:.1f}%", va='center', fontsize=10)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, max(confs) + 14)
        ax.set_title("Top 6 Predictions", fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        with st.expander("View Mel Spectrogram"):
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            import librosa.display
            librosa.display.specshow(mel_db, sr=SAMPLE_RATE, ax=ax2, cmap='magma')
            ax2.set_title("Mel Spectrogram")
            ax2.axis('off')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        os.unlink(tmp_path)

with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("""
| Property | Value |
|----------|-------|
| Base Model | MobileNetV2 |
| Format | TFLite |
| Input | 128×128 RGB |
| Feature | Mel Spectrogram |
| Classes | 12 |
| Expected Acc | 80–95% |
""")
    st.markdown("**Classes**")
    for c in CLASSES:
        st.markdown(f"- {c}")
