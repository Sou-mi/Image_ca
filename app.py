import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import gdown
import os

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────
IMG_SIZE    = 224
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)
NUM_CLASSES = 7
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ─────────────────────────────────────────────────────────────────
# 🔑 REPLACE with your Google Drive file ID for best_model.pth
#    e.g. from https://drive.google.com/file/d/THIS_PART/view
# ─────────────────────────────────────────────────────────────────
MODEL_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"
MODEL_PATH    = "best_model.pth"


# ── Model Definition (mirrors training notebook exactly) ──────────
class FaceExpressionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 1792

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Download model weights ────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights…"):
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)


# ── Load & cache model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    download_model()
    model = FaceExpressionModel(num_classes=NUM_CLASSES, pretrained=False)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Unwrap common dict wrappers
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    # Strip DataParallel 'module.' prefix
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


# ── Face detection ────────────────────────────────────────────────
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(img_rgb):
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    if len(faces) == 0:
        return img_rgb, None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return img_rgb[y : y + h, x : x + w], (x, y, w, h)


# ── Preprocessing ─────────────────────────────────────────────────
def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - np.array(MEAN)) / np.array(STD)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()


# ── Inference ─────────────────────────────────────────────────────
def predict(model, img_rgb):
    crop, bbox = detect_face(img_rgb)
    with torch.no_grad():
        logits = model(preprocess(crop))
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    return probs, bbox


# ── UI ────────────────────────────────────────────────────────────
EMOJI = {"Angry":"😠","Disgust":"🤢","Fear":"😨",
         "Happy":"😄","Neutral":"😐","Sad":"😢","Surprise":"😲"}

st.title("😊 Facial Expression Recognition")
st.caption("EfficientNet-B4 · FER-2013")
st.markdown("---")

# Load model
try:
    model = load_model()
    st.success("✅ Model loaded!")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.info("Set MODEL_FILE_ID in app.py to your Google Drive file ID.")
    st.stop()

mode = st.radio("Input method", ["Upload Image", "Webcam"], horizontal=True)
img_rgb = None

if mode == "Upload Image":
    f = st.file_uploader("Upload a face photo", type=["jpg","jpeg","png"])
    if f:
        img_rgb = np.array(Image.open(f).convert("RGB"))
else:
    snap = st.camera_input("Take a photo")
    if snap:
        img_rgb = np.array(Image.open(snap).convert("RGB"))

if img_rgb is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)

    probs, bbox = predict(model, img_rgb)

    with col1:
        st.subheader("Image")
        disp = img_rgb.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            st.warning("No face detected — using full image.")
        st.image(disp, use_container_width=True)

    with col2:
        st.subheader("Result")
        top_idx   = int(np.argmax(probs))
        top_label = CLASS_NAMES[top_idx]
        top_conf  = probs[top_idx] * 100

        st.markdown(f"## {EMOJI.get(top_label,'🤔')} {top_label}")
        st.metric("Confidence", f"{top_conf:.1f}%")

        st.markdown("**All probabilities**")
        for cls, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f"{cls}: {p*100:.1f}%")
