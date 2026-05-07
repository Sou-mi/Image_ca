import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm
import gdown
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😀",
    layout="centered"
)

st.title("😀 Facial Expression Recognition")
st.write("Upload a face image to detect emotion.")

# ----------------------------
# MODEL DOWNLOAD
# ----------------------------
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading trained model..."):
        file_id = "1cKCN-bwWqy3WkgR9vYqVX3skYzz5aA-W"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ----------------------------
# CLASS LABELS
# ----------------------------
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ----------------------------
# MODEL DEFINITION
# ✅ Fixed: efficientnet_b2 matches the trained checkpoint
# ----------------------------
class FERModel(nn.Module):

    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()
        self.backbone = timm.create_model(
            "efficientnet_b2",   # ← FIXED (was efficientnet_b0)
            pretrained=False
        )
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = FERModel(num_classes=7)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu")
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.eval()
    return model

model = load_model()

# ----------------------------
# IMAGE TRANSFORM
# ✅ efficientnet_b2 uses 260x260 input size
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((260, 260)),   # ← FIXED (b2 uses 260, b0 uses 224)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ----------------------------
# PREDICTION
# ----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No face detected. Try a clearer/closer image.")
    else:
        st.success(f"{len(faces)} face(s) detected.")

        for (x, y, w, h) in faces:

            # Draw rectangle on image
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop and preprocess face
            face = image_np[y:y + h, x:x + w]
            face_pil = Image.fromarray(face)
            input_tensor = transform(face_pil).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            emotion = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item() * 100

            # Display result
            st.subheader("Prediction Result")
            st.write(f"### Emotion: {emotion}")
            st.write(f"### Confidence: {confidence_score:.2f}%")

            # Show all class probabilities
            st.write("**Probability breakdown:**")
            prob_dict = {
                CLASS_NAMES[i]: float(probabilities[0][i]) * 100
                for i in range(len(CLASS_NAMES))
            }
            for emo, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                st.progress(int(prob), text=f"{emo}: {prob:.1f}%")

        # Show image with bounding boxes
        st.image(image_np, caption="Detected Face(s)", use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch + EfficientNet-B2")
