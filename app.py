# Application entry point
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import timm

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😀",
    layout="centered"
)

st.title("😀 Facial Expression Recognition App")
st.write("Upload a face image and predict the facial emotion.")

# ----------------------------
# CLASS LABELS
# ----------------------------
CLASS_NAMES = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
]

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class FERModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FERModel, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)


# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource

def load_model():
    model = FERModel(num_classes=7)

    # Replace with your trained model file
    model.load_state_dict(
        torch.load('best_model.pth', map_location=torch.device('cpu'))
    )

    model.eval()
    return model


model = load_model()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# FACE DETECTION
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ----------------------------
# FILE UPLOADER
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        st.warning("No face detected in the image.")

    else:
        st.success(f"Detected {len(faces)} face(s)")

        for (x, y, w, h) in faces:

            # Draw rectangle
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop face
            face = image_np[y:y+h, x:x+w]
            face_pil = Image.fromarray(face)

            # Preprocess
            input_tensor = transform(face_pil).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            predicted_class = CLASS_NAMES[predicted.item()]
            confidence_score = confidence.item() * 100

            st.subheader("Prediction Result")
            st.write(f"**Emotion:** {predicted_class}")
            st.write(f"**Confidence:** {confidence_score:.2f}%")

        st.image(image_np, caption='Detected Face(s)', use_container_width=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("Built with Streamlit + PyTorch")
