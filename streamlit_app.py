import streamlit as st
import os
import gdown
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
# Set page config to use wide layout
st.set_page_config(layout="wide")

# Google Drive file ID for YOLO model
file_id = "1QJNq5JCLfoex6NcpoW-nTtTaBrwxbbpM"
model_path = "best.pt"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load YOLO model
model = YOLO(model_path)

# Title for the app
st.title("🐶 Dog Detection with YOLO")


# Initialize session state variables
if "is_detecting" not in st.session_state:
    st.session_state.is_detecting = False
if "is_webcam_active" not in st.session_state:
    st.session_state.is_webcam_active = False


# SECTION 1: Real-time snapshot capture using st.camera_input()
st.subheader("📸 Detect Dogs from Your Camera (Snapshot)")

# Camera input (replaces streamlit-webrtc)
img_file_buffer = st.camera_input("Take a picture using your webcam")

if img_file_buffer is not None:
    # Read and display the captured image
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)

    # Convert to NumPy array
    img_np = np.array(image.convert("RGB"))

    # Run YOLO detection
    results = model(img_np)

    # Display result
    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

# SECTION 2: Offline image upload
st.subheader("🖼️ Upload an Image for Detection")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to NumPy array
    img_np = np.array(img.convert("RGB"))

    # Run YOLO detection
    results = model(img_np)

    # Display result
    st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
