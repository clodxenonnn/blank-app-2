import streamlit as st
import os
import gdown
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
st.set_page_config(layout="wide")
# Google Drive file ID for the model
file_id = "1QJNq5JCLfoex6NcpoW-nTtTaBrwxbbpM"
output = "best.pt"

# Download the model if it doesn't exist
if not os.path.exists(output):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Title for the app
st.title("Dog Detection with YOLO")

# Load the YOLO model (use the path to your trained model file)
model = YOLO(output)  # Use the path to the downloaded best.pt

# Define the video processing class
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")
        # Run YOLO model to detect objects
        results = model(img)
        # Annotate the image with detected boxes
        annotated_img = results[0].plot()
        return annotated_img

# Webcam stream
webrtc_streamer(key="dog-detection", video_processor_factory=YOLOTransformer)

# Image upload section
st.write("Upload an image for detection:")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Run detection on the uploaded image
    results = model(img)
    for r in results:
        st.image(r.plot(), caption="Detected Image", use_column_width=True)
