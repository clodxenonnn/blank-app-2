import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ✅ Set Streamlit to full screen layout
st.set_page_config(layout="wide")

# ✅ Title
st.title("🐶 Dog Detection with YOLOv8")

# ✅ Download model from Google Drive if not already present
file_id = "1QJNq5JCLfoex6NcpoW-nTtTaBrwxbbpM"
model_path = "best.pt"
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# ✅ Load YOLO model
model = YOLO(model_path)

# ✅ Define YOLO video processing for webcam
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_img = results[0].plot()
        return annotated_img

# ✅ Webcam stream section
st.subheader("📷 Real-time Dog Detection (Webcam)")
webrtc_streamer(key="dog-detection", video_processor_factory=YOLOTransformer)

# ✅ Image upload and detection
st.subheader("🖼️ Upload an Image for Detection")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array for YOLO
    img_np = np.array(img.convert("RGB"))

    # Run detection
    results = model(img_np)

    # Show result
    for r in results:
        st.image(r.plot(), caption="Detected Image", use_column_width=True)
