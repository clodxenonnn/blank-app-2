import streamlit as st
import os
import gdown
from PIL import Image
from ultralytics import YOLO
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Download YOLO model if not present
file_id = "1QJNq5JCLfoex6NcpoW-nTtTaBrwxbbpM"
model_path = "best.pt"

if not os.path.exists(model_path):
    with st.spinner("Downloading YOLO model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load YOLO model
model = YOLO(model_path)
yolo_classes = model.names  # Dictionary of class_id -> class_name

st.set_page_config(layout="wide")
st.title("🐶 Dog Detection with YOLO")

# --- SECTION 3: Live Webcam Detection (WebRTC) ---

st.subheader("🎥 Live Webcam Detection (WebRTC)")

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Class selection multiselect
selected_classes = st.multiselect(
    "Select classes to detect (leave empty to detect all):",
    options=list(yolo_classes.values()),
)

class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Run prediction
        results = self.model(img, conf=conf_threshold)[0]

        # Filter boxes by selected classes if any selected
        if selected_classes:
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            mask = np.array([yolo_classes[cls_id] in selected_classes for cls_id in cls_ids])
            if mask.any():
                # Filter boxes, scores, and classes
                filtered_boxes = results.boxes.xyxy.cpu().numpy()[mask]
                filtered_scores = results.boxes.conf.cpu().numpy()[mask]
                filtered_cls = cls_ids[mask]

                # Create a new Boxes object with filtered results
                from ultralytics.yolo.utils.ops import scale_boxes
                from ultralytics.yolo.engine.results import Results
                from ultralytics.yolo.utils import Ops
                import torch

                # Rebuild boxes object
                results.boxes.xyxy = torch.tensor(filtered_boxes)
                results.boxes.conf = torch.tensor(filtered_scores)
                results.boxes.cls = torch.tensor(filtered_cls)
            else:
                # No boxes to display
                results.boxes.xyxy = torch.empty((0,4))
                results.boxes.conf = torch.empty((0,))
                results.boxes.cls = torch.empty((0,))

        annotated_frame = results.plot()
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
        ]
    }
)

webrtc_ctx = webrtc_streamer(
    key="live-dog-detection",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    st.markdown("📡 **Streaming live video and detecting objects in real-time...**")
    st.markdown("👆 Adjust the confidence slider or filter by specific classes.")
else:
    st.warning("📷 Click the checkbox above to activate your webcam.")

# --- SECTION 1: Snapshot Detection (st.camera_input) ---
st.subheader("📸 Detect Dogs from Your Camera (Snapshot)")

img_file_buffer = st.camera_input("Take a picture using your webcam")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    st.image(image, caption="Captured Image", use_column_width=True)

    img_np = np.array(image.convert("RGB"))

    results = model(img_np)

    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)

# --- SECTION 2: Upload Image Detection ---
st.subheader("🖼️ Upload an Image for Detection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img.convert("RGB"))

    results = model(img_np)

    st.image(results[0].plot(), caption="Detected Image", use_column_width=True)
