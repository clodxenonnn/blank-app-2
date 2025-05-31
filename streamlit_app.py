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


from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

class YOLOVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # You can add detection or processing here
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="example",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
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
