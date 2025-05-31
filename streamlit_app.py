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

# Function for live object detection using webcam
def live_streaming(conf_threshold, selected_classes):
    stframe = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error(
            "Error: Could not access the webcam. Please make sure your webcam is working."
        )
        return

    try:
        while st.session_state.get("is_detecting", False) and st.session_state.get(
            "is_webcam_active", False
        ):
            ret, frame = cap.read()

            if not ret:
                st.warning("Warning: Failed to read frame from the webcam. Retrying...")
                continue

            try:
                results = model.predict(source=frame, conf=conf_threshold)
                detections = results[0]

                # Extract bounding boxes, confidence scores, and class IDs
                boxes = (
                    detections.boxes.xyxy.cpu().numpy() if len(detections) > 0 else []
                )
                confs = (
                    detections.boxes.conf.cpu().numpy() if len(detections) > 0 else []
                )
                class_ids = (
                    detections.boxes.cls.cpu().numpy().astype(int)
                    if len(detections) > 0
                    else []
                )

                # Filter based on selected classes
                if selected_classes:
                    filtered = [
                        (box, conf, class_id)
                        for box, conf, class_id in zip(boxes, confs, class_ids)
                        if yolo_classes[class_id] in selected_classes
                    ]
                    if filtered:
                        boxes, confs, class_ids = zip(*filtered)
                    else:
                        boxes, confs, class_ids = [], [], []

                # Draw bounding boxes and labels on the frame
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{yolo_classes[class_ids[i]]}: {confs[i]:.2f}"
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

                # Display the frame in Streamlit
                stframe.image(frame, channels="BGR")

            except Exception as e:
                st.error(f"Error during model prediction: {str(e)}")

    finally:
        # Ensure resources are properly released
        cap.release()
        cv2.destroyAllWindows()
# SECTION 3: Webcam Live Stream Detection
st.subheader("🎥 Live Webcam Detection")

# Toggle webcam activation
webcam_checkbox = st.checkbox("Activate Webcam for Live Detection")

if webcam_checkbox:
    st.session_state.is_webcam_active = True
    st.session_state.is_detecting = True

    # Confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    # Get class names from the model
    yolo_classes = model.names

    # Class selection multiselect
    selected_classes = st.multiselect(
        "Select classes to detect (leave empty to detect all):",
        options=list(yolo_classes.values()),
    )

    # Start the live webcam detection
    live_streaming(conf_threshold, selected_classes)

else:
    st.session_state.is_webcam_active = False
    st.session_state.is_detecting = False
        
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
