import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Zero Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = 'packaging/zero_defect_s/weights/best.pt'
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    return YOLO(model_path)

def process_image(model, image, conf_threshold=0.25):
    """Process a single image and return results."""
    results = model.predict(image, conf=conf_threshold)[0]
    return results

def main():
    st.title("🔍 Zero Defect Detection")
    st.write("Upload an image to detect defects in packaging materials.")

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # Sidebar
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR")

        # Process image
        results = process_image(model, image, conf_threshold)
        
        # Draw predictions
        annotated_img = results.plot()
        
        # Display results
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_img, channels="BGR")

        # Display detection details
        st.subheader("Detection Details")
        if len(results.boxes) > 0:
            for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                class_name = results.names[int(cls)]
                st.write(f"- {class_name}: {conf:.2f} confidence")
                st.write(f"  Location: {box.tolist()}")
        else:
            st.write("No defects detected.")

        # Download button for annotated image
        if len(results.boxes) > 0:
            _, buffer = cv2.imencode('.jpg', annotated_img)
            st.download_button(
                label="Download Annotated Image",
                data=buffer.tobytes(),
                file_name="detected_defects.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main() 