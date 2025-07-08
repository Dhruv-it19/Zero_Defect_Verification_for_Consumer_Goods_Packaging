"""
Streamlit application for zero-defect verification in consumer goods packaging.
Provides a user-friendly interface for real-time defect detection and quality assurance.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import time
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.infer import DefectDetector

# Page config
st.set_page_config(
    page_title="Zero-Defect Verification for Consumer Goods Packaging",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for image resizing
MAX_IMAGE_SIZE = (250, 250)  # Maximum dimensions for display

# Initialize session state for statistics
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'defects_found' not in st.session_state:
    st.session_state.defects_found = 0
if 'processing_times' not in st.session_state:
    st.session_state.processing_times = []

# Initialize session state for model and detector
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """Resize image while maintaining aspect ratio."""
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image)
        else:
            # Handle grayscale or other formats
            image = Image.fromarray(image.astype('uint8'))
    
    # Calculate new dimensions while maintaining aspect ratio
    ratio = min(max_size[0] / image.width, max_size[1] / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)

def update_statistics(has_defect, processing_time):
    """Update session statistics."""
    st.session_state.total_processed += 1
    if has_defect:
        st.session_state.defects_found += 1
    st.session_state.processing_times.append(processing_time)

def display_statistics():
    """Display current session statistics."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Session Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Processed", st.session_state.total_processed)
    with col2:
        st.metric("Defects Found", st.session_state.defects_found)
    
    if st.session_state.processing_times:
        avg_time = sum(st.session_state.processing_times) / len(st.session_state.processing_times)
        st.sidebar.metric("Avg. Processing Time", f"{avg_time:.2f}s")
    
    # Quality Score
    if st.session_state.total_processed > 0:
        quality_score = ((st.session_state.total_processed - st.session_state.defects_found) 
                        / st.session_state.total_processed * 100)
        st.sidebar.progress(quality_score / 100, 
                          text=f"Quality Score: {quality_score:.1f}%")

def load_model():
    """Load the default defect detection model."""
    default_model = Path('packaging/zero_defect_s3/weights/best.pt')
    if not default_model.exists():
        st.error("Default model not found. Please ensure the model exists at packaging/zero_defect_s3/weights/best.pt")
        return
    
    try:
        st.session_state.detector = DefectDetector(str(default_model))
        st.session_state.model_path = str(default_model)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.detector = None
        st.session_state.model_path = None

def process_image(image, conf_threshold):
    """Process an image and return results."""
    if st.session_state.detector is None:
        st.error("Please wait while the model is loading...")
        return None, None
    
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Process the image
        annotated_img, results = st.session_state.detector.predict_image(
            image, conf_threshold=conf_threshold
        )
        
        # If no defects found, return the original image
        if not results['has_defect']:
            return image, results
            
        return annotated_img, results
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def style_dataframe(df):
    """Apply styling to the dataframe."""
    try:
        # Convert all columns to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Create a styled dataframe
        styled_df = df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': 'lightgrey',
            'border-style': 'solid',
            'border-width': '1px',
            'padding': '5px'
        })
        
        # Add hover effect
        styled_df = styled_df.set_table_styles([
            {'selector': 'tr:hover',
             'props': [('background-color', '#f0f0f0')]}
        ])
        
        return styled_df
    except Exception as e:
        st.warning(f"Could not apply styling to the dataframe: {str(e)}")
        return df

def format_defect_details(df):
    """Format defect details in a user-friendly way."""
    try:
        # Create a copy of the dataframe to avoid modifying the original
        formatted_df = df.copy()
        
        # Rename columns to be more user-friendly
        column_mapping = {
            'class': 'Defect Type',
            'confidence': 'Confidence Level',
            'xmin': 'Left Position',
            'ymin': 'Top Position',
            'xmax': 'Right Position',
            'ymax': 'Bottom Position'
        }
        formatted_df = formatted_df.rename(columns=column_mapping)
        
        # Format confidence as percentage
        if 'Confidence Level' in formatted_df.columns:
            formatted_df['Confidence Level'] = formatted_df['Confidence Level'].apply(
                lambda x: f"{x*100:.1f}%"
            )
        
        # Format position values as integers
        position_columns = ['Left Position', 'Top Position', 'Right Position', 'Bottom Position']
        for col in position_columns:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{int(x)}px")
        
        return formatted_df
    except Exception as e:
        st.warning(f"Could not format defect details: {str(e)}")
        return df

def get_defect_description(defect_type):
    """Get user-friendly description for defect types."""
    defect_descriptions = {
        'scratch': 'Surface scratch or abrasion',
        'dent': 'Surface dent or depression',
        'crack': 'Surface crack or fracture',
        'stain': 'Surface stain or discoloration',
        'wrinkle': 'Surface wrinkle or fold',
        'tear': 'Material tear or rip',
        'hole': 'Surface hole or puncture',
        'deformity': 'Shape deformity or irregularity',
        'contamination': 'Surface contamination or foreign material',
        'misalignment': 'Component misalignment'
    }
    return defect_descriptions.get(defect_type.lower(), defect_type)

def draw_defect_details(image, detections):
    """Draw defect details on the image."""
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Create a copy of the image
        annotated_img = image.copy()
        
        # Define colors for different defect types
        colors = {
            'scratch': (255, 0, 0),    # Red
            'dent': (0, 255, 0),       # Green
            'crack': (0, 0, 255),      # Blue
            'stain': (255, 255, 0),    # Yellow
            'wrinkle': (255, 0, 255),  # Magenta
            'tear': (0, 255, 255),     # Cyan
            'hole': (128, 0, 0),       # Maroon
            'deformity': (0, 128, 0),  # Dark Green
            'contamination': (0, 0, 128), # Dark Blue
            'misalignment': (128, 128, 0) # Olive
        }
        
        # Draw each detection
        for i, detection in enumerate(detections):
            try:
                # Get coordinates from the detection
                if 'box' in detection:
                    # If coordinates are in 'box' format
                    x1, y1, x2, y2 = map(int, detection['box'])
                else:
                    # If coordinates are in separate xmin, ymin, xmax, ymax format
                    x1 = int(detection.get('xmin', 0))
                    y1 = int(detection.get('ymin', 0))
                    x2 = int(detection.get('xmax', 0))
                    y2 = int(detection.get('ymax', 0))
                
                # Get defect type and confidence
                defect_type = detection.get('class', 'unknown')
                confidence = float(detection.get('confidence', 0))
                
                # Get color for this defect type
                color = colors.get(defect_type.lower(), (255, 255, 255))  # Default to white
                
                # Draw rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{defect_type.title()} ({confidence*100:.1f}%)"
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1-label_height-10), (x1+label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            except Exception as e:
                st.warning(f"Could not draw detection {i}: {str(e)}")
                continue
        
        return Image.fromarray(annotated_img)
    except Exception as e:
        st.warning(f"Could not draw defect details: {str(e)}")
        return image

def display_defect_summary(results):
    """Display a simple summary of detected defects."""
    if not results['has_defect']:
        return
    
    st.markdown("### 📋 Defect Summary")
    st.markdown(f"**Total Defects Found: {results['defect_count']}**")

def main():
    st.title("Zero-Defect Verification for Consumer Goods Packaging")
    
    # Add a helpful description with emojis
    st.markdown("""
    🎯 **Quality Assurance Made Simple**
    
    This application uses advanced AI to ensure zero-defect quality in consumer goods packaging.
    Upload an image or use your webcam to start the verification process.
    
    ---
    """)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Load the default model
    if st.session_state.detector is None:
        load_model()
    
    # Confidence threshold with visual feedback
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Adjust the sensitivity of defect detection"
    )
    
    # Display current threshold level
    threshold_level = "Low" if conf_threshold < 0.3 else "Medium" if conf_threshold < 0.7 else "High"
    st.sidebar.markdown(f"Current Sensitivity: **{threshold_level}**")
    
    # Main content
    tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Webcam"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Convert image to RGB format if it has an alpha channel
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Resize the uploaded image
            resized_image = resize_image(image)
            
            if st.button("🔍 Detect Defects", help="Start defect detection process"):
                with st.spinner("Processing image..."):
                    start_time = time.time()
                    annotated_img, results = process_image(image, conf_threshold)
                    processing_time = time.time() - start_time
                    
                    if annotated_img is not None:
                        # Update statistics
                        update_statistics(results['has_defect'], processing_time)
                        
                        # Create three columns for layout
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.image(resized_image, caption="Original Image", width=250)
                        
                        with col2:
                            if results['has_defect']:
                                # Draw defect details on the image
                                annotated_img_with_details = draw_defect_details(annotated_img, results['detections'])
                                resized_annotated = resize_image(annotated_img_with_details)
                                st.image(resized_annotated, caption="Detection Results", width=250)
                            else:
                                # Show the original image when no defects are found
                                st.image(resize_image(annotated_img), caption="No Defects Detected", width=250)
                        
                        with col3:
                            if results['has_defect']:
                                st.error(f"❌ Defects Detected: {results['defect_count']}")
                                # Display defect summary
                                display_defect_summary(results)
                                
                                # Display detections table with enhanced styling
                                if results['detections']:
                                    df = pd.DataFrame(results['detections'])
                                    formatted_df = format_defect_details(df)
                                    styled_df = style_dataframe(formatted_df)
                                    st.dataframe(
                                        styled_df,
                                        use_container_width=True
                                    )
                                    
                                    # Export results with timestamp
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "📥 Download Results",
                                        csv,
                                        f"defect_detections_{timestamp}.csv",
                                        "text/csv",
                                        help="Download detection results as CSV"
                                    )
                            else:
                                st.success("✅ No Defects Detected")
                                # The third column is kept for consistency in layout, but no image is shown here
                                # as the 'No Defects Detected' image is already in col2.
                        
                            # Display processing time always at the bottom of the content section
                            st.info(f"⏱️ Processing Time: {processing_time:.2f} seconds")
    
    with tab2:
        st.write("Click 'Start Webcam' to begin real-time defect detection")
        
        if st.button("🎥 Start Webcam", help="Start real-time webcam feed"):
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Create placeholders for webcam feed and results
            col1, col2, col3 = st.columns(3)
            webcam_placeholder = col1.empty()
            results_placeholder = col2.empty()
            # The third column placeholder is for potential future use or consistency
            # For now, it will be used to display the success message when no defects.
            preview_placeholder = col3.empty() 
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                    
                    # Convert frame to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    start_time = time.time()
                    annotated_img, results = process_image(frame_rgb, conf_threshold)
                    processing_time = time.time() - start_time
                    
                    if annotated_img is not None:
                        # Update statistics
                        update_statistics(results['has_defect'], processing_time)
                        
                        # Display original and processed frames
                        webcam_placeholder.image(resize_image(frame_rgb), caption="Original", width=250)
                        
                        if results['has_defect']:
                            # Draw defect details on the image
                            annotated_img_with_details = draw_defect_details(annotated_img, results['detections'])
                            results_placeholder.image(resize_image(annotated_img_with_details), caption="Detection Results", width=250)
                            preview_placeholder.empty() # Clear the success message if defects are found
                            st.error(f"❌ Defects Detected: {results['defect_count']}")
                        else:
                            # Show the original frame when no defects are found
                            results_placeholder.image(resize_image(annotated_img), caption="No Defects Detected", width=250)
                            preview_placeholder.success("✅ No Defects Detected") # Display success message in the third column
                        
                        # Display processing time
                        st.info(f"⏱️ Processing Time: {processing_time:.2f} seconds")
                    
                    # Check for stop button
                    if st.button("⏹️ Stop Webcam"):
                        break
                        
            finally:
                cap.release()
    
    # Display statistics
    display_statistics()

if __name__ == "__main__":
    main() 