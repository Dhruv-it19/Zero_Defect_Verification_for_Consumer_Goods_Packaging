# Zero-Defect Verification for Consumer Goods Packaging

A Streamlit-based application for real-time defect detection in consumer goods packaging using computer vision and deep learning.

## 🎯 Overview

This application provides a user-friendly interface for quality assurance in consumer goods packaging. It uses advanced AI to detect various types of defects such as scratches, dents, cracks, stains, and more.

## ✨ Features

- 📸 Image Upload: Upload and analyze images for defects
- 🎥 Real-time Webcam Detection: Live defect detection using webcam feed
- 📊 Real-time Statistics: Track processing metrics and quality scores
- 🔍 Configurable Detection: Adjust confidence thresholds for defect detection
- 📥 Export Results: Download detection results as CSV files
- 🎨 Visual Annotations: Color-coded defect highlighting with confidence scores

## 🏗️ Project Structure

```
zero-defect-streamlit/
├── streamlit_app/
│   └── app.py              # Main Streamlit application
├── src/
│   └── infer.py           # Inference module for defect detection
├── packaging/
│   └── zero_defect_s3/
│       └── weights/       # Model weights directory
├── data/                  # Dataset directory
├── config/               # Configuration files
├── notebooks/           # Jupyter notebooks
├── predictions/         # Prediction outputs
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster inference)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/zero-defect-streamlit.git
cd zero-defect-streamlit
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run streamlit_app/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 🛠️ Usage

1. **Image Upload**:
   - Click on the "Image Upload" tab
   - Upload an image of the packaging
   - Click "Detect Defects" to start analysis

2. **Webcam Detection**:
   - Click on the "Webcam" tab
   - Click "Start Webcam" to begin real-time detection
   - Click "Stop Webcam" to end the session

3. **Adjusting Detection Sensitivity**:
   - Use the confidence threshold slider in the sidebar
   - Lower values increase sensitivity (may detect more potential defects)
   - Higher values increase specificity (fewer false positives)

## 📊 Supported Defect Types

- Scratches
- Dents
- Cracks
- Stains
- Wrinkles
- Tears
- Holes
- Deformities
- Contamination
- Misalignment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Computer Vision with [OpenCV](https://opencv.org/) 