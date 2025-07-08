"""
Inference script for packaging defect detection using trained YOLOv8 model.
Provides functions for image prediction and result formatting.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Dict, List, Union
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

class DefectDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the defect detector with a trained YOLOv8 model.
        
        Args:
            model_path: Path to the trained model weights. If None, uses the best model
                       from the latest training run.
        """
        if model_path is None:
            # Find the best model from the latest training run
            runs_dir = Path('runs/packaging')
            if not runs_dir.exists():
                raise FileNotFoundError("No training runs found. Please train a model first.")
            
            # Get the latest run directory
            latest_run = max(runs_dir.glob('zero_defect_*'), key=lambda x: x.stat().st_mtime)
            model_path = str(latest_run / 'weights' / 'best.pt')
        
        self.model = YOLO(model_path)
        self.class_names = self.model.names
    
    def predict_image(self, image: Union[str, np.ndarray, Image.Image], 
                     conf_threshold: float = 0.25) -> Tuple[np.ndarray, Dict]:
        """
        Predict defects in an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (annotated_image, results_dict)
        """
        # Convert input to numpy array if needed
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run inference
        results = self.model(image, conf=conf_threshold)[0]
        
        # Get detections
        boxes = results.boxes
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'class': self.class_names[cls],
                'confidence': conf,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Create annotated image
        annotated_img = results.plot()
        
        # Create results dictionary
        results_dict = {
            'detections': detections,
            'has_defect': len(detections) > 0,
            'defect_count': len(detections)
        }
        
        return annotated_img, results_dict
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]], 
                     conf_threshold: float = 0.25) -> List[Tuple[np.ndarray, Dict]]:
        """
        Predict defects in a batch of images.
        
        Args:
            images: List of input images
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of (annotated_image, results_dict) tuples
        """
        return [self.predict_image(img, conf_threshold) for img in images]

def main():
    """Example usage of the DefectDetector class."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    # Initialize detector
    detector = DefectDetector(args.model)
    
    # Run inference
    annotated_img, results = detector.predict_image(args.image, args.conf)
    
    # Save annotated image
    output_path = f"prediction_{Path(args.image).stem}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    
    # Print results
    print(f"\nResults saved to: {output_path}")
    print(f"Defects found: {results['defect_count']}")
    print("\nDetections:")
    for det in results['detections']:
        print(f"- {det['class']}: {det['confidence']:.2f}")

if __name__ == '__main__':
    main() 