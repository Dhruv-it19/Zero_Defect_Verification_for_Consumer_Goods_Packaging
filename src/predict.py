import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from rich.console import Console
import argparse

console = Console()

def predict_image(model, image_path, conf_threshold=0.25):
    """Predict defects in a single image."""
    results = model.predict(image_path, conf=conf_threshold)[0]
    return results

def process_image(results, save_dir=None):
    """Process prediction results and optionally save annotated image."""
    # Get the original image
    img = results.orig_img
    
    # Draw predictions
    annotated_img = results.plot()
    
    # Save if requested
    if save_dir:
        save_path = os.path.join(save_dir, Path(results.path).name)
        cv2.imwrite(save_path, annotated_img)
        console.print(f"Saved annotated image to: {save_path}")
    
    return {
        'image': annotated_img,
        'boxes': results.boxes.xyxy.cpu().numpy(),
        'confidences': results.boxes.conf.cpu().numpy(),
        'class_ids': results.boxes.cls.cpu().numpy().astype(int),
        'class_names': [results.names[i] for i in results.boxes.cls.cpu().numpy().astype(int)]
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run defect detection on images')
    parser.add_argument('--test_dir', type=str, default='data/test/images',
                      help='Directory containing test images')
    parser.add_argument('--model_path', type=str, default='packaging/zero_defect_s/weights/best.pt',
                      help='Path to the model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold for detections')
    args = parser.parse_args()

    # Load the model
    if not os.path.exists(args.model_path):
        console.print(f"[red]Error: Model not found at {args.model_path}[/red]")
        return
    
    model = YOLO(args.model_path)
    
    # Create output directory
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test directory if it doesn't exist
    os.makedirs(args.test_dir, exist_ok=True)
    
    # Check if test directory is empty
    test_images = list(Path(args.test_dir).glob('*.jpg')) + list(Path(args.test_dir).glob('*.png'))
    if not test_images:
        console.print(f"[yellow]Warning: No images found in {args.test_dir}[/yellow]")
        console.print("Please add some images to the test directory and run again.")
        return
    
    console.print("[bold blue]Starting predictions...[/bold blue]")
    
    for img_path in test_images:
        console.print(f"\nProcessing {img_path.name}...")
        
        # Get predictions
        results = predict_image(model, str(img_path), conf_threshold=args.conf)
        
        # Process and save results
        pred_results = process_image(results, save_dir=output_dir)
        
        # Print detection summary
        if len(pred_results['boxes']) > 0:
            console.print(f"Found {len(pred_results['boxes'])} defects:")
            for box, conf, cls_name in zip(pred_results['boxes'], 
                                         pred_results['confidences'], 
                                         pred_results['class_names']):
                console.print(f"- {cls_name}: {conf:.2f} confidence")
        else:
            console.print("No defects detected.")
    
    console.print(f"\n[bold green]Predictions completed! Results saved in {output_dir}[/bold green]")

if __name__ == "__main__":
    main() 