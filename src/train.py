"""
Training script for packaging defect detection using YOLOv8.
Supports different model sizes and training configurations.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb
from rich.console import Console
import torch

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for packaging defect detection')
    parser.add_argument('--model_size', type=str, default='s',
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=640,
                      help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Training batch size')
    parser.add_argument('--binary', action='store_true',
                      help='Use binary classification mode')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    return parser.parse_args()

def train(args):
    # Initialize model with pretrained weights
    model = YOLO('packaging/zero_defect_s3/weights/best.pt')
    
    # Configure training parameters
    train_args = {
        'data': 'config/data.yaml',
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'project': 'packaging',
        'name': f'zero_defect_s3_finetune',
        'patience': 20,  # Early stopping patience
        'save': True,    # Save best model
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Improved training parameters
        'lr0': 0.001,    # Lower initial learning rate for fine-tuning
        'lrf': 0.001,    # Lower final learning rate fraction
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    # Enable W&B logging if requested
    if args.wandb:
        if not os.environ.get('WANDB_API_KEY'):
            console.print("[yellow]Warning: WANDB_API_KEY not set. Skipping W&B logging.[/yellow]")
        else:
            wandb.init(
                project="packaging-defect-detection",
                config={
                    "model_size": 's',
                    "epochs": args.epochs,
                    "img_size": args.img_size,
                    "batch_size": args.batch_size,
                    "binary_mode": args.binary
                }
            )
            train_args['project'] = 'wandb'
    
    # Start training
    console.print(f"[bold blue]Starting training with YOLOv8-s[/bold blue]")
    console.print(f"Training arguments: {train_args}")
    
    try:
        results = model.train(**train_args)
        
        # Print final metrics
        console.print("\n[bold green]Training completed![/bold green]")
        console.print(f"Best mAP50: {results.maps[0]:.3f}")
        console.print(f"Best model saved at: {results.save_dir}")
        
    except Exception as e:
        console.print(f"[bold red]Training failed: {str(e)}[/bold red]")
        raise
    
    finally:
        if args.wandb and wandb.run is not None:
            wandb.finish()

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main() 