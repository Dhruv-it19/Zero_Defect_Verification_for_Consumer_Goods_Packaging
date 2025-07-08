"""
Dataset preparation script for packaging defect detection.
Combines and converts multiple datasets into YOLOv8 format.
"""

import os
import shutil
import argparse
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import track
from typing import List, Dict, Tuple

# Initialize rich console for pretty printing
console = Console()

class DatasetPreparator:
    def __init__(self, raw_data_dir: str, output_dir: str, binary_mode: bool = True):
        """
        Initialize the dataset preparator.
        
        Args:
            raw_data_dir: Path to raw datasets
            output_dir: Path to save processed datasets
            binary_mode: If True, convert all defects to single class
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.binary_mode = binary_mode
        self.class_mapping = self._get_class_mapping()
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _get_class_mapping(self) -> Dict[str, int]:
        """Get class mapping based on binary or multiclass mode."""
        if self.binary_mode:
            return {'defect': 0}
        else:
            return {
                'scratch': 0,
                'dent': 1,
                'misalignment': 2,
                'tear': 3,
                'wrinkle': 4
            }
    
    def _process_roboflow_dataset(self, dataset_dir: Path) -> List[Tuple[str, str, str]]:
        """
        Process Roboflow dataset in YOLO format, searching train/valid/test subfolders.
        Returns: List of (split, image_path, label_path)
        """
        converted_data = []
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            for img_path in split_dir.glob('images/*.jpg'):
                label_path = split_dir / 'labels' / f"{img_path.stem}.txt"
                if label_path.exists():
                    converted_data.append((split, str(img_path), str(label_path)))
        return converted_data
    
    def prepare_datasets(self):
        """Main method to prepare all datasets."""
        console.print("[bold blue]Starting dataset preparation (Roboflow only)...[/bold blue]")
        all_data = []
        for dataset in ['Cardboard Box.v1-carboard-dataset1.yolov8', 'Defect Detection.v6i.yolov8']:
            roboflow_data = self._process_roboflow_dataset(self.raw_data_dir / dataset)
            all_data.extend(roboflow_data)
        if not all_data:
            console.print("[red]Error: No data found in any of the Roboflow dataset directories![/red]")
            console.print(f"Please ensure the following datasets are present in {self.raw_data_dir}:")
            console.print("1. Cardboard Box.v1-carboard-dataset1.yolov8/")
            console.print("2. Defect Detection.v6i.yolov8/")
            return
        # Organize by split
        split_data = {'train': [], 'val': [], 'test': []}
        for split, img_path, label_path in all_data:
            if split == 'valid':
                split_data['val'].append((img_path, label_path))
            else:
                split_data[split].append((img_path, label_path))
        # Copy files and rewrite labels if binary
        for split in ['train', 'val', 'test']:
            for img_path, label_path in track(split_data[split], description=f"Processing {split} set"):
                # Copy image
                dest_img = self.output_dir / split / 'images' / Path(img_path).name
                shutil.copy2(img_path, dest_img)
                # Copy and rewrite label
                dest_label = self.output_dir / split / 'labels' / Path(label_path).name
                if self.binary_mode:
                    with open(label_path, 'r') as fin, open(dest_label, 'w') as fout:
                        for line in fin:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # Overwrite class index to 0
                                fout.write('0 ' + ' '.join(parts[1:]) + '\n')
                else:
                    shutil.copy2(label_path, dest_label)
        self._create_data_yaml()
        self._log_statistics()
    
    def _create_data_yaml(self):
        """Create YOLOv8 data configuration file."""
        yaml_data = {
            'path': str(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {v: k for k, v in self.class_mapping.items()}
        }
        
        with open('config/data.yaml', 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
    
    def _log_statistics(self):
        """Log dataset statistics."""
        stats = {
            'train': len(list((self.output_dir / 'train/images').glob('*.jpg'))),
            'val': len(list((self.output_dir / 'val/images').glob('*.jpg'))),
            'test': len(list((self.output_dir / 'test/images').glob('*.jpg')))
        }
        
        console.print("\n[bold green]Dataset Statistics:[/bold green]")
        console.print(f"Total images: {sum(stats.values())}")
        console.print(f"Train set: {stats['train']} images")
        console.print(f"Validation set: {stats['val']} images")
        console.print(f"Test set: {stats['test']} images")
        
        # Save statistics to CSV
        pd.DataFrame([stats]).to_csv('dataset_statistics.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for packaging defect detection')
    parser.add_argument('--raw_dir', default='data/raw', help='Path to raw datasets')
    parser.add_argument('--output_dir', default='data/yolov8', help='Path to save processed datasets')
    parser.add_argument('--binary', action='store_true', help='Convert to binary classification')
    args = parser.parse_args()
    
    preparator = DatasetPreparator(args.raw_dir, args.output_dir, args.binary)
    preparator.prepare_datasets()

if __name__ == '__main__':
    main() 