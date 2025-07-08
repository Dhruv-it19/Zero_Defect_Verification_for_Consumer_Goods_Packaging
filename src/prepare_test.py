import os
import shutil
from pathlib import Path
from rich.console import Console

console = Console()

def prepare_test_images():
    # Source directories
    source_dirs = [
        'data/raw/Defect Detection.v6i.yolov8/test/images',
        'data/raw/Cardboard Box.v1-carboard-dataset1.yolov8/test/images'
    ]
    
    # Destination directory
    dest_dir = 'data/test/images'
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy images
    copied_count = 0
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for img_path in Path(source_dir).glob('*.jpg'):
                dest_path = os.path.join(dest_dir, img_path.name)
                shutil.copy2(img_path, dest_path)
                copied_count += 1
                console.print(f"Copied: {img_path.name}")
    
    if copied_count > 0:
        console.print(f"\n[green]Successfully copied {copied_count} images to {dest_dir}[/green]")
    else:
        console.print("[yellow]No images were found in the source directories[/yellow]")

if __name__ == "__main__":
    prepare_test_images() 