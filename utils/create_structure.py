import os
from pathlib import Path

def create_folder_structure(base_path=".."):  # Go up one level from utils
    """Create the folder structure for the multimodal financial risk project."""
    
    # Define all directories and files to create
    structure = {
        "": ["main.py", "config.py", "README.md", "requirements.txt"],  # Root files
        "data": ["scraper.py", "processor.py", "image_processor.py"],
        "text": ["text_processor.py"],
        "models": ["encoders.py", "attention.py", "multimodal_model.py"],
        "train": ["dataset.py", "trainer.py"],
        "utils": ["metrics.py"],  # This is where your script is located
        "data/raw": [],
        "charts": [],
        "checkpoints": [],
    }
    
    # Create all directories and files
    for dir_path, files in structure.items():
        full_dir_path = Path(base_path) / dir_path
        
        # Create directory if it doesn't exist
        try:
            full_dir_path.mkdir(parents=True, exist_ok=False)
            print(f"Created directory: {full_dir_path}")
        except FileExistsError:
            print(f"Directory already exists: {full_dir_path}")
        
        # Create files in the directory
        for file in files:
            file_path = full_dir_path / file
            if not file_path.exists():
                file_path.touch()
                print(f"Created file: {file_path}")
            else:
                print(f"File already exists: {file_path}")

if __name__ == "__main__":
    create_folder_structure()