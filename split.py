# tools/split_train_val.py
import os, shutil, random
from pathlib import Path

BASE_DIR = Path(r"C:\Users\lenovo\Desktop\crop-disease-detector\crop-disease-detector")
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
VAL_RATIO = 0.2  # 20% for validation

random.seed(42)
VAL_DIR.mkdir(parents=True, exist_ok=True)

for class_folder in [d for d in TRAIN_DIR.iterdir() if d.is_dir()]:
    val_class_dir = VAL_DIR / class_folder.name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    images = [f for f in class_folder.glob("*") if f.is_file()]
    val_count = int(len(images) * VAL_RATIO)
    val_images = random.sample(images, val_count)

    for img in val_images:
        shutil.move(str(img), str(val_class_dir / img.name))

print("✅ Validation split completed.")
