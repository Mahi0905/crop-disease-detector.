import os
import random
import shutil
from pathlib import Path

# Set working directory
base_dir = Path(__file__).resolve().parent
train_dir = base_dir / "data" / "train" / "plantvillage dataset"/"segmented"

val_dir = base_dir / "data" / "validation"
val_dir.mkdir(parents=True, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(train_dir):
    class_path = train_dir / class_name
    if not class_path.is_dir():
        continue

    val_class_path = val_dir / class_name
    val_class_path.mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    split_count = int(0.2 * len(images))
    val_images = images[:split_count]

    for img in val_images:
        src = class_path / img
        dst = val_class_path / img
        shutil.move(src, dst)

    print(f"✅ {class_name}: Moved {split_count} images to validation.")
