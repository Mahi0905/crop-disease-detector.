# tools/flatten_plantvillage.py
import os, shutil, uuid
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\lenovo\Desktop\crop-disease-detector\crop-disease-detector")
DATA_ROOT = PROJECT_ROOT / "data"
SUBSETS = ["color", "grayscale", "segmented"]
SPLITS = ["train", "val"]

def move_all(src_class_dir: Path, dst_class_dir: Path, prefix: str):
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    for p in src_class_dir.rglob("*"):
        if p.is_file():
            new_name = f"{prefix}_{p.stem}_{uuid.uuid4().hex[:8]}{p.suffix.lower()}"
            shutil.move(str(p), str(dst_class_dir / new_name))

def flatten_split(split_root: Path):
    pv_root = split_root / "plantvillage dataset"
    if not pv_root.exists():
        print(f"Skip {split_root} (no 'plantvillage dataset' folder).")
        return

    print(f"Flattening {pv_root} into {split_root}")
    for subset in SUBSETS:
        subset_root = pv_root / subset
        if not subset_root.exists():
            print(f"  - {subset_root} not found; skipping.")
            continue

        for class_dir in [d for d in subset_root.iterdir() if d.is_dir()]:
            dst_class_dir = split_root / class_dir.name
            prefix = "col" if subset == "color" else "gray" if subset == "grayscale" else "seg"
            print(f"  • {subset}/{class_dir.name} -> {dst_class_dir.name} (prefix={prefix})")
            move_all(class_dir, dst_class_dir, prefix)

    try:
        shutil.rmtree(pv_root)
        print(f"Removed {pv_root}")
    except Exception as e:
        print(f"Note: couldn't remove {pv_root}: {e}")

def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    for split in SPLITS:
        split_root = DATA_ROOT / split
        if split_root.exists():
            flatten_split(split_root)
        else:
            print(f"Split folder not found: {split_root}")

if __name__ == "__main__":
    main()
