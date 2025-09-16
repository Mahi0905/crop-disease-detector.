# train_model.py — handles nested PlantVillage structure (color/grayscale/segmented)
import os, json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 7
SEED = 123

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
os.makedirs(MODEL_DIR, exist_ok=True)

SUBTREES = ["color", "grayscale", "segmented"]  # auto-detect which exist
PV_FOLDER_NAME = "plantvillage dataset"

def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def ds_from_root(root, subset=None, validation_split=None, shuffle=True):
    kwargs = dict(
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=shuffle,
    )
    if validation_split:
        kwargs.update(validation_split=validation_split, subset=subset)

    ds = tf.keras.utils.image_dataset_from_directory(root, **kwargs)
    return ds

def concat_datasets(datasets):
    ds = datasets[0]
    for nxt in datasets[1:]:
        ds = ds.concatenate(nxt)
    return ds

def load_datasets():
    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")

    # Case 1: two-folder layout but nested inside 'plantvillage dataset'
    pv_train = os.path.join(train_dir, PV_FOLDER_NAME)
    pv_val = os.path.join(val_dir, PV_FOLDER_NAME)

    if os.path.isdir(train_dir) and os.path.isdir(val_dir) and os.path.isdir(pv_train):
        print("📂 Detected nested two-folder layout under 'plantvillage dataset'")
        train_roots = [os.path.join(pv_train, s) for s in SUBTREES if os.path.isdir(os.path.join(pv_train, s))]
        if not train_roots:
            raise RuntimeError(f"No {SUBTREES} folders inside {pv_train}")
        val_roots = []
        if os.path.isdir(pv_val):
            val_roots = [os.path.join(pv_val, s) for s in SUBTREES if os.path.isdir(os.path.join(pv_val, s))]
            if not val_roots:
                raise RuntimeError(f"No {SUBTREES} folders inside {pv_val}")
        else:
            val_roots = None  # fall back to split from train

        # Build train datasets
        ds_list_train = [ds_from_root(r) for r in train_roots]
        class_names = ds_list_train[0].class_names
        # Sanity: ensure class_names align across roots
        for dsi in ds_list_train[1:]:
            if dsi.class_names != class_names:
                raise RuntimeError("Class sets differ across color/grayscale/segmented roots.")
        train_ds = concat_datasets(ds_list_train)

        # Build validation datasets
        if val_roots:
            ds_list_val = [ds_from_root(r, shuffle=False) for r in val_roots]
            for dsi in ds_list_val:
                if dsi.class_names != class_names:
                    raise RuntimeError("Class sets differ across train and val roots.")
            val_ds = concat_datasets(ds_list_val)
        else:
            # Split from train roots (same split/seed across all)
            ds_list_train = [tf.keras.utils.image_dataset_from_directory(
                r, validation_split=0.2, subset="training", seed=SEED,
                image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=True) for r in train_roots]
            ds_list_val = [tf.keras.utils.image_dataset_from_directory(
                r, validation_split=0.2, subset="validation", seed=SEED,
                image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, shuffle=False) for r in train_roots]
            for dsi in ds_list_val + ds_list_train:
                if dsi.class_names != class_names:
                    raise RuntimeError("Class sets differ across split roots.")
            train_ds = concat_datasets(ds_list_train)
            val_ds = concat_datasets(ds_list_val)

        return train_ds, val_ds, class_names

    # Case 2: standard two-folder layout (already flattened)
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print("📂 Using standard two-folder layout: data/train and data/val")
        train_ds = ds_from_root(train_dir)
        val_ds = ds_from_root(val_dir, shuffle=False)
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names

    # Case 3: single root with split
    if os.path.isdir(DATA_ROOT):
        print("📂 Using single-folder layout with validation_split")
        train_ds = ds_from_root(DATA_ROOT, validation_split=0.2, subset="training")
        val_ds = ds_from_root(DATA_ROOT, validation_split=0.2, subset="validation", shuffle=False)
        class_names = train_ds.class_names
        return train_ds, val_ds, class_names

    raise RuntimeError("Dataset not found. Expected data/train and data/val or a data/ folder with class subdirs.")

print("🔄 Loading dataset…")
train_dataset, validation_dataset, class_names = load_datasets()
print(f"✅ Found {len(class_names)} classes.")
print("Example classes:", class_names[:8])

# Save class names (order matters!)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)
print(f"📝 Saved class names -> {CLASS_NAMES_PATH}")

# Performance pipeline
AUTOTUNE = tf.data.AUTOTUNE
def prep(ds, shuffle=False):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle: ds = ds.shuffle(1000, seed=SEED)
    return ds.prefetch(AUTOTUNE)

train_dataset = prep(train_dataset, shuffle=True)
validation_dataset = prep(validation_dataset, shuffle=False)

# Build or rebuild model to match classes
def build_model(num_classes):
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])
    return model

model = None
if os.path.exists(MODEL_PATH):
    try:
        tmp = tf.keras.models.load_model(MODEL_PATH)
        out_units = tmp.output_shape[-1]
        if out_units == len(class_names):
            print("🔁 Resuming from saved model…")
            model = tmp
        else:
            print(f"⚠️ Saved model has {out_units} classes, but dataset has {len(class_names)}. Rebuilding.")
            model = build_model(len(class_names))
    except Exception as e:
        print("⚠️ Could not load existing model:", e)
        model = build_model(len(class_names))
else:
    print("🆕 Building new model…")
    model = build_model(len(class_names))

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1)

print("🚀 Starting training…")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
print("✅ Training complete. Best model saved to:", MODEL_PATH)

model.save(MODEL_PATH)
print("💾 Final model saved to:", MODEL_PATH)

# Plot
plt.figure(figsize=(8,8))
plt.subplot(2,1,1); plt.plot(history.history["accuracy"], label="Train"); plt.plot(history.history["val_accuracy"], label="Val"); plt.legend(); plt.title("Accuracy")
plt.subplot(2,1,2); plt.plot(history.history["loss"], label="Train"); plt.plot(history.history["val_loss"], label="Val"); plt.legend(); plt.title("Loss")
graph_path = os.path.join(MODEL_DIR, "training_results.png")
plt.tight_layout(); plt.savefig(graph_path)
print(f"📊 Training graphs saved at: {graph_path}")