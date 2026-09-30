"""
ASTRA Challenge-02 — Defence Object Classification
MobileNetV2 transfer learning → TF.js export
"""
import os, csv, shutil, math, json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\astra-dataset\ASTRA-Challenge-Starter\challenge-02-vision")
OUT_DIR    = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 8
EPOCHS_FT1 = 10   # train only head
EPOCHS_FT2 = 10   # fine-tune top layers
VAL_SPLIT  = 0.2
SEED       = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"TensorFlow {tf.__version__}")
print(f"Dataset: {BASE}")
print(f"Output:  {OUT_DIR}\n")

# ── Read labels.csv ────────────────────────────────────────────────────────────
label_map = {}
with open(BASE / "labels.csv") as f:
    for row in csv.DictReader(f):
        fname = row["file_name"].split("/")[-1]
        label_map[fname] = row["category"]

classes = sorted(set(label_map.values()))
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Classes ({len(classes)}): {classes}")

# ── Load images ────────────────────────────────────────────────────────────────
images, labels = [], []
for img_path in sorted((BASE / "images").rglob("*")):
    if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
        fname = img_path.name
        if fname not in label_map:
            print(f"  skip (no label): {fname}")
            continue
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            images.append(np.array(img, dtype=np.float32))
            labels.append(class_to_idx[label_map[fname]])
        except Exception as e:
            print(f"  skip (error): {fname} — {e}")

images = np.stack(images)
labels = np.array(labels)
print(f"Loaded {len(images)} images")

# ── Train / val split ─────────────────────────────────────────────────────────
indices = np.arange(len(images))
np.random.shuffle(indices)
split = int(len(indices) * (1 - VAL_SPLIT))
train_idx, val_idx = indices[:split], indices[split:]

# Preprocess for MobileNetV2 (scale to [-1, 1])
def preprocess(x):
    return (x / 127.5) - 1.0

X_train = preprocess(images[train_idx])
X_val   = preprocess(images[val_idx])
y_train = keras.utils.to_categorical(labels[train_idx], len(classes))
y_val   = keras.utils.to_categorical(labels[val_idx],   len(classes))

print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# ── Augmentation ──────────────────────────────────────────────────────────────
data_aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomBrightness(0.15),
    keras.layers.RandomContrast(0.15),
], name="augmentation")

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = data_aug(inputs)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(classes), activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n── Phase 1: Training head ──")
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=EPOCHS_FT1, batch_size=BATCH_SIZE, verbose=1)

# ── Fine-tune top layers ──────────────────────────────────────────────────────
print("\n── Phase 2: Fine-tuning top MobileNetV2 layers ──")
base_model.trainable = True
# Freeze all but last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=EPOCHS_FT2, batch_size=BATCH_SIZE, verbose=1)

# ── Final accuracy ────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nFinal val accuracy: {acc*100:.1f}%  |  loss: {loss:.4f}")

# ── Save Keras model ──────────────────────────────────────────────────────────
keras_path = OUT_DIR / "keras_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save(str(keras_path))
print(f"Keras model saved: {keras_path}")

# ── Export to TF.js ───────────────────────────────────────────────────────────
tfjs_path = OUT_DIR / "tfjs"
os.system(f'tensorflowjs_converter --input_format=keras_saved_model "{keras_path}" "{tfjs_path}"')
print(f"TF.js model exported: {tfjs_path}")

# ── Write class labels JSON for the app ───────────────────────────────────────
labels_json = OUT_DIR / "tfjs" / "labels.json"
with open(labels_json, "w") as f:
    json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
print(f"Labels JSON: {labels_json}")

print("\n✅ Done! Load model in browser app from: model/tfjs/model.json")
