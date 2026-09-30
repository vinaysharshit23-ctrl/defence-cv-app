import os, json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import csv

BASE    = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\astra-dataset\ASTRA-Challenge-Starter\challenge-02-vision")
OUT_DIR = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load data
label_map = {}
with open(BASE / "labels.csv") as f:
    for row in csv.DictReader(f):
        fname = row["file_name"].split("/")[-1]
        label_map[fname] = row["category"]

classes = sorted(set(label_map.values()))
class_to_idx = {c: i for i, c in enumerate(classes)}

images, labels = [], []
for img_path in sorted((BASE / "images").rglob("*")):
    if img_path.is_file() and img_path.suffix.lower() in (".jpg",".jpeg",".png",".webp"):
        fname = img_path.name
        if fname not in label_map: continue
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            images.append(np.array(img, dtype=np.float32))
            labels.append(class_to_idx[label_map[fname]])
        except: pass

images = np.stack(images)
labels_arr = np.array(labels)

indices = np.arange(len(images))
np.random.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

def preprocess(x): return (x / 127.5) - 1.0
X_train = preprocess(images[train_idx])
X_val   = preprocess(images[val_idx])
y_train = keras.utils.to_categorical(labels_arr[train_idx], len(classes))
y_val   = keras.utils.to_categorical(labels_arr[val_idx],   len(classes))

data_aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
], name="augmentation")

base_model = keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False

inputs  = keras.Input(shape=(*IMG_SIZE, 3))
x       = data_aug(inputs)
x       = base_model(x, training=False)
x       = keras.layers.GlobalAveragePooling2D()(x)
x       = keras.layers.Dropout(0.3)(x)
x       = keras.layers.Dense(128, activation="relu")(x)
x       = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(classes), activation="softmax")(x)
model   = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
print("Phase 1: training head (10 epochs)...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=BATCH_SIZE, verbose=0)

base_model.trainable = True
for layer in base_model.layers[:-30]: layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
print("Phase 2: fine-tuning (10 epochs)...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=BATCH_SIZE, verbose=0)

loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Val accuracy: {acc*100:.1f}%  loss: {loss:.4f}")

# Save as SavedModel format (for TF.js converter)
OUT_DIR.mkdir(parents=True, exist_ok=True)
saved_model_path = OUT_DIR / "saved_model"
model.export(str(saved_model_path))
print(f"SavedModel exported: {saved_model_path}")

# Also save .keras for reference
model.save(str(OUT_DIR / "model.keras"))
print("Keras .keras saved")

# Convert to TF.js
tfjs_path = OUT_DIR / "tfjs"
ret = os.system(f'tensorflowjs_converter --input_format=tf_saved_model "{saved_model_path}" "{tfjs_path}"')
print(f"TF.js converter exit: {ret}")

# Write labels JSON
labels_json = OUT_DIR / "tfjs" / "labels.json"
with open(labels_json, "w") as f:
    json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
print(f"Labels saved: {labels_json}")
print("DONE")
