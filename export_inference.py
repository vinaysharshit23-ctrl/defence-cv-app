"""
Export inference-only MobileNetV2 model for TF.js.
Strips augmentation layers (RandomFlip, RandomRotation, RandomZoom)
and exports just the feature extractor + classifier head.
"""
import os, json
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

OUT_DIR  = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
tfjs_dir = OUT_DIR / "tfjs"

# Load full trained model
full_model = keras.models.load_model(str(OUT_DIR / "model.keras"))
print("Full model layers:")
for l in full_model.layers:
    print(f"  {l.name}  ({l.__class__.__name__})")

# Build inference model: skip the augmentation Sequential, go straight to MobileNetV2 + head
# The architecture is: Input -> augmentation (Sequential) -> mobilenetv2 -> GAP -> Dropout -> Dense -> Dropout -> Dense(5)
# For inference we skip augmentation entirely

inputs = full_model.input   # keras.Input shape (None,224,224,3)

# Walk layers and skip the augmentation Sequential
x = inputs
skip_aug = True
for layer in full_model.layers[1:]:   # skip InputLayer
    if skip_aug and layer.__class__.__name__ == "Sequential":
        skip_aug = False
        continue   # skip augmentation
    x = layer(x)

inference_model = keras.Model(inputs=inputs, outputs=x, name="defence_inference")
inference_model.summary(print_fn=lambda s: None)

# Verify shapes
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
out = inference_model.predict(dummy, verbose=0)
print(f"\nInference model: input {inference_model.input_shape}  output {inference_model.output_shape}")
print(f"Dummy prediction: {out[0].round(3)}")

# ── Export weights ─────────────────────────────────────────────────────────
import shutil
if tfjs_dir.exists(): shutil.rmtree(tfjs_dir)
tfjs_dir.mkdir(parents=True)

weight_specs = []
weight_bytes_list = []
for layer in inference_model.layers:
    for w in layer.weights:
        arr = w.numpy().astype(np.float32)
        weight_specs.append({"name": w.name, "shape": list(arr.shape), "dtype": "float32"})
        weight_bytes_list.append(arr.tobytes())

all_bytes = b"".join(weight_bytes_list)
shard = "group1-shard1of1.bin"
with open(tfjs_dir / shard, "wb") as f:
    f.write(all_bytes)
print(f"Weights: {len(all_bytes)/1024/1024:.1f} MB  ({len(weight_specs)} tensors)")

# ── Build model.json ───────────────────────────────────────────────────────
model_cfg = json.loads(inference_model.to_json())

def fix_layers(layers):
    for layer in layers:
        cfg = layer.get("config", {})
        if "batch_shape" in cfg:
            cfg["batchInputShape"] = cfg.pop("batch_shape")
        if "optional" in cfg:
            del cfg["optional"]
        if "layers" in cfg:
            fix_layers(cfg["layers"])

fix_layers(model_cfg["config"]["layers"])

model_json = {
    "modelTopology": {
        "keras_version": "2.15.0",
        "backend": "tensorflow",
        "model_config": model_cfg
    },
    "format": "layers-model",
    "generatedBy": "inference-export",
    "convertedBy": "custom",
    "weightsManifest": [{"paths": [shard], "weights": weight_specs}]
}

with open(tfjs_dir / "model.json", "w") as f:
    json.dump(model_json, f)

# ── Labels ─────────────────────────────────────────────────────────────────
classes = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]
with open(tfjs_dir / "labels.json", "w") as f:
    json.dump({"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, f, indent=2)

print("\nExported files:")
for fp in sorted(tfjs_dir.iterdir()):
    print(f"  {fp.name}: {fp.stat().st_size/1024:.0f} KB")

# Verify no augmentation layers in exported JSON
with open(tfjs_dir / "model.json") as f:
    raw = f.read()
aug_layers = ["RandomFlip", "RandomRotation", "RandomZoom", "RandomBrightness", "RandomContrast"]
for al in aug_layers:
    if al in raw:
        print(f"WARNING: {al} still present in model.json!")
    else:
        print(f"OK: {al} absent")
print("DONE")
