"""
Build a valid TF.js layers-model export WITHOUT tensorflowjs package.
Uses tf.lite + manual weight packing in the format TF.js expects.
"""
import os, json, struct
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

OUT_DIR  = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
tfjs_dir = OUT_DIR / "tfjs"

import shutil
if tfjs_dir.exists(): shutil.rmtree(tfjs_dir)
tfjs_dir.mkdir(parents=True)

# Load model
model = keras.models.load_model(str(OUT_DIR / "model.keras"))
print(f"Model loaded. Input: {model.input_shape}  Output: {model.output_shape}")

# ── Collect weights ────────────────────────────────────────────────────────
weight_specs = []
weight_bytes_list = []

for layer in model.layers:
    for w in layer.weights:
        arr = w.numpy().astype(np.float32)
        # TF.js dtype map
        dtype_map = {np.float32: 'float32', np.int32: 'int32', np.bool_: 'bool'}
        dtype_str = 'float32'

        weight_specs.append({
            "name": w.name,
            "shape": list(arr.shape),
            "dtype": dtype_str,
        })
        weight_bytes_list.append(arr.tobytes())

all_bytes = b"".join(weight_bytes_list)
shard_name = "group1-shard1of1.bin"
with open(tfjs_dir / shard_name, "wb") as f:
    f.write(all_bytes)
print(f"Weights: {len(all_bytes)/1024/1024:.1f} MB  ({len(weight_specs)} tensors)")

# ── Build model.json in TF.js layers-model format ─────────────────────────
# TF.js expects: { modelTopology: { keras_version, backend, model_config }, weightsManifest: [...] }
# model_config must be the OLD Keras v1/v2 serialization format

def keras_layer_to_tfjs_config(layer):
    """Convert a layer config to something TF.js understands."""
    cfg = layer.get_config()
    return {
        "class_name": layer.__class__.__name__,
        "name": layer.name,
        "trainable": layer.trainable,
        "config": cfg
    }

# Get the model config in the format TF.js expects
model_config = json.loads(model.to_json())

# TF.js model.json top-level structure
model_json = {
    "modelTopology": {
        "keras_version": "2.15.0",
        "backend": "tensorflow",
        "model_config": model_config
    },
    "format": "layers-model",
    "generatedBy": "keras v2 manual",
    "convertedBy": "custom-exporter",
    "weightsManifest": [
        {
            "paths": [shard_name],
            "weights": weight_specs
        }
    ]
}

with open(tfjs_dir / "model.json", "w") as f:
    json.dump(model_json, f)

# Labels
classes = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]
with open(tfjs_dir / "labels.json", "w") as f:
    json.dump({"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, f, indent=2)

print("\nExported files:")
for f in sorted(tfjs_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
print("\nDONE")
