"""Patch numpy compat and export via tensorflowjs Python API"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Patch numpy before importing tensorflowjs
import numpy as np
if not hasattr(np, 'object'):  np.object  = object
if not hasattr(np, 'bool'):    np.bool    = bool
if not hasattr(np, 'int'):     np.int     = int
if not hasattr(np, 'float'):   np.float   = float
if not hasattr(np, 'complex'): np.complex = complex

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

OUT_DIR  = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
tfjs_dir = OUT_DIR / "tfjs"

# Remove old tfjs dir and recreate
import shutil
if tfjs_dir.exists(): shutil.rmtree(tfjs_dir)
tfjs_dir.mkdir(parents=True)

# Load model
model = keras.models.load_model(str(OUT_DIR / "model.keras"))
print(f"Model loaded: {model.output_shape}")

# Export via Python API (now numpy is patched)
tfjs.converters.save_keras_model(model, str(tfjs_dir))
print("Export succeeded")

# Write labels
import json
classes = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]
with open(tfjs_dir / "labels.json", "w") as f:
    json.dump({"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, f, indent=2)

print("Files:")
for f in sorted(tfjs_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
