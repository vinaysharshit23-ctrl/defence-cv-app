"""Manual TF.js export using tf.lite + JSON weight packing (works around numpy compat issue)"""
import os, json, struct, base64
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")
tfjs_dir = OUT_DIR / "tfjs"
tfjs_dir.mkdir(parents=True, exist_ok=True)

# Load the saved .keras model
model = keras.models.load_model(str(OUT_DIR / "model.keras"))
print(f"Model loaded, output shape: {model.output_shape}")

# Try the newer tensorflowjs API directly (skip CLI)
try:
    import tensorflowjs as tfjs
    print(f"tensorflowjs version: {tfjs.__version__}")
    tfjs.converters.save_keras_model(model, str(tfjs_dir))
    print("TF.js export succeeded via Python API")
except Exception as e:
    print(f"Python API failed: {e}")
    print("Falling back to manual weight export...")

    # Manual export: save weights as binary + topology JSON
    import struct

    # Get model topology (config)
    topology = {
        "modelTopology": json.loads(model.to_json()),
        "format": "layers-model",
        "generatedBy": "custom-exporter",
        "convertedBy": "manual"
    }

    # Export weights
    weight_data = []
    weight_specs = []
    for layer in model.layers:
        for weight in layer.weights:
            arr = weight.numpy().astype(np.float32)
            weight_specs.append({
                "name": weight.name,
                "shape": list(arr.shape),
                "dtype": "float32"
            })
            weight_data.append(arr.flatten().tobytes())

    all_bytes = b"".join(weight_data)

    # Write weights binary
    with open(tfjs_dir / "group1-shard1of1.bin", "wb") as f:
        f.write(all_bytes)

    topology["weightsManifest"] = [{
        "paths": ["group1-shard1of1.bin"],
        "weights": weight_specs
    }]

    with open(tfjs_dir / "model.json", "w") as f:
        json.dump(topology, f)

    print(f"Manual export: {len(all_bytes)/1024/1024:.1f} MB written")

# Labels
classes = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]
with open(tfjs_dir / "labels.json", "w") as f:
    json.dump({"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, f, indent=2)

print(f"\nFiles in {tfjs_dir}:")
for f in tfjs_dir.iterdir():
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
print("DONE")
