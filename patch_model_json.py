"""
Patch model.json for TF.js compatibility:
  - InputLayer: batch_shape -> batchInputShape
  - Any other Keras 3 -> TF.js field name differences
"""
import json
from pathlib import Path

tfjs_dir = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs")
model_json_path = tfjs_dir / "model.json"

with open(model_json_path) as f:
    m = json.load(f)

layers = m["modelTopology"]["model_config"]["config"]["layers"]
patched = 0

for layer in layers:
    cfg = layer.get("config", {})

    # Fix InputLayer: batch_shape -> batchInputShape
    if layer["class_name"] == "InputLayer":
        if "batch_shape" in cfg and "batchInputShape" not in cfg:
            cfg["batchInputShape"] = cfg.pop("batch_shape")
            patched += 1
            print(f"  Patched InputLayer '{cfg.get('name')}': batchInputShape = {cfg['batchInputShape']}")

    # Fix any layer that uses batch_input_shape -> batchInputShape
    if "batch_input_shape" in cfg and "batchInputShape" not in cfg:
        cfg["batchInputShape"] = cfg.pop("batch_input_shape")
        patched += 1

    # TF.js doesn't understand 'optional' field on InputLayer - remove it
    if layer["class_name"] == "InputLayer" and "optional" in cfg:
        del cfg["optional"]

# Also ensure the weight names match what TF.js will look up
# TF.js uses the weight name without the `:0` suffix if present
weights_manifest = m.get("weightsManifest", [{}])[0].get("weights", [])
for w in weights_manifest:
    if w["name"].endswith(":0"):
        w["name"] = w["name"][:-2]

with open(model_json_path, "w") as f:
    json.dump(m, f)

print(f"\nPatched {patched} layer(s)")
print(f"Weight specs: {len(weights_manifest)}")
print("model.json rewritten")
