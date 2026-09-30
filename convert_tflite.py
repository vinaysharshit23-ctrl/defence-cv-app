import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

OUT_DIR = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model")

# Load model
model = keras.models.load_model(str(OUT_DIR / "model.keras"))

# Build inference-only model (strip augmentation)
inputs = model.input
x = inputs
skip_aug = True
for layer in model.layers[1:]:
    if skip_aug and layer.__class__.__name__ == "Sequential":
        skip_aug = False
        continue
    x = layer(x)
inference_model = keras.Model(inputs=inputs, outputs=x, name="defence_inference")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # float16 quantization -> smaller + faster
tflite_model = converter.convert()

tflite_path = OUT_DIR / "model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"TFLite model: {size_kb:.0f} KB -> {tflite_path}")

# Quick sanity check
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()
print(f"Input:  {inp[0]['shape']}  dtype={inp[0]['dtype']}")
print(f"Output: {out[0]['shape']}  dtype={out[0]['dtype']}")

# Run dummy inference
dummy = np.zeros((1,224,224,3), dtype=np.float32)
interpreter.set_tensor(inp[0]['index'], dummy)
interpreter.invoke()
result = interpreter.get_tensor(out[0]['index'])
print(f"Dummy output: {result[0].round(3)}")
print("DONE")
