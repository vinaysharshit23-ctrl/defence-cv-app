"""End-to-end test: run inference on one image from each class."""
import sys, os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.chdir(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app")
sys.path.insert(0, os.getcwd())

import server
from pathlib import Path

dataset = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\astra-dataset\ASTRA-Challenge-Starter\challenge-02-vision\images")

classes = ["aircraft", "helicopter", "military-vehicle", "naval", "drone"]
for cls in classes:
    folder = dataset / cls
    img_path = list(folder.iterdir())[0]
    print(f"\nTesting [{cls}]: {img_path.name}")
    result = server.run_inference(img_path.read_bytes())
    top = result[0]
    print(f"  Category:  {top['class']} ({top['score']:.0%})")
    print(f"  Specific:  {top['specific_name']}")
    print(f"  Note:      {top['specific_note']}")
