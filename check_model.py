import json
from pathlib import Path

tfjs_dir = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs")

with open(tfjs_dir / "model.json") as f:
    m = json.load(f)

layers = m["modelTopology"]["model_config"]["config"]["layers"]

# Print all unique class names and any suspicious config keys
class_names = {}
suspicious_keys = {"module", "registered_name", "build_config", "compile_config", "optional"}
issues = []

for layer in layers:
    cn = layer["class_name"]
    class_names[cn] = class_names.get(cn, 0) + 1
    cfg = layer.get("config", {})
    bad = suspicious_keys & set(cfg.keys())
    if bad:
        issues.append(f"  {cn} '{cfg.get('name','')}': has keys {bad}")

print("Layer classes:")
for cn, count in sorted(class_names.items()):
    print(f"  {cn}: {count}")

if issues:
    print(f"\nSuspicious keys found ({len(issues)} layers):")
    for i in issues[:20]:
        print(i)
else:
    print("\nNo suspicious keys found")

# Check weight names - do any still have :0?
wm = m["weightsManifest"][0]["weights"]
with_suffix = [w["name"] for w in wm if ":0" in w["name"]]
print(f"\nWeights with ':0' suffix: {len(with_suffix)}")
if with_suffix:
    print("  Examples:", with_suffix[:3])
