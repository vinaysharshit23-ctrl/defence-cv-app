import json

with open(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs\model.json") as f:
    m = json.load(f)

problems = []

def scan_layers(layers, path=""):
    for layer in layers:
        cn  = layer.get("class_name", "?")
        cfg = layer.get("config", {})
        name = cfg.get("name", "")
        full = f"{path}/{cn}[{name}]"

        if cn == "InputLayer":
            if "batchInputShape" not in cfg:
                problems.append(f"MISSING batchInputShape: {full}  keys={list(cfg.keys())}")
            if "optional" in cfg:
                del cfg["optional"]
                problems.append(f"REMOVED optional: {full}")

        # Fix batch_shape in nested layers too
        if "batch_shape" in cfg and "batchInputShape" not in cfg:
            cfg["batchInputShape"] = cfg.pop("batch_shape")
            problems.append(f"FIXED batch_shape->batchInputShape: {full}")

        if "layers" in cfg:
            scan_layers(cfg["layers"], full)

scan_layers(m["modelTopology"]["model_config"]["config"]["layers"])

if problems:
    print("Issues found/fixed:")
    for p in problems:
        print(" ", p)
    with open(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs\model.json", "w") as f:
        json.dump(m, f)
    print("model.json rewritten")
else:
    print("All InputLayers have batchInputShape. No issues.")
