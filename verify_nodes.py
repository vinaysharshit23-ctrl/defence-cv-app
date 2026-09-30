import json

with open(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs\model.json") as f:
    m = json.load(f)

top_layers = m["modelTopology"]["model_config"]["config"]["layers"]

# Find MobileNetV2 nested layers
mobilenet = next(l for l in top_layers if l.get("class_name") == "Functional")
nested = mobilenet.get("config", {}).get("layers", [])
print(f"MobileNetV2 has {len(nested)} nested layers")

# Check first 5 non-InputLayer for node format
count_k3 = 0
count_k2 = 0
for layer in nested:
    for node in layer.get("inbound_nodes", []):
        if isinstance(node, dict) and "args" in node:
            count_k3 += 1
        else:
            count_k2 += 1

print(f"Keras 3 format nodes remaining: {count_k3}")
print(f"Keras 2 format nodes: {count_k2}")

# Show a few examples
for layer in nested[1:4]:
    cn = layer.get("class_name")
    name = layer.get("config",{}).get("name","")
    nodes = layer.get("inbound_nodes",[])
    print(f"  {cn} '{name}': {json.dumps(nodes)[:80]}")
