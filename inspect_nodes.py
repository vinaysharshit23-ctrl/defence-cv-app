import json

with open(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs\model.json") as f:
    m = json.load(f)

layers = m["modelTopology"]["model_config"]["config"]["layers"]

# Show inbound_nodes / node_indices for first few layers
for layer in layers[:6]:
    cn = layer.get("class_name")
    cfg = layer.get("config", {})
    in_nodes = layer.get("inbound_nodes", [])
    print(f"{cn} '{cfg.get('name')}': inbound_nodes={json.dumps(in_nodes)[:120]}")
