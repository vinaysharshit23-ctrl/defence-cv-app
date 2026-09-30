"""
Convert Keras 3 model.json to TF.js-compatible Keras 2 format.

Keras 3 inbound_nodes: [{"args": [{"class_name": "__keras_tensor__", "config": {"keras_history": [layer_name, node_idx, tensor_idx]}}]}]
TF.js expects:         [[[layer_name, node_idx, tensor_idx]]]
"""
import json, os

src = r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\defence-cv-app\model\tfjs\model.json"

with open(src) as f:
    m = json.load(f)

def extract_keras_history(tensor_obj):
    """Extract [layer_name, node_idx, tensor_idx] from a __keras_tensor__ object."""
    if isinstance(tensor_obj, dict) and tensor_obj.get("class_name") == "__keras_tensor__":
        h = tensor_obj.get("config", {}).get("keras_history")
        if h and len(h) >= 3:
            return [h[0], h[1], h[2]]
    return None

def convert_inbound_nodes_k3_to_k2(inbound_nodes_k3):
    """Convert Keras 3 inbound_nodes to Keras 2 format."""
    if not inbound_nodes_k3:
        return []
    result = []
    for node in inbound_nodes_k3:
        if isinstance(node, dict) and "args" in node:
            args = node["args"]
            node_inputs = []
            for arg in args:
                if isinstance(arg, list):
                    # Multiple inputs
                    for item in arg:
                        h = extract_keras_history(item)
                        if h:
                            node_inputs.append(h)
                else:
                    h = extract_keras_history(arg)
                    if h:
                        node_inputs.append(h)
            if node_inputs:
                result.append(node_inputs)
        elif isinstance(node, list):
            result.append(node)  # Already Keras 2 format
    return result

def fix_layers_recursive(layers):
    for layer in layers:
        # Convert inbound_nodes
        if "inbound_nodes" in layer:
            layer["inbound_nodes"] = convert_inbound_nodes_k3_to_k2(layer["inbound_nodes"])

        cfg = layer.get("config", {})

        # Fix InputLayer shape fields
        if layer["class_name"] == "InputLayer":
            if "batch_shape" in cfg and "batchInputShape" not in cfg:
                cfg["batchInputShape"] = cfg.pop("batch_shape")
            cfg.pop("optional", None)

        # Recurse into nested Functional/Sequential models
        if "layers" in cfg:
            fix_layers_recursive(cfg["layers"])

        # Fix nested model inbound_nodes too
        if "inbound_nodes" in cfg:
            cfg["inbound_nodes"] = convert_inbound_nodes_k3_to_k2(cfg["inbound_nodes"])

top_layers = m["modelTopology"]["model_config"]["config"]["layers"]
fix_layers_recursive(top_layers)

# Show result
for layer in top_layers[:6]:
    cn = layer.get("class_name")
    name = layer.get("config", {}).get("name", "")
    nodes = layer.get("inbound_nodes", [])
    print(f"{cn} '{name}': {json.dumps(nodes)[:100]}")

with open(src, "w") as f:
    json.dump(m, f)
print("\nmodel.json rewritten with Keras 2 node format")
