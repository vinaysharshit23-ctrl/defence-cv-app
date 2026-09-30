import requests, base64, json, os
from pathlib import Path

KEY = os.getenv("OPENROUTER_API_KEY", "")

# Find a test image
DATASET = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\astra-dataset\ASTRA-Challenge-Starter\challenge-02-vision\images")
img_path = None
for cls in ["aircraft", "helicopter", "drone"]:
    folder = DATASET / cls
    if folder.exists():
        imgs = list(folder.iterdir())
        if imgs:
            img_path = imgs[0]
            break

if not img_path:
    print("No dataset images found")
    exit(1)

print(f"Testing with: {img_path}")
img_bytes = img_path.read_bytes()
img_b64 = base64.b64encode(img_bytes).decode()

PROMPT = (
    "You are a military hardware recognition expert. "
    "Identify the specific military aircraft, helicopter, drone, ground vehicle, "
    "or naval vessel in this image. "
    "Reply with ONLY the specific name/designation, for example: "
    "MiG-29 Fulcrum, AH-64 Apache, T-90 Main Battle Tank, F-22 Raptor. "
    "No explanations, just the name."
)

MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "qwen/qwen3.8-27b:free",
]

for model in MODELS:
    print(f"\n--- {model} ---")
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {KEY}",
            "HTTP-Referer": "http://localhost:8383",
            "X-Title": "Defence CV",
        },
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }],
            "max_tokens": 40,
            "temperature": 0.1,
        },
        timeout=30
    )
    print(f"Status: {resp.status_code}")
    if resp.ok:
        answer = resp.json()["choices"][0]["message"]["content"]
        print(f"Response: {answer}")
    else:
        try:
            err = resp.json()
            print(f"Error: {json.dumps(err)[:400]}")
        except Exception:
            print(f"Raw: {resp.text[:400]}")
