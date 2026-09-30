"""Find which OpenRouter vision models are actually responding right now."""
import requests, base64, json
from pathlib import Path

import os
KEY = os.getenv("OPENROUTER_API_KEY", "")

# Small test image
DATASET = Path(r"C:\Users\S. HARSHIT VINAY\.gemini\antigravity\scratch\astra-dataset\ASTRA-Challenge-Starter\challenge-02-vision\images\aircraft")
imgs = list(DATASET.iterdir())
img_bytes = imgs[0].read_bytes()[:50000]  # cap at 50KB for speed
img_b64 = base64.b64encode(img_bytes).decode()

# Vision-capable models to test, cheapest first
CANDIDATES = [
    ("qwen/qwen3.8-omni-flash",          0.00000015),
    ("qwen/qwen3.8-flash",               0.00000015),
    ("google/gemma-4-26b-a4b-it",        0.00000009),
    ("google/gemma-4-31b-it",            0.00000009),
    ("qwen/qwen3.7-flash",               0.00000003),
    ("qwen/qwen3.6-flash",               0.0000001875),
    ("meta-llama/llama-4-scout",         0.00000008),
    ("meta-llama/llama-4-maverick",      0.00000018),
    ("qwen/qwen3.8-27b:free",            0),
    ("google/gemma-4-31b-it:free",       0),
]

PROMPT = "What military aircraft is in this image? Reply with only the specific name like 'F-22 Raptor'."

print("Testing vision models (first working one wins)...\n")
for model, price in CANDIDATES:
    try:
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
                "max_tokens": 30,
                "temperature": 0.1,
            },
            timeout=15
        )
        if resp.status_code == 429:
            print(f"  RATE-LIMITED: {model}")
        elif resp.ok:
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            cost_str = f"${price * 1000:.6f}/1k tokens" if price > 0 else "FREE"
            print(f"  WORKS: {model} ({cost_str})")
            print(f"  Answer: {answer}")
            break
        else:
            print(f"  ERROR {resp.status_code}: {model} - {resp.text[:200]}")
    except Exception as e:
        print(f"  TIMEOUT/ERROR: {model} - {e}")
