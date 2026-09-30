"""Quick test: send a small solid-colour JPEG to Groq vision to verify the API works."""
import requests, json, base64, io, pathlib
from PIL import Image

key = None
for line in pathlib.Path(__file__).parent.joinpath('.env').read_text().splitlines():
    if line.startswith('GROQ_API_KEY='):
        key = line.split('=', 1)[1].strip()

if not key:
    print("No GROQ_API_KEY in .env")
    raise SystemExit(1)

# Create a tiny test image (grey square)
img = Image.new("RGB", (64, 64), color=(128, 128, 128))
buf = io.BytesIO()
img.save(buf, format="JPEG")
img_b64 = base64.b64encode(buf.getvalue()).decode()

resp = requests.post(
    'https://api.groq.com/openai/v1/chat/completions',
    headers={'Authorization': f'Bearer {key}'},
    json={
        'model': 'llama-3.2-11b-vision-preview',
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + img_b64}},
            {'type': 'text', 'text': 'What color is this image? Reply in one word.'}
        ]}],
        'max_tokens': 10,
        'temperature': 0.1
    },
    timeout=20
)
if resp.ok:
    print('Groq vision works! Response:', resp.json()['choices'][0]['message']['content'].strip())
else:
    print('Error:', resp.status_code, resp.text[:400])
