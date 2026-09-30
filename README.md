# Raksha Drishti — रक्षा दृष्टि

**Defence Object Classification & Intelligence System**

Real-time military object identification using a TFLite category classifier + GPT-4o vision for specific naming.

## Features

- 5 defence categories: Aircraft, Helicopter, Drone, Military Vehicle, Naval
- Specific identification via OpenRouter vision API (GPT-4o-mini)
- Top-3 predictions with confidence scores
- Image preprocessing pipeline (resize, normalise, format handling)
- Low-confidence warning for uncertain predictions
- Cinematic intro screen, dark military HUD UI

## Stack

| Layer | Tech |
|---|---|
| Frontend | Vanilla JS + HTML5 Canvas |
| Backend | Python `http.server` |
| ML Model | TFLite (MobileNetV2, 5-class) |
| Vision API | OpenRouter → GPT-4o-mini |

## Run Locally

```bash
pip install -r requirements.txt
# Create .env with: OPENROUTER_API_KEY=sk-or-...
python server.py
# Open http://127.0.0.1:8383
```

## Deploy

Set `OPENROUTER_API_KEY` as an environment variable on your host. The server reads `PORT` from the environment automatically.
