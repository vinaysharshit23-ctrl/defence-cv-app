# RAKSHA DRISHTI — रक्षा दृष्टि

**Defence Object Classification & Intelligence System**

A real-time AI-powered military hardware identification system that classifies defence objects from images using a two-stage hybrid AI pipeline — fast on-device TFLite inference followed by GPT-4o Vision for specific model identification.

🌐 **Live Demo:** [web-production-1e9b2.up.railway.app](https://web-production-1e9b2.up.railway.app)

---

## What It Does

Upload or drop any image of military hardware and the system will:

1. **Classify the category** (Aircraft, Helicopter, Tank, Naval, Drone) in under 200ms using an on-device TFLite model
2. **Identify the specific model** (e.g. Su-30 MKI, Leopard 2A7, IAI Heron) using GPT-4o Vision via OpenRouter
3. **Explain the identification** — a 20–35 word natural-language reason citing the visual features observed
4. **Show top-3 predictions** with confidence scores
5. **Flag low-confidence results** with an amber warning badge

---

## Features

| Feature | Description |
|---|---|
| **Two-stage AI pipeline** | TFLite for speed (<200ms), GPT-4o for accuracy |
| **5 defence categories** | Aircraft, Helicopter, Ground Vehicle, Naval Vessel, Drone/UAV |
| **Top-3 predictions** | Primary ID + 2 AI alternative guesses |
| **Visual explanation** | Why the model identified what it did |
| **Image preprocessing** | RGBA→RGB conversion, resize 224×224, normalise [−1,1] — shown in UI |
| **Low-confidence warning** | Triggers when score <55% or top-2 spread <15% |
| **Graceful error handling** | Readable banners for bad files, oversized images, corrupt data |
| **Webcam support** | Real-time live camera analysis |
| **URL input** | Analyse any publicly accessible image URL |
| **Tactical HUD overlay** | Canvas-based targeting frame and result overlay on the image |
| **Model justification** | "Why This Model?" sidebar explaining the hybrid architecture choice |

---

## Tech Stack

**Backend**
- Python 3.11 (stdlib `http.server` — no Flask)
- TFLite Runtime — MobileNetV2 trained on 5 defence classes
- OpenRouter API → GPT-4o Vision for specific identification
- PIL/Pillow for image preprocessing

**Frontend**
- Vanilla HTML / CSS / JavaScript (no frameworks)
- Canvas API for HUD overlay rendering
- Dark military terminal UI theme

**Deployment**
- Railway.app — auto-deploys on every push to `master`
- GitHub — [vinaysharshit23-ctrl/defence-cv-app](https://github.com/vinaysharshit23-ctrl/defence-cv-app)

---

## Project Structure

```
defence-cv-app/
├── server.py          # Backend — TFLite inference + OpenRouter vision API
├── app.js             # Frontend logic — upload, webcam, chip rendering, HUD
├── index.html         # UI — intro page, app shell, sidebar
├── style.css          # Dark military terminal theme
├── start.bat          # One-click local launcher (Windows)
├── requirements.txt   # Python dependencies
├── Procfile           # Railway/Render deployment config
├── .python-version    # Pin Python 3.11 for Railway
├── .gitignore         # Excludes .env, model weights, __pycache__
└── model/
    └── model.tflite   # Trained MobileNetV2 TFLite model (2.6 MB)
```

---

## Running Locally

**Prerequisites:** Python 3.11, pip

```bash
# 1. Clone the repo
git clone https://github.com/vinaysharshit23-ctrl/defence-cv-app.git
cd defence-cv-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env with your OpenRouter API key
echo OPENROUTER_API_KEY=sk-or-v1-... > .env

# 4. Start the server
python server.py
# or on Windows:
start.bat
```

Open [http://127.0.0.1:8383](http://127.0.0.1:8383) in your browser.

Get a free OpenRouter API key at [openrouter.ai/keys](https://openrouter.ai/keys).

---

## Deployment (Railway)

1. Fork/push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repo — Railway auto-detects the `Procfile`
4. Add environment variable: `OPENROUTER_API_KEY` = your key
5. Railway deploys and gives you a public URL

Every push to `master` triggers an automatic redeploy.

---

## AI Architecture

```
Image Input
    │
    ▼
Preprocessing (PIL)
  • RGBA → RGB conversion
  • Resize to 224×224
  • Normalise to [−1, 1]
    │
    ├──► TFLite (MobileNetV2) ──► Category + Top-5 scores  [<200ms]
    │
    └──► GPT-4o Vision (OpenRouter) ──► Specific model + explanation  [1-3s]
                │
                ▼
         Top-3 Results + HUD Overlay
```

**Why this hybrid?**
- TFLite gives instant on-device speed without an internet round-trip
- GPT-4o handles fine-grained identification that no compact model can match (e.g. Su-30 MKI vs MiG-29)
- Graceful fallback to heuristic when the API is unavailable

---

## Supported Categories & Examples

| Category | Examples |
|---|---|
| ✈ Fighter Aircraft | Su-30 MKI, Rafale, F-22 Raptor, Mirage 2000, Eurofighter Typhoon, HAL TEJAS |
| 🚁 Military Helicopter | AH-64 Apache, Mi-28 Havoc, Dhruv ALH, CH-47 Chinook, UH-60 Black Hawk |
| 🛡 Ground Vehicle | T-90 Bhishma, Leopard 2A7, T-14 Armata, Arjun MBT, BMP-2 IFV |
| ⚓ Naval Vessel | Destroyers, frigates, aircraft carriers, submarines |
| 🔭 Drone / UAV | IAI Heron, MQ-9 Reaper, Bayraktar TB2, Global Hawk |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for GPT-4o Vision |
| `PORT` | No | Server port (default 8383; Railway injects this automatically) |

**Never commit your `.env` file.** It is excluded by `.gitignore`.

---

## License

MIT License — free to use, modify and distribute.

---

*Built with Kiro AI · Deployed on Railway · Powered by GPT-4o Vision + TFLite*
